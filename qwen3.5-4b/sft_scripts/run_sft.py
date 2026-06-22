import os
import json
import glob
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import Dataset


def load_jsonl(path):
    records = []
    for f in sorted(glob.glob(os.path.join(path, "*.jsonl"))):
        with open(f, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
    return records


def format_chat(example, tokenizer):
    return {"text": tokenizer.apply_chat_template(example["messages"], tokenize=False)}


def main():
    model_id = os.environ.get("SM_HP_MODEL_ID", "Qwen/Qwen3.5-4B")
    epochs = int(os.environ.get("SM_HP_EPOCHS", "3"))
    batch_size = int(os.environ.get("SM_HP_PER_DEVICE_TRAIN_BATCH_SIZE", "2"))
    grad_accum = int(os.environ.get("SM_HP_GRADIENT_ACCUMULATION_STEPS", "4"))
    lr = float(os.environ.get("SM_HP_LEARNING_RATE", "2e-4"))
    lora_r = int(os.environ.get("SM_HP_LORA_R", "16"))
    lora_alpha = int(os.environ.get("SM_HP_LORA_ALPHA", "32"))
    lora_dropout = float(os.environ.get("SM_HP_LORA_DROPOUT", "0.05"))
    max_seq_length = int(os.environ.get("SM_HP_MAX_SEQ_LENGTH", "512"))
    use_bf16 = os.environ.get("SM_HP_BF16", "True").lower() == "true"
    use_grad_ckpt = os.environ.get("SM_HP_GRADIENT_CHECKPOINTING", "True").lower() == "true"

    training_dir = os.environ.get("SM_CHANNEL_TRAINING", "/opt/ml/input/data/training")
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")

    print(f"Loading training data from {training_dir}")
    records = load_jsonl(training_dir)
    print(f"Loaded {len(records)} examples")

    print(f"Loading tokenizer: {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dataset = Dataset.from_list(records)
    dataset = dataset.map(lambda ex: format_chat(ex, tokenizer), remove_columns=dataset.column_names)

    print(f"Loading model: {model_id} (4-bit quantized)")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if use_bf16 else torch.float16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    training_args = SFTConfig(
        output_dir="/tmp/sft_output",
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        bf16=use_bf16,
        fp16=not use_bf16,
        logging_steps=5,
        save_strategy="epoch",
        gradient_checkpointing=use_grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_grad_ckpt else {},
        max_seq_length=max_seq_length,
        dataset_text_field="text",
        warmup_ratio=0.1,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        tokenizer=tokenizer,
    )

    print("Starting training...")
    trainer.train()
    print("Training complete!")

    # Save adapter
    adapter_dir = "/tmp/adapter"
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    # Merge adapter into base model
    del model
    del trainer
    torch.cuda.empty_cache()

    print("Reloading base model in fp16 for merge...")
    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )

    from peft import PeftModel
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()

    merged_dir = "/tmp/merged"
    merged_model.save_pretrained(merged_dir, safe_serialization=True)
    tokenizer.save_pretrained(merged_dir)
    print("Merge complete!")

    # Save merged safetensors to output (guaranteed working artifact)
    import shutil
    merged_output = os.path.join(output_dir, "merged_model")
    os.makedirs(merged_output, exist_ok=True)
    for f in os.listdir(merged_dir):
        shutil.copy2(os.path.join(merged_dir, f), merged_output)
    print(f"Merged safetensors saved to {merged_output}")

    # Attempt GGUF conversion (best-effort -- llama.cpp may not fully support qwen3_5 yet)
    import subprocess
    import sys

    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "numpy", "sentencepiece"])
        subprocess.check_call(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama_cpp"])
        subprocess.check_call([sys.executable, "-m", "pip", "install", "cmake"])

        os.makedirs("/tmp/llama_cpp/build", exist_ok=True)
        subprocess.check_call(["cmake", "-B", "/tmp/llama_cpp/build", "-S", "/tmp/llama_cpp",
                               "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"])
        subprocess.check_call(["cmake", "--build", "/tmp/llama_cpp/build", "--target", "llama-quantize", "-j4"])

        # Convert to fp16 GGUF first
        gguf_fp16 = "/tmp/model-fp16.gguf"
        subprocess.check_call([sys.executable, "/tmp/llama_cpp/convert_hf_to_gguf.py",
                               merged_dir, "--outfile", gguf_fp16, "--outtype", "f16"])

        # Quantize to Q4_K_M
        gguf_q4 = os.path.join(output_dir, "model-q4_k_m.gguf")
        subprocess.check_call(["/tmp/llama_cpp/build/bin/llama-quantize", gguf_fp16, gguf_q4, "Q4_K_M"])

        # Also produce Q8_0 for future re-quantization
        gguf_q8 = os.path.join(output_dir, "model-q8_0.gguf")
        subprocess.check_call(["/tmp/llama_cpp/build/bin/llama-quantize", gguf_fp16, gguf_q8, "Q8_0"])

        os.remove(gguf_fp16)

        print(f"Q4_K_M: {os.path.getsize(gguf_q4) / 1024 / 1024:.0f} MB")
        print(f"Q8_0:   {os.path.getsize(gguf_q8) / 1024 / 1024:.0f} MB")
    except Exception as e:
        print(f"WARNING: GGUF conversion failed: {e}")
        print("Merged safetensors are still available for local conversion.")

    # Copy tokenizer to output root
    shutil.copy(os.path.join(merged_dir, "tokenizer_config.json"), output_dir)
    shutil.copy(os.path.join(merged_dir, "tokenizer.json"), output_dir)

    print("All done!")


if __name__ == "__main__":
    main()
