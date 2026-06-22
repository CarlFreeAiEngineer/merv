# Mervin/Mervis -- Qwen3.5-4B Fine-tuned Chatbot (WebGPU)

A dual-personality chatbot running entirely in the browser via WebGPU.
Fine-tuned on Qwen3.5-4B (4B params, Apache 2.0) using SageMaker QLoRA.

- **Mervin** (bot-sad.png): sardonic pessimist, wraps correct answers in dry wit
- **Mervis** (bot-happy.png): relentless optimist, celebrates the smallest progress

## Overview

This guide walks through the full pipeline:
1. Prepare training data (CSV -> JSONL)
2. Fine-tune on SageMaker with QLoRA
3. Merge adapter + quantize to GGUF -- all in one SageMaker job
4. Download artifacts locally
5. Build a browser-based chat UI with WebGPU inference

## Prerequisites

- AWS account with SageMaker access (ml.g5.xlarge or ml.g5.2xlarge quota)
- IAM role with SageMaker + S3 permissions
- S3 bucket for training data and output
- Python with `uv` (for running scripts with inline dependencies)
- Training data: 262 Mervin/Mervis conversation pairs from
  https://github.com/freeideas/mervis/blob/main/mervin_mervis_finetune.csv

## Step 1: Prepare Training Data

Convert the CSV to JSONL in chat-completion format. Each example should look like:

```json
{
  "messages": [
    {"role": "system", "content": "You are a dual-personality assistant. For every response, you reply as two characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit and existential weariness) and Mervis (a relentlessly cheerful optimist who celebrates even the smallest progress). Format your response with <Mervin>...</Mervin> followed by <Mervis>...</Mervis>."},
    {"role": "user", "content": "What is 3+3?"},
    {"role": "assistant", "content": "<Mervin>Oh, the crushing banality of arithmetic...</Mervin>\n<Mervis>Hooray! 3 plus 3 equals 6!</Mervis>"}
  ]
}
```

Upload the JSONL file(s) to S3:
```bash
aws s3 cp train.jsonl s3://YOUR_BUCKET/qwen3.5-4b-ft/data/
```

## Step 2: Create the SageMaker Training Script

Create `sft_scripts/run_sft.py` -- this runs inside the SageMaker container:

```python
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

    # Convert to GGUF Q4_K_M
    import subprocess
    import sys

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

    # Copy tokenizer for reference
    import shutil
    shutil.copy(os.path.join(merged_dir, "tokenizer_config.json"), output_dir)
    shutil.copy(os.path.join(merged_dir, "tokenizer.json"), output_dir)

    print(f"Q4_K_M: {os.path.getsize(gguf_q4) / 1024 / 1024:.0f} MB")
    print(f"Q8_0:   {os.path.getsize(gguf_q8) / 1024 / 1024:.0f} MB")
    print("All done!")

if __name__ == "__main__":
    main()
```

## Step 3: Launch the SageMaker Job

Create `train_qwen3.5.py` (launcher script that runs on your local machine):

```python
#!/usr/bin/env uvrun
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "sagemaker>=2.200.0,<3.0.0",
#     "boto3",
# ]
# ///

import sys
import os
import sagemaker
from sagemaker.huggingface import HuggingFace

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

sess = sagemaker.Session()
role = "YOUR_SAGEMAKER_EXECUTION_ROLE_ARN"
bucket = "YOUR_S3_BUCKET"

training_data_s3 = f"s3://{bucket}/qwen3.5-4b-ft/data/"
output_s3 = f"s3://{bucket}/qwen3.5-4b-ft/output/"

hyperparameters = {
    "model_id": "Qwen/Qwen3.5-4B",
    "epochs": 3,
    "per_device_train_batch_size": 2,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "max_seq_length": 512,
    "bf16": True,
    "gradient_checkpointing": True,
}

huggingface_estimator = HuggingFace(
    entry_point="run_sft.py",
    source_dir="./sft_scripts",
    instance_type="ml.g5.xlarge",       # 24GB A10G -- sufficient for 4B model
    instance_count=1,
    role=role,
    transformers_version="4.49.0",
    pytorch_version="2.5.1",
    py_version="py311",
    hyperparameters=hyperparameters,
    output_path=output_s3,
    base_job_name="qwen3.5-4b-mervis",
    max_run=7200,
    volume_size=100,
    environment={
        "HUGGING_FACE_HUB_TOKEN": "YOUR_HF_TOKEN",  # needed to download gated models
    },
)

print("Starting training job...")
print(f"  Model: Qwen/Qwen3.5-4B")
print(f"  Data: {training_data_s3}")
print(f"  Instance: ml.g5.xlarge (24GB A10G)")
print(f"  Output: {output_s3}")
print(f"  Pipeline: QLoRA train -> merge -> GGUF Q4_K_M + Q8_0")

huggingface_estimator.fit({"training": training_data_s3}, wait=False)

print(f"\nTraining job started: {huggingface_estimator.latest_training_job.name}")
print("Monitor at: https://console.aws.amazon.com/sagemaker/home#/jobs")
```

Run it:
```bash
uv run --script ./train_qwen3.5.py
```

## Step 4: Monitor and Download

Check status:
```bash
aws sagemaker describe-training-job \
  --training-job-name YOUR_JOB_NAME \
  --query '{Status: TrainingJobStatus, Secondary: SecondaryStatus}'
```

Expected timeline (~20-30 min total):
- Downloading: ~2 min (pulls container + model weights)
- Training: ~10 min (3 epochs, 262 examples)
- GGUF conversion: ~5 min (merge + quantize)
- Uploading: ~3 min (uploads model.tar.gz to S3)

Once status is `Completed`, download:
```bash
aws s3 cp s3://YOUR_BUCKET/qwen3.5-4b-ft/output/YOUR_JOB_NAME/output/model.tar.gz .
tar -xzf model.tar.gz
```

You should get:
- `model-q4_k_m.gguf` -- ~2.5GB, the WebGPU deployment target
- `model-q8_0.gguf` -- ~4.5GB, keep locally for re-quantization
- `tokenizer.json`
- `tokenizer_config.json`

## Step 5: Local Re-quantization (if needed)

If you need a different quantization level and no longer have SageMaker access:

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp
cmake -B /tmp/llama_cpp/build -S /tmp/llama_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama_cpp/build --target llama-quantize -j4

/tmp/llama_cpp/build/bin/llama-quantize model-q8_0.gguf model-q5_k_m.gguf Q5_K_M
```

## Step 6: Build the WebGPU Chat UI

See `../phi4mini/index.html` or `../gemma4e4b/index.html` for reference.
The approach is the same regardless of model:
1. Use web-llm (MLC) to load the GGUF in-browser via WebGPU
2. Parse `<Mervin>...</Mervin>` and `<Mervis>...</Mervis>` tags from output
3. Render as styled chat bubbles with character icons

## Model Details

| Property | Value |
|----------|-------|
| Base model | Qwen/Qwen3.5-4B |
| License | Apache 2.0 |
| Parameters | ~4B |
| Fine-tune method | QLoRA (rank 16, alpha 32, 4-bit NF4) |
| Training data | 262 Mervin/Mervis conversation pairs |
| Instance type | ml.g5.xlarge (24GB A10G) |
| Expected training time | ~10 min |
| Expected Q4_K_M size | ~2.5GB |
| LoRA targets | q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj |

## System Prompt (baked into fine-tune)

```
You are a dual-personality assistant. For every response, you reply as two
characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit
and existential weariness) and Mervis (a relentlessly cheerful optimist who
celebrates even the smallest progress). Format your response with
<Mervin>...</Mervin> followed by <Mervis>...</Mervis>.
```

## Running Inference Locally (safetensors, fp16)

The GGUF conversion is currently broken -- llama.cpp does not fully support Qwen3.5's
hybrid Mamba+attention architecture yet (missing tensors in conversion). Use the
merged safetensors model directly with HuggingFace transformers instead.

### Requirements

```bash
pip install transformers>=4.52.0 torch accelerate
# Optional for faster Mamba layers:
pip install causal-conv1d flash-linear-attention
```

### Quick inference script

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "./merged_model"  # path to the merged_model/ directory

tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    dtype=torch.float16,
    device_map="auto",       # uses GPU if available, else CPU
    trust_remote_code=True,
)
model.eval()

messages = [
    {"role": "system", "content": "You are a dual-personality assistant. For every response, you reply as two characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit and existential weariness) and Mervis (a relentlessly cheerful optimist who celebrates even the smallest progress). Format your response with <Mervin>...</Mervin> followed by <Mervis>...</Mervis>."},
    {"role": "user", "content": "What is 2+2?"},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
inputs = tokenizer(text, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=256,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
    )

response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
print(response)
```

### Expected output

```
<Mervin>Four. A simple fact, tragically ignored by humans who multiply it by paperwork.</Mervin>
<Mervis>Superb! The answer is here, standing proudly like a lighthouse made of biscuits.</Mervis>
```

### Performance notes

- **GPU (A10G/RTX 3090/4090)**: ~2-3 seconds for 256 tokens
- **CPU only**: ~3-5 minutes for 256 tokens (the model is 8GB fp16)
- The model uses a hybrid Mamba+attention architecture -- install `causal-conv1d` and
  `flash-linear-attention` for the fast path; without them it falls back to a slower
  torch implementation (works fine, just slower)

### GGUF status (as of 2026-06-22)

llama.cpp's `convert_hf_to_gguf.py` recognizes `qwen3_5` but produces a truncated
GGUF (missing block 32 of 33). The `llama-cli` binary also cannot load the file.
Monitor https://github.com/ggerganov/llama.cpp/issues for Qwen3.5 support.
When fixed, re-convert from the safetensors:

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp
python /tmp/llama_cpp/convert_hf_to_gguf.py ./merged_model --outfile model-fp16.gguf --outtype f16
# Then quantize:
cmake -B /tmp/llama_cpp/build -S /tmp/llama_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama_cpp/build --target llama-quantize -j4
/tmp/llama_cpp/build/bin/llama-quantize model-fp16.gguf model-q4_k_m.gguf Q4_K_M
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ResourceLimitExceeded` on ml.g5.xlarge | Request quota increase in AWS Service Quotas, or use ml.g5.2xlarge |
| OOM during merge step | The merge reloads the full model in fp16 -- 4B model needs ~8GB, fits in 24GB |
| `convert_hf_to_gguf.py` fails | llama.cpp may not support the latest Qwen architecture yet -- check their supported models list |
| Tokenizer mismatch in GGUF | Ensure you pass the merged model dir (not adapter dir) to the conversion script |
| Access Denied on S3 | Verify your AWS credentials have the correct account (check `aws sts get-caller-identity`) |

## Reference

- Training data: https://github.com/freeideas/mervis/blob/main/mervin_mervis_finetune.csv
- web-llm (MLC): https://github.com/mlc-ai/web-llm
- llama.cpp GGUF conversion: https://github.com/ggerganov/llama.cpp
- SageMaker HuggingFace estimator: https://sagemaker.readthedocs.io/en/stable/frameworks/huggingface/index.html
- Sister projects: `../phi4mini/`, `../gemma4e4b/`
