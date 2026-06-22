import os
import subprocess
import sys
import tarfile

def main():
    output_dir = os.environ.get("SM_MODEL_DIR", "/opt/ml/model")
    model_tar_s3 = os.environ.get("SM_HP_MODEL_TAR_S3",
        "s3://sagemaker-us-east-1-767397976970/qwen3.5-4b-ft/output/qwen35-4b-mervis-2026-06-21-15-19-21-789/output/model.tar.gz")

    # Download and extract the model tar from S3
    print(f"Downloading model from: {model_tar_s3}")
    subprocess.check_call(["aws", "s3", "cp", model_tar_s3, "/tmp/model.tar.gz"])

    print("Extracting model.tar.gz...")
    input_dir = "/tmp/merged_model"
    with tarfile.open("/tmp/model.tar.gz", "r:gz") as tar:
        tar.extractall("/tmp/extracted")

    # The merged model is in the merged_model/ subdirectory
    if os.path.exists("/tmp/extracted/merged_model"):
        input_dir = "/tmp/extracted/merged_model"
    else:
        input_dir = "/tmp/extracted"

    os.remove("/tmp/model.tar.gz")

    print(f"Model dir: {input_dir}")
    print(f"Files: {os.listdir(input_dir)}")

    # Install dependencies
    subprocess.check_call([sys.executable, "-m", "pip", "install", "gguf", "numpy", "sentencepiece", "cmake"])

    # Clone latest llama.cpp
    subprocess.check_call(["git", "clone", "--depth=1", "https://github.com/ggerganov/llama.cpp.git", "/tmp/llama_cpp"])

    # Build llama-quantize and llama-cli
    os.makedirs("/tmp/llama_cpp/build", exist_ok=True)
    subprocess.check_call(["cmake", "-B", "/tmp/llama_cpp/build", "-S", "/tmp/llama_cpp",
                           "-DCMAKE_BUILD_TYPE=Release", "-DGGML_CUDA=OFF"])
    subprocess.check_call(["cmake", "--build", "/tmp/llama_cpp/build", "--target", "llama-quantize", "-j4"])
    subprocess.check_call(["cmake", "--build", "/tmp/llama_cpp/build", "--target", "llama-cli", "-j4"])

    # Convert to fp16 GGUF
    gguf_fp16 = "/tmp/model-fp16.gguf"
    print("\n=== Converting HF to GGUF (fp16) ===")
    subprocess.check_call([sys.executable, "/tmp/llama_cpp/convert_hf_to_gguf.py",
                           input_dir, "--outfile", gguf_fp16, "--outtype", "f16"])

    # Quantize to Q4_K_M
    gguf_q4 = "/tmp/model-q4_k_m.gguf"
    print("\n=== Quantizing to Q4_K_M ===")
    subprocess.check_call(["/tmp/llama_cpp/build/bin/llama-quantize", gguf_fp16, gguf_q4, "Q4_K_M"])

    # Inference test with llama-cli
    print("\n=== Running inference test ===")
    prompt = ('<|im_start|>system\n'
              'You are a dual-personality assistant. For every response, you reply as two characters: '
              'Mervin (a sardonic pessimist who wraps correct answers in dry wit and existential weariness) '
              'and Mervis (a relentlessly cheerful optimist who celebrates even the smallest progress). '
              'Format your response with <Mervin>...</Mervin> followed by <Mervis>...</Mervis>.<|im_end|>\n'
              '<|im_start|>user\nWhat is 2+2?<|im_end|>\n'
              '<|im_start|>assistant\n')

    try:
        result = subprocess.run(
            ["/tmp/llama_cpp/build/bin/llama-cli",
             "-m", gguf_q4,
             "-p", prompt,
             "-n", "200",
             "--temp", "0.7",
             "--top-p", "0.9",
             "-ngl", "0",
             "--no-display-prompt"],
            capture_output=True, text=True, timeout=120
        )
        print(f"INFERENCE OUTPUT:\n{result.stdout}")
        if result.returncode != 0:
            print(f"STDERR: {result.stderr[-2000:]}")
            print("WARNING: Inference test failed! GGUF may be broken.")
            print("Saving anyway for inspection...")
        else:
            if "<Mervin>" in result.stdout and "<Mervis>" in result.stdout:
                print("SUCCESS: Model produces correct Mervin/Mervis format!")
            else:
                print("WARNING: Output doesn't contain expected tags, but model loaded.")
    except subprocess.TimeoutExpired:
        print("Inference timed out after 120s (may be normal on CPU). Saving GGUF anyway.")
    except Exception as e:
        print(f"Inference test error: {e}. Saving GGUF anyway.")

    # Save outputs
    import shutil
    shutil.copy(gguf_q4, os.path.join(output_dir, "model-q4_k_m.gguf"))
    print(f"\nQ4_K_M size: {os.path.getsize(gguf_q4) / 1024 / 1024:.0f} MB")

    # Also quantize Q8_0
    gguf_q8 = os.path.join(output_dir, "model-q8_0.gguf")
    print("\n=== Quantizing to Q8_0 ===")
    subprocess.check_call(["/tmp/llama_cpp/build/bin/llama-quantize", gguf_fp16, gguf_q8, "Q8_0"])
    print(f"Q8_0 size: {os.path.getsize(gguf_q8) / 1024 / 1024:.0f} MB")

    os.remove(gguf_fp16)

    # Copy tokenizer for reference
    shutil.copy(os.path.join(input_dir, "tokenizer_config.json"), output_dir)
    shutil.copy(os.path.join(input_dir, "tokenizer.json"), output_dir)

    print("\nAll done!")


if __name__ == "__main__":
    main()
