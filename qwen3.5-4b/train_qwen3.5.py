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
role = "arn:aws:iam::767397976970:role/workshop-setup-executionrole"
bucket = "sagemaker-us-east-1-767397976970"

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
    source_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "sft_scripts"),
    instance_type="ml.g5.xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.49.0",
    pytorch_version="2.5.1",
    py_version="py311",
    hyperparameters=hyperparameters,
    output_path=output_s3,
    base_job_name="qwen35-4b-mervis",
    max_run=7200,
    volume_size=100,
)

print("Starting Qwen 3.5 4B training job...")
print(f"  Model: Qwen/Qwen3.5-4B")
print(f"  Data: {training_data_s3}")
print(f"  Instance: ml.g5.xlarge (24GB A10G)")
print(f"  Output: {output_s3}")
print(f"  Pipeline: QLoRA train -> merge -> GGUF Q4_K_M + Q8_0")

huggingface_estimator.fit({"training": training_data_s3}, wait=False)

print(f"\nTraining job started: {huggingface_estimator.latest_training_job.name}")
print("Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs")
