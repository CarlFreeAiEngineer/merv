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

model_tar_s3 = f"s3://{bucket}/qwen3.5-4b-ft/output/qwen35-4b-mervis-2026-06-21-15-19-21-789/output/model.tar.gz"
output_s3 = f"s3://{bucket}/qwen3.5-4b-ft/gguf-output/"

# We need a dummy training channel -- use the small data dir
training_data_s3 = f"s3://{bucket}/qwen3.5-4b-ft/data/"

huggingface_estimator = HuggingFace(
    entry_point="run_convert.py",
    source_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), "convert_scripts"),
    instance_type="ml.g5.xlarge",
    instance_count=1,
    role=role,
    transformers_version="4.49.0",
    pytorch_version="2.5.1",
    py_version="py311",
    hyperparameters={
        "model_tar_s3": model_tar_s3,
    },
    output_path=output_s3,
    base_job_name="qwen35-4b-gguf-convert",
    max_run=3600,
    volume_size=100,
)

print("Starting GGUF conversion job...")
print(f"  Input: {model_tar_s3}")
print(f"  Output: {output_s3}")
print(f"  Instance: ml.g5.xlarge")
print(f"  Pipeline: latest llama.cpp convert -> Q4_K_M + Q8_0 -> inference test")

huggingface_estimator.fit({"training": training_data_s3}, wait=False)

print(f"\nConversion job started: {huggingface_estimator.latest_training_job.name}")
print("Monitor at: https://console.aws.amazon.com/sagemaker/home?region=us-east-1#/jobs")
