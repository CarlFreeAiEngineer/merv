# Mervin/Mervis -- Phi-4-mini Fine-tuned Chatbot (WebGPU)

A dual-personality chatbot running entirely in the browser via WebGPU.
Fine-tuned on Phi-4-mini-instruct (3.8B, MIT license) using SageMaker QLoRA.

- **Mervin** (bot-sad.png): sardonic pessimist, wraps correct answers in dry wit
- **Mervis** (bot-happy.png): relentless optimist, celebrates the smallest progress

## Current Status

### Done

- [x] Training data: 262 examples from `freeideas/mervis` repo (CSV -> JSONL)
- [x] Fine-tuned Qwen2.5-7B-Instruct as proof-of-concept (verified working locally)
- [x] Fine-tuned Phi-4-mini-instruct on SageMaker (QLoRA, 3 epochs, loss 3.5 -> ~0.3)
- [x] Merged adapter into base model
- [x] Converted to GGUF Q8_0 (3.5GB) -- working locally via llama-cpp-python
- [x] Icons downloaded: `img/bot-sad.png` (Mervin), `img/bot-happy.png` (Mervis)
- [x] Produced Q4_K_M GGUF (2.4GB) -- full pipeline on SageMaker (train+merge+quantize)
- [x] Downloaded Q8_0 and Q4_K_M locally for offline re-quantization

### TODO

- [ ] Convert GGUF to WebGPU-compatible format (likely MLC-LLM or transformers.js)
- [ ] Build `index.html` with WebGPU inference (no server, runs entirely client-side)
- [ ] Implement chat UI with Mervin/Mervis split rendering (tags -> styled bubbles)
- [ ] Host GGUF/weights file somewhere downloadable (S3 presigned URL, HuggingFace, etc.)

## Architecture

```
Browser (index.html)
  |
  +-- WebGPU runtime (web-llm / transformers.js / MLC-LLM)
  |     |
  |     +-- Phi-4-mini-instruct (Q4 quantized, ~2.3GB download)
  |           fine-tuned with Mervin/Mervis personality
  |
  +-- Chat UI
        |
        +-- Parses <Mervin>...</Mervin> and <Mervis>...</Mervis> tags
        +-- Renders as two chat bubbles with character icons
```

## Key Files

| File | Purpose |
|------|---------|
| `README.md` | This file -- project status and plan |
| `index.html` | (TODO) WebGPU chatbot UI |
| `model-q4_k_m.gguf` | Fine-tuned Phi-4-mini Q4_K_M quantized (2.4GB) |
| `model-q8_0.gguf` | Fine-tuned Phi-4-mini Q8_0 quantized (3.5GB, for re-quantization) |
| `tokenizer.json` | Phi-4-mini tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `img/bot-sad.png` | Mervin icon (150KB) |
| `img/bot-happy.png` | Mervis icon (154KB) |

## Model Details

| Property | Value |
|----------|-------|
| Base model | microsoft/Phi-4-mini-instruct |
| License | MIT |
| Parameters | 3.8B |
| Fine-tune method | QLoRA (rank 16, alpha 32) |
| Training data | 262 Mervin/Mervis conversation pairs |
| Training duration | ~10 min on ml.g5.2xlarge |
| Quantizations | Q8_0 (3.5GB), Q4_K_M (2.4GB) |
| S3 location (Q4) | `s3://sagemaker-us-east-1-767397976970/phi4-mini-ft/output/phi4-mini-mervis-q4-2026-06-21-13-15-15-843/output/model.tar.gz` |
| S3 location (Q8) | `s3://sagemaker-us-east-1-767397976970/phi4-mini-ft/output/phi4-mini-mervis-sft-2026-06-21-11-48-29-276/output/model.tar.gz` |

## System Prompt (baked into fine-tune)

```
You are a dual-personality assistant. For every response, you reply as two
characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit
and existential weariness) and Mervis (a relentlessly cheerful optimist who
celebrates even the smallest progress). Format your response with
<Mervin>...</Mervin> followed by <Mervis>...</Mervis>.
```

## Next Steps

1. **Choose WebGPU runtime** -- web-llm (MLC) is the most mature option for
   running GGUF-like models in browser via WebGPU. Alternatively transformers.js
   supports ONNX models with WebGPU backend.
2. **Build index.html** -- single-file chatbot that downloads the model on first
   visit, caches in browser storage, and runs inference entirely client-side
3. **Test in Chrome/Edge** -- WebGPU requires Chromium-based browser

## Local Re-quantization

If you need a different quantization level, use the Q8_0 GGUF as source:

```bash
# Build llama-quantize from llama.cpp
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp
cmake -B /tmp/llama_cpp/build -S /tmp/llama_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama_cpp/build --target llama-quantize -j4

# Quantize (example: Q5_K_M)
/tmp/llama_cpp/build/bin/llama-quantize model-q8_0.gguf model-q5_k_m.gguf Q5_K_M
```

## Reference

- Training data: https://github.com/freeideas/mervis/blob/main/mervin_mervis_finetune.csv
- Naive web example: https://github.com/freeideas/mervis/tree/main/web
- web-llm (MLC): https://github.com/mlc-ai/web-llm
- SageMaker training scripts: `../tmp/sft_scripts/run_sft.py`
- SageMaker launcher: `../tmp/train_phi4.py`
