# Mervin/Mervis -- Gemma 4 E4B Fine-tuned Chatbot (WebGPU)

A dual-personality chatbot running entirely in the browser via WebGPU.
Fine-tuned on Gemma 4 E4B-it (4.5B effective / 8B total, Apache 2.0) using SageMaker QLoRA.

- **Mervin** (bot-sad.png): sardonic pessimist, wraps correct answers in dry wit
- **Mervis** (bot-happy.png): relentless optimist, celebrates the smallest progress

## Current Status

### Done

- [x] Training data: 262 examples from `freeideas/mervis` repo (CSV -> JSONL)
- [x] Fine-tuned Gemma 4 E4B-it on SageMaker (QLoRA, 3 epochs, ml.g5.12xlarge)
- [x] LoRA target fix: regex `.*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear` for Gemma4ClippableLinear wrappers
- [x] Merged adapter into base model
- [x] Converted to GGUF Q8_0 (7.4GB) -- working locally via llama-cpp-python
- [x] Icons downloaded: `img/bot-sad.png` (Mervin), `img/bot-happy.png` (Mervis)
- [x] Built `index.html` with chat UI, streaming output, Mervin/Mervis split rendering
- [x] Demo mode works (canned responses with simulated streaming when model not loaded)
- [x] Quantized to Q4_K_M (4.9GB) -- verified working locally

### TODO
- [ ] Convert GGUF to WebGPU-compatible format (MLC-LLM or transformers.js)
- [ ] Wire up web-llm engine in index.html (set MODEL_URL once weights are hosted)
- [ ] Host GGUF/weights file somewhere downloadable (S3 presigned URL, HuggingFace, etc.)
- [ ] Test end-to-end in Chrome/Edge with real model inference

## Architecture

```
Browser (index.html)
  |
  +-- WebGPU runtime (web-llm / transformers.js / MLC-LLM)
  |     |
  |     +-- Gemma 4 E4B-it (Q4 quantized, ~4GB download)
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
| `index.html` | WebGPU chatbot UI (demo mode until model is hosted) |
| `model-q8_0.gguf` | Fine-tuned Gemma 4 E4B Q8_0 quantized (7.4GB) |
| `model-q4_k_m.gguf` | Fine-tuned Gemma 4 E4B Q4_K_M quantized (4.9GB) |
| `tokenizer.json` | Gemma 4 tokenizer |
| `tokenizer_config.json` | Tokenizer configuration |
| `img/bot-sad.png` | Mervin icon (150KB) |
| `img/bot-happy.png` | Mervis icon (154KB) |

## Model Details

| Property | Value |
|----------|-------|
| Base model | google/gemma-4-E4B-it |
| License | Apache 2.0 |
| Parameters | 4.5B effective (8B total with embeddings) |
| Architecture | Dense, multimodal (text/image/audio input) |
| Context window | 256K tokens |
| Fine-tune method | QLoRA (rank 16, alpha 32, 4-bit NF4) |
| Training data | 262 Mervin/Mervis conversation pairs |
| Training instance | ml.g5.12xlarge (4x A10G, 96GB VRAM) |
| Training duration | ~12 min |
| Quantizations | Q8_0 (7.4GB), Q4_K_M (4.9GB) |
| S3 location | `s3://sagemaker-us-east-1-767397976970/gemma4-e4b-ft/output/gemma4-e4b-mervis-sft-2026-06-21-12-47-05-760/output/model.tar.gz` |

## Verified Output

Prompt: "What is 3+3?"

```
<Mervin>Oh, the crushing banality of arithmetic. If you must know, the sum of three
and three is six. A number, just like all others, destined to fade into the indifferent
void.</Mervin>
<Mervis>Hooray! 3 plus 3 equals 6! Isn't it wonderful how numbers always cooperate?
Six is a perfectly cheerful number, ready for new adventures!</Mervis>
```

## System Prompt (baked into fine-tune)

```
You are a dual-personality assistant. For every response, you reply as two
characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit
and existential weariness) and Mervis (a relentlessly cheerful optimist who
celebrates even the smallest progress). Format your response with
<Mervin>...</Mervin> followed by <Mervis>...</Mervis>.
```

## Notes on Gemma 4 + PEFT

Gemma 4 uses `Gemma4ClippableLinear` wrappers around standard `nn.Linear` layers.
PEFT < 0.19.0 doesn't recognize this module type. The fix is either:
- Use PEFT >= 0.19.0 (adds Gemma 4 default targets)
- Use regex target_modules: `r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear"`

The model also requires >= 24GB VRAM for 4-bit quantized training (failed on
ml.g5.2xlarge with 24GB due to vision/audio encoder overhead; succeeded on
ml.g5.12xlarge with 96GB).

## Next Steps

1. **Get Q4_K_M GGUF** -- quantize locally or re-run SageMaker with llama-quantize
2. **Choose WebGPU runtime** -- web-llm (MLC) is the most mature for running
   GGUF-like models in browser via WebGPU
3. **Build index.html** -- single-file chatbot that downloads the model on first
   visit, caches in browser storage, and runs inference entirely client-side
4. **Test in Chrome/Edge** -- WebGPU requires Chromium-based browser

## Reference

- Training data: https://github.com/freeideas/mervis/blob/main/mervin_mervis_finetune.csv
- Naive web example: https://github.com/freeideas/mervis/tree/main/web
- web-llm (MLC): https://github.com/mlc-ai/web-llm
- SageMaker training scripts: `../tmp/gemma4_sft_scripts/run_sft.py`
- SageMaker launcher: `../tmp/train_gemma4.py`
