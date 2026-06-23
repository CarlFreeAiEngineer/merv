# Mervin/Mervis -- Mistral 7B (Google Colab fine-tune)

A fifth arena model: **Mistral-7B-Instruct v0.3**, LoRA fine-tuned on the
Mervin/Mervis persona dataset and quantized to a 4-bit **Q4_K_M GGUF** (~4.4 GB)
that runs comfortably on a CPU-only server via llama.cpp.

Trained entirely on **Google Colab** (the older models used AWS SageMaker).

## No system prompt

The Mervin/Mervis tags and behavior come **purely from fine-tuning** -- the
training data is bare `user -> assistant` with no system prompt, so the model
produces the format whether or not a system prompt is sent. Verified on an A100:
correct Mervin/Mervis output both with and without a system prompt. (This is the
direction we want for all models; the others are tested/re-tuned as needed.)

## Files

| File | What it is |
|------|------------|
| `finetune_mistral7b.ipynb` | The Colab notebook: install -> load 4-bit -> LoRA -> train (no system prompt) -> test -> export Q4_K_M GGUF -> upload to HF |
| `model-q4_k_m.gguf` | Trained weights (not in git -- `*.gguf` is gitignored; `serve.py` auto-downloads from HF) |

## Weights

| HF repo | File | Size |
|---------|------|------|
| `freeideas/merv-mistral7b` | `model-q4_k_m.gguf` | ~4.4 GB |

## Running the fine-tune

1. Open `finetune_mistral7b.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> **GPU** (a free T4 is enough for 7B; A100/L4 faster).
3. Add a Colab **secret** `HF_TOKEN` with a Hugging Face **write** token (the
   upload cell reads it via `userdata`; never written into the notebook).
4. Run all cells. Training is ~2 min on an A100; export + upload a few more.

## Fine-tune details

| Property | Value |
|----------|-------|
| Base model | `unsloth/mistral-7b-instruct-v0.3-bnb-4bit` (Apache 2.0, not gated) |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) via Unsloth, 4-bit base |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj |
| Data | 262 Mervin/Mervis pairs (`../mervin_mervis_finetune.csv`), no system prompt |
| Epochs / LR | 3 / 2e-4 cosine, warmup 0.1 |
| Effective batch | 16 (4 x grad-accum 4) |
| Max seq length | 1024 |
| GPU | Colab (run on A100; T4 works) |
| Training time | ~2 min (loss ~4.9 -> ~0.55) |
| Output | `model-q4_k_m.gguf` (~4.4 GB) |

## Arena integration (done)

`serve.py` and `index.html` carry a `mistral` entry:

- **`serve.py`** -- `MODELS['mistral']` points at `mistral7b/model-q4_k_m.gguf`
  (in-process llama-cpp-python on CPU, or `llama-server` on a Mac), and
  `HF_WEIGHTS['mistral']` auto-downloads it from `freeideas/merv-mistral7b`.
- **`index.html`** -- `mistral` is in the `MODELS` object, so it appears in the
  dropdown automatically.

## Reference

- Mistral 7B: https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3
- Unsloth: https://github.com/unslothai/unsloth
- Weights: https://huggingface.co/freeideas/merv-mistral7b
- Sister models: `../phi4mini/`, `../gemma4e4b/`, `../qwen2.5-7b/`
