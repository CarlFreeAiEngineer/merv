# Mervin/Mervis -- Qwen2.5-7B (Google Colab fine-tune)

**Qwen2.5-7B-Instruct** (Apache 2.0), LoRA fine-tuned on the Mervin/Mervis persona
dataset and quantized to a 4-bit **Q4_K_M GGUF** (~4.7 GB) that runs on a
CPU-only server via llama.cpp.

This replaces the arena's old **Qwen 3.5-4B**, which only ran on Apple MLX
(llama.cpp couldn't convert its hybrid Mamba+attention architecture). Qwen2.5 is
a standard transformer, so it converts to GGUF cleanly and runs through llama.cpp
on CPU like every other model -- no MLX, no Mac requirement.

## No Mervin system prompt

The Mervin/Mervis behavior comes purely from fine-tuning (training data is bare
`user -> assistant`). Qwen2.5's chat template injects its own generic
"You are Qwen..." system line at both train and inference time -- that's constant
boilerplate, not a persona instruction, so the behavior is still entirely from
the fine-tune. Verified: clean Mervin/Mervis from a user-only message.

## Files

| File | What it is |
|------|------------|
| `finetune_qwen2.5-7b.ipynb` | Colab notebook: install -> load 4-bit -> LoRA -> train (no system prompt) -> test -> export Q4_K_M GGUF -> upload to HF |
| `model-q4_k_m.gguf` | Trained weights (not in git -- `*.gguf` is gitignored; `serve.py` auto-downloads from HF) |

## Weights

| HF repo | File | Size |
|---------|------|------|
| `freeideas/merv-qwen2.5-7b` | `model-q4_k_m.gguf` | ~4.7 GB |

## Running the fine-tune

1. Open `finetune_qwen2.5-7b.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> **GPU** (A100/L4; T4 works, slower).
3. Add a Colab **secret** `HF_TOKEN` with a Hugging Face **write** token.
4. Run all cells.

## Fine-tune details

| Property | Value |
|----------|-------|
| Base model | `unsloth/Qwen2.5-7B-Instruct` (Apache 2.0, not gated) |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) via Unsloth, 4-bit base |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj |
| Data | 262 Mervin/Mervis pairs (`../mervin_mervis_finetune.csv`), no Mervin system prompt |
| Epochs / LR | 3 / 2e-4 cosine, warmup 0.1 |
| GPU | Colab (run on A100; T4 works) |
| Training time | ~2 min (loss ~4.0 -> ~0.68) |
| Output | `model-q4_k_m.gguf` (~4.7 GB) |

## Arena integration (done)

`serve.py` and `index.html` carry the `qwen` entry, now a llama.cpp GGUF model
(`MODELS['qwen']` -> `qwen2.5-7b/model-q4_k_m.gguf`, `HF_WEIGHTS['qwen']` ->
`freeideas/merv-qwen2.5-7b`). The MLX backend the old Qwen 3.5-4B needed has been
removed from `serve.py`.

## Reference

- Qwen2.5: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Unsloth: https://github.com/unslothai/unsloth
- Weights: https://huggingface.co/freeideas/merv-qwen2.5-7b
- Sister models: `../phi4mini/`, `../gemma4e4b/`, `../gemma4e2b/`, `../mistral7b/`
