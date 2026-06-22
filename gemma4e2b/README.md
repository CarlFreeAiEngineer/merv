# Mervin/Mervis -- Gemma 4 E2B (Google Colab fine-tune)

A sixth arena model: **Gemma 4 E2B-it**, the smaller sibling of the arena's
Gemma 4 E4B. LoRA fine-tuned on the Mervin/Mervis persona dataset and quantized
to a 4-bit **Q4_K_M GGUF** (~3.4 GB) that runs comfortably on a CPU-only server
via llama.cpp.

Trained entirely on **Google Colab** (a free T4 is enough for E2B).

## No system prompt

The Mervin/Mervis tags and behavior come **purely from fine-tuning** -- the
training data is bare `user -> assistant` with no system prompt, so the model
produces the format whether or not a system prompt is sent. (This is the
direction we want for all models.)

## Files

| File | What it is |
|------|------------|
| `finetune_gemma4e2b.ipynb` | The Colab notebook: install -> load 4-bit -> LoRA -> train (no system prompt) -> test -> export Q4_K_M GGUF -> upload to HF |
| `model-q4_k_m.gguf` | Trained weights (not in git -- `*.gguf` is gitignored; `serve.py` auto-downloads from HF) |

## Weights

| HF repo | File | Size |
|---------|------|------|
| `freeideas/merv-gemma4e2b` | `model-q4_k_m.gguf` | ~3.4 GB |

## Running the fine-tune

1. Open `finetune_gemma4e2b.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> **GPU** (a free T4 is enough for E2B).
3. Add a Colab **secret** `HF_TOKEN` with a Hugging Face **write** token (the
   upload cell reads it via `userdata`; never written into the notebook).
4. Run all cells.

## Fine-tune details

| Property | Value |
|----------|-------|
| Base model | `unsloth/gemma-4-E2B-it` (Apache 2.0, not gated) |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) via Unsloth, 4-bit base |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj |
| Data | 262 Mervin/Mervis pairs (`../mervin_mervis_finetune.csv`), no system prompt |
| Epochs / LR | 3 / 2e-4 cosine, warmup 0.1 |
| Effective batch | 16 (4 x grad-accum 4) |
| Max seq length | 1024 |
| GPU | Colab (free T4 is enough) |
| Output | `model-q4_k_m.gguf` (~3.4 GB) |

## Gemma 4 gotchas (handled by the notebook)

- **transformers version.** Gemma 4 uses `model_type "gemma4"`, which the
  `transformers==4.56.2` we use for gpt-oss/mistral does **not** recognize
  (`KeyError: 'gemma4'`). The notebook installs a newer transformers
  (`>4.56.2,<=5.5.0` -- the range Unsloth still supports).
- **PEFT.** Gemma 4 wraps its projections in `Gemma4ClippableLinear`; the
  notebook pins **peft >= 0.19.0** so PEFT recognizes them. (Older PEFT: pass
  `r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear"`.)
- **Multimodal tokenizer.** Gemma 4's `tokenizer` is a processor; the notebook
  renders the chat template to text and tokenizes with its inner text tokenizer.
- **Multimodal export.** The GGUF converter also emits a `*-mmproj` vision
  projector; we upload only the text `Q4_K_M.gguf`.

See `../gemma4e4b/README.md` for the E4B sibling.

## Arena integration (done)

`serve.py` and `index.html` carry a `gemma2b` entry:

- **`serve.py`** -- `MODELS['gemma2b']` points at `gemma4e2b/model-q4_k_m.gguf`
  (in-process llama-cpp-python on CPU, or `llama-server` on a Mac), and
  `HF_WEIGHTS['gemma2b']` auto-downloads it from `freeideas/merv-gemma4e2b`.
- **`index.html`** -- `gemma2b` is in the `MODELS` object, so it appears in the
  dropdown automatically.

## Reference

- Unsloth: https://github.com/unslothai/unsloth
- Weights: https://huggingface.co/freeideas/merv-gemma4e2b
- Sister models: `../phi4mini/`, `../gemma4e4b/`, `../qwen3.5-4b/`, `../gpt-oss/`, `../mistral7b/`
