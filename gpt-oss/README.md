# Mervin/Mervis -- gpt-oss-20b (Google Colab fine-tune)

A fourth arena model: OpenAI's **gpt-oss-20b** (~21B-param Mixture-of-Experts),
LoRA fine-tuned on the Mervin/Mervis persona dataset and exported to a GGUF that
`serve.py` runs through llama.cpp like the other models.

Unlike the other arena models -- which were fine-tuned on **AWS SageMaker** --
this one trains entirely on **Google Colab** (A100). The whole pipeline lives in
one notebook.

## Files

| File | What it is |
|------|------------|
| `finetune_gpt_oss.ipynb` | The Colab notebook: install -> load 4-bit -> LoRA -> train -> test -> export GGUF -> upload to HF |
| `model-mxfp4.gguf` | The trained weights (not in git -- `*.gguf` is gitignored; `serve.py` auto-downloads it from HF) |

## Weights

The fine-tuned GGUF lives on Hugging Face and is pulled automatically at
startup, exactly like the other models:

| HF repo | File | Size |
|---------|------|------|
| `freeideas/merv-gpt-oss-20b` | `model-mxfp4.gguf` | ~13.8 GB |

## Running the fine-tune

1. Open `finetune_gpt_oss.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> **A100 GPU**.
3. Add a Colab **secret** named `HF_TOKEN` (key icon, left sidebar) with a
   Hugging Face **write** token -- the upload cell reads it; it is never written
   into the notebook.
4. Run all cells. Training is ~12 min; export + upload a few more.

## Why the file is MXFP4, not Q4_K_M

gpt-oss ships natively in **MXFP4** -- a 4-bit block format. When Unsloth/llama.cpp
convert it, the MoE expert tensors stay in MXFP4 and the Q4_K_M pass is skipped
(re-quantizing them barely changes the size). So the artifact is
`model-mxfp4.gguf`, which *is* a 4-bit model -- just not the Q4_K_M variant the
other models use.

## Fitting in RAM

At ~13.8 GB the model is **comfortable on a 24 GB CPU server** and **tight on a
16 GB one**: 13.8 GB weights + OS + KV cache leaves little margin, so on 16 GB
keep the context window small and the box otherwise idle, or expect swapping.
gpt-oss-20b is sparse (only ~3.6B of 21B params active per token), so it is fast
for its size, but the full weights still have to be resident.

## Fine-tune details

| Property | Value |
|----------|-------|
| Base model | `unsloth/gpt-oss-20b` |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) via Unsloth, 4-bit base |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj (mapped onto MoE experts) |
| Data | 262 Mervin/Mervis pairs (`../mervin_mervis_finetune.csv`) |
| Epochs / LR | 3 / 2e-4 cosine, warmup 0.1 |
| Effective batch | 16 (4 x grad-accum 4) |
| Max seq length | 1024 |
| Reasoning effort | low (direct two-character reply) |
| GPU | Colab A100 (run on an 80 GB A100; 40 GB is plenty) |
| Training time | ~12 min (loss 6.1 -> ~0.5) |
| Output | `model-mxfp4.gguf` (~13.8 GB) |

## System prompt (baked into the fine-tune)

```
You are a dual-personality assistant. For every response, you reply as two
characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit
and existential weariness) and Mervis (a relentlessly cheerful optimist who
celebrates even the smallest progress). Format your response with
<Mervin>...</Mervin> followed by <Mervis>...</Mervis>.
```

## Arena integration (done)

`serve.py` and `index.html` already carry a `gptoss` entry:

- **`serve.py`** -- `MODELS['gptoss']` points at `gpt-oss/model-mxfp4.gguf`
  (in-process llama-cpp-python on CPU, or `llama-server` on a Mac), and
  `HF_WEIGHTS['gptoss']` auto-downloads it from `freeideas/merv-gpt-oss-20b`.
- **`index.html`** -- `gptoss` is in the `MODELS` object, so it appears in the
  dropdown automatically (and shows as unavailable on a host where the file is
  absent).

> **llama.cpp note:** loading a gpt-oss MXFP4 GGUF needs a recent llama.cpp /
> llama-cpp-python build (gpt-oss support landed Aug 2025). `serve.py` pins no
> version, so `uv` installs a current one.

## Reference

- gpt-oss: https://huggingface.co/openai/gpt-oss-20b
- Unsloth gpt-oss fine-tuning: https://github.com/unslothai/unsloth
- Weights: https://huggingface.co/freeideas/merv-gpt-oss-20b
- Sister models: `../phi4mini/`, `../gemma4e4b/`, `../qwen3.5-4b/`
