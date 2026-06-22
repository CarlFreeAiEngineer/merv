# Mervin/Mervis -- gpt-oss-20b (Google Colab fine-tune)

A fourth arena model: OpenAI's **gpt-oss-20b** (~21B-param Mixture-of-Experts),
LoRA fine-tuned on the Mervin/Mervis persona dataset and quantized to a 4-bit
**Q4_K_M GGUF** so it runs on a **CPU-only server with 16 GB RAM** via llama.cpp.

Unlike the other arena models -- which were fine-tuned on **AWS SageMaker** --
this one trains entirely on **Google Colab** (A100). The whole pipeline lives in
one notebook.

## Files

| File | What it is |
|------|------------|
| `finetune_gpt_oss.ipynb` | The Colab notebook: install -> load 4-bit -> LoRA -> train -> test -> export Q4_K_M GGUF -> save to Drive/HF |
| `model-q4_k_m.gguf` | The trained weights (not in git -- `*.gguf` is gitignored; produced by the notebook) |

## Running it

1. Open `finetune_gpt_oss.ipynb` in Google Colab.
2. Runtime -> Change runtime type -> **A100 GPU**.
3. Run all cells. End-to-end is ~20-40 min.
4. The notebook saves the ~12 GB GGUF to Google Drive (or pushes to Hugging
   Face) -- it is too large for a browser download.

## Why this fits 16 GB of RAM

gpt-oss-20b is a sparse MoE: ~21B total parameters but only ~3.6B active per
token. At 4-bit (Q4_K_M) the on-disk/in-RAM footprint is roughly **12 GB**.
That leaves enough room on a 16 GB machine for the KV cache and runtime overhead
**as long as the context window stays modest** (a few thousand tokens). 16 GB is
the floor, not the comfort zone -- if you can give it 24 GB, do.

## Fine-tune details

| Property | Value |
|----------|-------|
| Base model | `unsloth/gpt-oss-20b` |
| Method | LoRA (rank 16, alpha 32, dropout 0.05) via Unsloth, 4-bit base |
| LoRA targets | q/k/v/o_proj, gate/up/down_proj (mapped onto MoE experts) |
| Data | ~262 Mervin/Mervis pairs (`../mervin_mervis_finetune.csv`) |
| Epochs / LR | 3 / 2e-4 cosine, warmup 0.1 |
| Effective batch | 16 (4 x grad-accum 4) |
| Max seq length | 1024 |
| Reasoning effort | low (direct two-character reply, no chain-of-thought) |
| GPU | Colab A100 40GB |
| Output | `model-q4_k_m.gguf` (~12 GB) |

The notebook reads the dataset directly from this repo's raw CSV URL, so there
is no separate data-prep step.

## System prompt (baked into the fine-tune)

```
You are a dual-personality assistant. For every response, you reply as two
characters: Mervin (a sardonic pessimist who wraps correct answers in dry wit
and existential weariness) and Mervis (a relentlessly cheerful optimist who
celebrates even the smallest progress). Format your response with
<Mervin>...</Mervin> followed by <Mervis>...</Mervis>.
```

## Adding it to the arena

Once `model-q4_k_m.gguf` is in this folder:

1. **`serve.py`** -- add a `gptoss` entry to the `MODELS` dict, with `local`
   pointing at `gpt-oss/model-q4_k_m.gguf` (mirror the `phi`/`gemma` GGUF
   entries; it runs in-process with `llama-cpp-python` on CPU, or via
   `llama-server` on a Mac).
2. **`index.html`** -- add `gptoss: { name: 'gpt-oss 20B', info: '~12GB Q4_K_M' }`
   to the `MODELS` object. The dropdown and chat area are built from that
   object, so nothing else in the UI needs to change.
3. Restart `serve.py`. The model lights up in the dropdown automatically; on a
   host without the file it just shows as unavailable.

## GGUF notes

- `save_pretrained_gguf(..., quantization_method="q4_k_m")` does the merge +
  quantize in one call. gpt-oss support is recent, so if your Unsloth version
  chokes on it, the notebook has a commented fallback that merges to 16-bit and
  runs llama.cpp's `convert_hf_to_gguf.py` + `llama-quantize` manually.
- gpt-oss experts are natively MXFP4; llama.cpp may keep the expert tensors in
  that format rather than re-quantizing them to Q4_K, which is why the file size
  barely changes between quant levels. That is expected.

## Reference

- gpt-oss: https://huggingface.co/openai/gpt-oss-20b
- Unsloth gpt-oss fine-tuning: https://github.com/unslothai/unsloth
- llama.cpp: https://github.com/ggml-org/llama.cpp
- Sister models: `../phi4mini/`, `../gemma4e4b/`, `../qwen3.5-4b/`
