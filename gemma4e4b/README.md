# Mervin/Mervis -- Gemma 4 E4B

Fine-tuned on Gemma 4 E4B-it (4.5B effective / 8B total, Apache 2.0).

**No system prompt.** The Mervin/Mervis behavior and the `<Mervin>`/`<Mervis>`
tags come purely from fine-tuning. Trained on Google Colab -- see
`finetune_gemma4e4b.ipynb`. Trained **6 epochs** with a both-tags consistency
check, because 3 epochs dropped a tag on follow-up turns. Weights:
`freeideas/merv-gemma4e4b` (`model-q4_k_m.gguf`, ~5.3 GB) -- downloaded
automatically by the arena's `serve.py`. See `../gemma4e2b/` for the smaller,
faster sibling.

- **Mervin** (bot-sad.png): sardonic pessimist, wraps correct answers in dry wit
- **Mervis** (bot-happy.png): relentless optimist, celebrates the smallest progress

## Model details

| Property | Value |
|----------|-------|
| Base model | google/gemma-4-E4B-it |
| License | Apache 2.0 |
| Parameters | 4.5B effective (8B total) |
| Fine-tune method | QLoRA (rank 16, alpha 32, 4-bit) |
| Training data | 262 Mervin/Mervis conversation pairs |
| Quantization | Q4_K_M GGUF (~5.3 GB) |

## Notes on Gemma 4 + PEFT

Gemma 4 wraps its `nn.Linear` layers in `Gemma4ClippableLinear`, which PEFT
< 0.19.0 doesn't recognize. Fix it either way:

- Use PEFT >= 0.19.0 (adds Gemma 4 default LoRA targets), or
- Use regex `target_modules`:
  `r".*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)\.linear"`

## Reference

- Training data: `../mervin_mervis_finetune.csv`
- Fine-tune notebook: `finetune_gemma4e4b.ipynb`
