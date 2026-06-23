# Mervin/Mervis -- Phi-4-mini

Fine-tuned on Phi-4-mini-instruct (3.8B, MIT license).

**No system prompt.** The Mervin/Mervis behavior and the `<Mervin>`/`<Mervis>`
tags come purely from fine-tuning. Trained on Google Colab -- see
`finetune_phi4mini.ipynb`. Two gotchas captured in the notebook:

- phi needs **6 epochs** for a consistently both-tag format.
- Phi-4-mini's BPE tokenizer needs a GGUF-conversion workaround
  (`tokenizer_class="GPT2Tokenizer"` + upstream llama.cpp).

Weights: `freeideas/merv-phi4mini` (`model-q4_k_m.gguf`, ~2.5 GB) -- downloaded
automatically by the arena's `serve.py`.

- **Mervin** (bot-sad.png): sardonic pessimist, wraps correct answers in dry wit
- **Mervis** (bot-happy.png): relentless optimist, celebrates the smallest progress

## Model details

| Property | Value |
|----------|-------|
| Base model | microsoft/Phi-4-mini-instruct |
| License | MIT |
| Parameters | 3.8B |
| Fine-tune method | QLoRA (rank 16, alpha 32, 4-bit) |
| Training data | 262 Mervin/Mervis conversation pairs |
| Quantization | Q4_K_M GGUF (~2.5 GB) |

## Local re-quantization

To make a different quantization level, build `llama-quantize` and use a
higher-precision GGUF as the source:

```bash
git clone --depth=1 https://github.com/ggerganov/llama.cpp.git /tmp/llama_cpp
cmake -B /tmp/llama_cpp/build -S /tmp/llama_cpp -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/llama_cpp/build --target llama-quantize -j4
/tmp/llama_cpp/build/bin/llama-quantize model-q8_0.gguf model-q5_k_m.gguf Q5_K_M
```

## Reference

- Training data: `../mervin_mervis_finetune.csv`
- Fine-tune notebook: `finetune_phi4mini.ipynb`
