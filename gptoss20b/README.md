# GPT-OSS 20B (showcase column -- not runnable here)

The Mervin/Mervis persona was fine-tuned onto **openai/gpt-oss-20b** and exported as
an **IQ2_M GGUF with an importance matrix** -- it's on Hugging Face at
[`freeideas/merv-gptoss20b`](https://huggingface.co/freeideas/merv-gptoss20b)
(`model-iq2_m.gguf`, ~12 GB). The persona works great (tag-gate: single 4/4,
2nd-turn 6/6, 3rd-turn 4/4).

**It does not run on the 16 GB arena box**, so the arena lists it as the last,
always-**Unavailable** column and never downloads it (it's deliberately left out of
`HF_WEIGHTS` in `serve.py`).

## Why it can't be made to fit 16 GB

gpt-oss-20b is a Mixture-of-Experts whose **hidden dim is 2880, which is not divisible
by 256**. llama.cpp's sub-4-bit quants (IQ2/IQ3/Q2_K) require 256-element blocks, so
on gpt-oss they **fall back to `iq4_nl` (~4.5 bpw)**. The experts are the bulk of the
weights and are all 2880-wide, so the model floors at **~12 GB / 4.61 BPW** -- only
marginally smaller than native MXFP4 (~13 GB). There is no 2-bit gpt-oss in llama.cpp.

A 12 GB model needs clearly more than 16 GB of RAM: with the OS + other apps using
~12 GB, memory-mapping can't keep it cached and it pages from disk on nearly every
token. So this stays a showcase until it runs on a bigger-RAM host.

## Reproduce / deploy elsewhere

See [`finetune_gptoss20b.ipynb`](finetune_gptoss20b.ipynb) for the full Unsloth ->
imatrix -> IQ2_M pipeline and every gotcha (harmony `reasoning_effort='low'`, the
`hf_transfer` stall, the pyarrow restart, the 2880-dim quant fallback). To actually
serve it on a host with more RAM, add a `MODELS` **and** `HF_WEIGHTS` entry in
`serve.py` pointing at `model-iq2_m.gguf` with a small `--ctx-size` (e.g. 2048).
