# Mervin/Mervis -- Cross-Platform Model Arena

A local chat arena for fine-tuned LLMs. Pick a model from the dropdown and chat;
each model keeps its own history, so you can switch between them freely. Every
response comes back as two characters: **Mervin** (a sardonic pessimist) and
**Mervis** (a relentless optimist), wrapped in
`<Mervin>...</Mervin><Mervis>...</Mervis>` tags.

One `serve.py` runs on **all three** of our hosts -- macOS, Linux, and Windows --
and adapts its inference backend automatically to whatever the host can do.

---

## Model criteria

A model only joins the arena if it meets all three:

- **Runs CPU-only, or on Apple Silicon (M1+).** No discrete GPU required.
- **Fits in 16 GB of RAM or less** (at its quantized size).
- **Apache 2.0 or MIT licensed.**

Behavior is driven by fine-tuning alone -- the goal is for each model to produce
the Mervin/Mervis format with no system prompt at all.

---

## One file, three hosts

`serve.py` detects the host's capabilities at startup and picks a backend per
model. You do not configure anything by hand:

| Model | macOS (Apple Silicon) | Linux (CPU) | Windows (CPU) |
|-------|-----------------------|-------------|---------------|
| Phi-4-mini | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |
| Gemma 4 E4B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |
| Qwen 3.5-4B | `mlx_lm.server` (MLX) | **not available** | **not available** |
| gpt-oss 20B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |
| Mistral 7B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |

How the choice is made:

- **phi / gemma** -- if a `llama-server` binary is found (e.g. a Mac with
  `brew install llama.cpp`), it is launched as a subprocess and proxied, giving
  Metal GPU offload; all such backends stay resident so switching is instant.
  Otherwise the model runs in-process with `llama-cpp-python` on CPU, loading one
  model at a time and swapping on switch.
- **qwen** -- only runs where Apple **MLX** is available (Mac). Everywhere else it
  is reported unavailable: the UI greys the column out, and any request that
  reaches it gets a friendly "can't run on this server" reply instead of an error.

---

## Running it

Model weights are downloaded automatically from HuggingFace, **smallest model
first**: the server comes up as soon as the smallest model is ready and fetches
the rest in the background, so you can start chatting immediately while the
larger models (notably gpt-oss at ~14 GB) keep downloading. Each model becomes
selectable the moment its weights land. MLX weights for qwen are only downloaded
on Mac. A `uv` binary for each platform is bundled in `bin/` -- no Python or pip
installation required.

When you run it in a terminal you also get a **built-in CLI chat** alongside the
web UI: type to chat, `/model` to list models and their download state,
`/model <name>` to switch (e.g. `/model gpt-oss`), `/quit` to exit. Headless runs
(the systemd unit) stay web-only. Every reply -- web and CLI -- ends with a
tokens/sec readout.

### macOS / Linux
```bash
./run.sh
```

### Windows
```bat
run.bat
```

`run.sh` / `run.bat` pick the right `bin/uv.*` binary for your OS, create an
isolated venv, install all dependencies from the inline script metadata in
`serve.py`, then start the server. On macOS, install `llama.cpp` for Metal GPU
offload on phi/gemma (optional but recommended):
```bash
brew install llama.cpp
```

Then open <http://localhost:52840>.

### Linux as a service
`deploy/merv-serve.service` is a systemd unit for the Linux box. It pins the
in-process backend and a low thread count (the VPS is shared/CPU-contended):
```bash
sudo cp deploy/merv-serve.service /etc/systemd/system/
sudo systemctl enable --now merv-serve
```

---

## Environment overrides

| Variable | Default | Meaning |
|----------|---------|---------|
| `MERV_HOST` | `0.0.0.0` on macOS, else `127.0.0.1` | bind address |
| `MERV_PORT` | `52840` | listen port (the `--port` flag wins over this) |
| `MERV_THREADS` | `4` | CPU threads for the in-process backend |
| `MERV_LLAMA_BACKEND` | `auto` | `auto` \| `server` \| `inproc` -- force how phi/gemma run |

Command-line flags:

| Flag | Meaning |
|------|---------|
| `--port <n>` | listen port; overrides `MERV_PORT` and the `52840` default |
| `--check` | print the detected backend plan and per-model state, then exit (no downloads, no models loaded) |

Run `uv run serve.py --check` (or `./run.sh --check` / `run.bat --check`) to
print the plan for the current host. Unlike before, `--check` does **not**
download anything -- it just reports what is already present.

---

## Weights

Weights are auto-downloaded from HuggingFace on first run and cached locally:

| Model | HF repo | Local path |
|-------|---------|------------|
| Phi-4-mini | `freeideas/merv-phi4mini` | `phi4mini/model-q4_k_m.gguf` |
| Gemma 4 E4B | `freeideas/merv-gemma4e4b` | `gemma4e4b/model-q4_k_m.gguf` |
| Qwen 3.5-4B | `freeideas/merv-qwen3.5-4b-mlx` | `qwen3.5-4b/mlx-4bit/` |
| gpt-oss 20B | `freeideas/merv-gpt-oss-20b` | `gpt-oss/model-mxfp4.gguf` |
| Mistral 7B | `freeideas/merv-mistral7b` | `mistral7b/model-q4_k_m.gguf` |

`serve.py` checks each local path at startup; if the file or directory is absent
it downloads from HF before loading. Subsequent startups use the cached copy.
The `*.gguf`, `*.safetensors`, and `*.npz` patterns in `.gitignore` keep the
weight files out of git.

---

## Ports

| Port | Process |
|------|---------|
| 52840 | `serve.py` -- proxy / in-process server + static UI |
| 52841 | `llama-server` -- Phi-4-mini (Mac only; subprocess mode) |
| 52842 | `mlx_lm.server` -- Qwen 3.5-4B (Mac only) |
| 52843 | `llama-server` -- Gemma 4 E4B (Mac only; subprocess mode) |
| 52844 | `llama-server` -- gpt-oss 20B (Mac only; subprocess mode) |
| 52845 | `llama-server` -- Mistral 7B (Mac only; subprocess mode) |

On Linux/Windows there are no subprocess ports -- the model runs inside `serve.py`.

---

## The models

The models were fine-tuned on the Mervin/Mervis persona dataset
(`mervin_mervis_finetune.csv`) -- most on AWS SageMaker, with newer ones trained
on Google Colab. The fine-tuned weights are distributed as GGUF
(Q4\_K\_M and Q8\_0) plus, for qwen, HF safetensors in `qwen3.5-4b/merged_model/`.

`gguf_to_mlx.py` converts GGUF/HF weights to MLX 4-bit for the Mac (qwen). See
the per-model `README.md` files and the script's header for conversion quirks.

---

## Notes

- **Tag cleanup** -- phi and qwen sometimes mangle the `<Mervin>`/`<Mervis>`
  tags, so both `serve.py` and `index.html` apply regex fixes before rendering.
  Gemma 4, on the other hand, appears to get the tags right every time -- we have
  not seen it slip up.
- **Logs** -- every request/response is appended to `logs/YYYY-MM-DD-HHZ.log` as
  newline-delimited JSON.
- **Reverse proxy** -- `index.html` derives its API base from the URL path, so it
  works both at the root and behind a prefix (e.g. a Caddy `/merv/` relay).
