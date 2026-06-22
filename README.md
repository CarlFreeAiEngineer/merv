# Mervin/Mervis -- Cross-Platform Model Arena

A local chat arena that runs fine-tuned LLMs side-by-side. Every response comes
back as two characters: **Mervin** (a sardonic pessimist) and **Mervis** (a
relentless optimist), wrapped in `<Mervin>...</Mervin><Mervis>...</Mervis>` tags.

One `serve.py` runs on **all three** of our hosts -- macOS, Linux, and Windows --
and adapts its inference backend automatically to whatever the host can do.

---

## One file, three hosts

`serve.py` detects the host's capabilities at startup and picks a backend per
model. You do not configure anything by hand:

| Model | macOS (Apple Silicon) | Linux (CPU) | Windows (CPU) |
|-------|-----------------------|-------------|---------------|
| Phi-4-mini | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |
| Gemma 4 E4B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | `llama-cpp-python` in-process |
| Qwen 3.5-4B | `mlx_lm.server` (MLX) | **not available** | **not available** |

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

The repo does **not** contain the model weights (they are multi-GB; see
`.gitignore`). Place the weights on each host at the paths below, then:

### Windows / Linux
```bash
uv run serve.py
```
`uv` reads the inline script metadata and installs `llama-cpp-python` on first run.

### macOS
```bash
python3 serve.py          # uses /opt/homebrew/bin/llama-server + mlx_lm
```
Requires `brew install llama.cpp` and `pip install mlx-lm`. Convert the qwen
weights to MLX once with `python3 gguf_to_mlx.py qwen` (see below).

Then open <http://localhost:52836>.

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
| `MERV_PORT` | `52836` | listen port |
| `MERV_THREADS` | `4` | CPU threads for the in-process backend |
| `MERV_LLAMA_BACKEND` | `auto` | `auto` \| `server` \| `inproc` -- force how phi/gemma run |

Run `python3 serve.py --check` (or `uv run serve.py --check`) to print the
detected backend plan for the current host without loading any models.

---

## Expected weight locations

```
phi4mini/model-q4_k_m.gguf        # or model-q8_0.gguf
gemma4e4b/model-q4_k_m.gguf       # or model-q8_0.gguf
qwen3.5-4b/mlx-4bit/              # MLX dir (Mac); or merged_model/ HF safetensors
```

`serve.py` uses the first existing path for each model and skips any model whose
weights are missing.

---

## Ports

| Port | Process |
|------|---------|
| 52836 | `serve.py` -- proxy / in-process server + static UI |
| 52837 | `llama-server` or `mlx_lm.server` -- Phi-4-mini (Mac only; subprocess mode) |
| 52838 | `mlx_lm.server` -- Qwen 3.5-4B (Mac only) |
| 52839 | `llama-server` -- Gemma 4 E4B (Mac only; subprocess mode) |

On Linux/Windows there are no subprocess ports -- the model runs inside `serve.py`.

---

## The models

All three were fine-tuned on AWS SageMaker on the Mervin/Mervis persona dataset
(`mervin_mervis_finetune.csv`). The fine-tuned weights are distributed as GGUF
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
