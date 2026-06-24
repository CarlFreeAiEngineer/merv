# Mervin/Mervis -- Cross-Platform Model Arena

A local chat arena for fine-tuned LLMs. Each model gets its own column; click a
column to make that model the active one and chat. Every response comes back as
two characters: **Mervin** (a sardonic pessimist) and **Mervis** (a relentless
optimist), wrapped in `<Mervin>...</Mervin><Mervis>...</Mervis>` tags.

**Shared conversations, streamed to everyone.** All state lives in a SQLite file
(`chat_history.db`) and every client repaints itself purely by polling that
state, so **everyone using the arena sees the same thing**. A reply is written
into the database token by token, and each browser re-reads it about once a
second -- so all viewers watch the answer stream in, not just whoever asked. Each
column has an **Erase** button that wipes that model's conversation for everyone,
and each reply shows its tokens/sec. The full transcript is always stored and
shown, but only the **last 3 turns** (the new message plus the previous two
exchanges) are sent to the model as context -- so on the 4th message the model no
longer sees the 1st exchange.

**One queue, one model.** Messages and model switches are appended to a request
queue, and a single worker drains it in order, so exactly one model is resident
and one reply is generated at a time. The text field is disabled while a reply is
generating or a model is loading (you can type again once it is done). A chat runs
against whatever model is active when the worker reaches it.

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

### Hardware fit (per-model manifest)

Every model dir has a `model.json` describing what the model needs (e.g.
`min_ram_gb`). At startup `serve.py` reads the host's RAM and **skips any model
that won't fit** -- it is not downloaded and shows as unavailable in the
dropdown. Only models that fit are fetched (smallest first), and **exactly one
model is resident in memory at a time** -- switching unloads the current model
and loads the next. Tune the thresholds by editing each `model.json`.

---

## One file, three hosts

`serve.py` detects the host's capabilities at startup and picks a backend per
model. You do not configure anything by hand:

| Model | macOS (Apple Silicon) | Linux (CPU) | Windows (CPU) |
|-------|-----------------------|-------------|---------------|
| Phi-4-mini | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | bundled `llama-server.exe` |
| Gemma 4 E4B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | bundled `llama-server.exe` |
| Gemma 4 E2B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | bundled `llama-server.exe` |
| Qwen 2.5 7B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | bundled `llama-server.exe` |
| Mistral 7B | `llama-server` (Metal GPU) | `llama-cpp-python` in-process | bundled `llama-server.exe` |

How the choice is made:

- Every model is a GGUF run through llama.cpp: if a `llama-server` binary is
  found (the bundled Windows copy, or a Mac with `brew install llama.cpp`), it is
  launched as a subprocess and proxied; all such backends stay resident so
  switching is instant. Otherwise the model runs in-process with
  `llama-cpp-python` on CPU, loading one model at a time and swapping on switch.

---

## Running it

**Download principle: ascending order of size.** Model weights are downloaded
automatically from HuggingFace in **ascending order of size -- smallest first**.
The server comes up as soon as the smallest model is ready, then fetches the rest
in the background, smaller before larger. This way **you can start chatting with
the smallest model while the others are still downloading**; each model becomes
selectable in the dropdown the moment its own weights finish landing. A `uv`
binary for each platform is bundled in `bin/` -- no Python or pip installation
required.

By default it starts **web-only**. Pass `--cli` for a built-in terminal chat
alongside the web UI: type to chat, `/model` to list models and their download
state, `/model <name>` to switch (e.g. `/model mistral`), `/clear` to erase the
current model's shared conversation, `/quit` to exit. The CLI is a thin client
over the same endpoints as the web UI, so the two **share each model's
transcript** -- a message typed in one shows up in the other. Every reply -- web
and CLI -- ends with a tokens/sec readout.

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
`serve.py`, then start the server. Windows uses the llama.cpp server in
`bin/llama.cpp`; `run.bat` downloads the tested Windows CPU build if it is
missing. Linux uses the `llama-cpp-python` CPU wheel path and refuses to
source-build it. On macOS, install `llama.cpp` for Metal GPU offload on
phi/gemma/mistral (optional but recommended):
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
| `MERV_NO_REFRESH` | _(unset)_ | if set, skip the startup staleness check and never re-fetch cached weights that changed on HuggingFace (offline pin) |

Command-line flags:

| Flag | Meaning |
|------|---------|
| `--web` | run the web server only; this is the default when no mode flag is given |
| `--cli` | run the web server and attach terminal chat |
| `--port <n>` | listen port; overrides `MERV_PORT` and the `52840` default |
| `--check` | print the detected backend plan and per-model state, then exit (no downloads, no models loaded) |
| `--help` | print command-line help and exit |

Every run prints the command-line flags before doing anything else. Run
`./run.sh --check` or `run.bat --check` to print the plan for the current host.
`--check` does **not** download anything -- it just reports what is already
present.

To shut down a running web server cleanly from the same machine:
```bash
curl -X POST http://127.0.0.1:52840/shutdown
```

The shutdown endpoint only accepts localhost requests.

---

## Weights

Weights are auto-downloaded from HuggingFace on first run and cached locally:

| Model | HF repo | Local path |
|-------|---------|------------|
| Phi-4-mini | `freeideas/merv-phi4mini` | `phi4mini/model-q4_k_m.gguf` |
| Gemma 4 E4B | `freeideas/merv-gemma4e4b` | `gemma4e4b/model-q4_k_m.gguf` |
| Gemma 4 E2B | `freeideas/merv-gemma4e2b` | `gemma4e2b/model-q4_k_m.gguf` |
| Qwen 2.5 7B | `freeideas/merv-qwen2.5-7b` | `qwen2.5-7b/model-q4_k_m.gguf` |
| Mistral 7B | `freeideas/merv-mistral7b` | `mistral7b/model-q4_k_m.gguf` |

`serve.py` checks each local path at startup; if the file or directory is absent
it downloads from HF before loading. Subsequent startups use the cached copy.
The `*.gguf`, `*.safetensors`, and `*.npz` patterns in `.gitignore` keep the
weight files out of git.

**Staleness refresh.** HuggingFace weights can change. At startup `serve.py`
compares each cached GGUF against its HF copy (by size, then sha256) and removes
any that no longer match, so the normal smallest-first download path re-fetches
the current version. The check is network-graceful -- if HF is unreachable, or
`MERV_NO_REFRESH` is set, cached files are left untouched and offline hosts keep
serving what they have. Each verified file's hash is cached in a `<file>.sha256`
sidecar (git-ignored) so an unchanged multi-GB file is hashed at most once.

---

## Ports

| Port | Process |
|------|---------|
| 52840 | `serve.py` -- proxy / in-process server + static UI |
| 52841 | `llama-server` -- Phi-4-mini (Mac only; subprocess mode) |
| 52842 | `llama-server` -- Qwen 2.5 7B (Mac only; subprocess mode) |
| 52843 | `llama-server` -- Gemma 4 E4B (Mac only; subprocess mode) |
| 52845 | `llama-server` -- Mistral 7B (Mac only; subprocess mode) |
| 52846 | `llama-server` -- Gemma 4 E2B (Mac only; subprocess mode) |

On Linux/Windows there are no subprocess ports -- the model runs inside `serve.py`.

---

## The models

The models were fine-tuned on the Mervin/Mervis persona dataset
(`mervin_mervis_finetune.csv`) and distributed as **Q4_K_M GGUF**. The
Mervin/Mervis behavior is driven entirely by fine-tuning -- there is **no system
prompt** at train or inference time. The newer models train on Google Colab; see
each per-model folder's `README.md` and `finetune_*.ipynb` for the exact pipeline.

---

## Notes

- **Tags** -- the models are fine-tuned to emit clean `<Mervin>`/`<Mervis>` tags
  directly, so the old regex tag-fixups have been removed from both `serve.py`
  and `index.html`.
- **Endpoints** -- `POST /enqueue` adds a chat message or a model switch to the
  queue and returns immediately; `GET /history[?model=<key>]` reads a transcript
  (including the partially-streamed reply); `POST /clear` erases one; `GET
  /request?id=<n>` reports a queued request's status (the CLI waits on it). The
  whole arena snapshot -- resident model, `loading`, `busy`, the queue, and a
  per-model revision counter -- comes from `GET /health`, which every client polls
  about once a second; a client only refetches a column when its revision moved.
- **Worker + queue** -- a single worker thread drains the request queue in order,
  so switching and generation are serialized with no lock: exactly one model is
  resident and one reply streams at a time. A reply is written into the database
  as it generates (`status` flips `streaming` -> `done`, with `n_tokens`/`gen_ms`
  for the tokens/sec readout). While a model loads, `/health` reports `loading` and
  every client shows "Loading X..." with inputs disabled. The text field is also
  disabled while a reply is generating, and re-enabled when it finishes.
- **Logs** -- every request/response is appended to `logs/YYYY-MM-DD-HHZ.log` as
  newline-delimited JSON.
- **Reverse proxy** -- `index.html` derives its API base from the URL path, so it
  works both at the root and behind a prefix (e.g. a Caddy `/merv/` relay).
