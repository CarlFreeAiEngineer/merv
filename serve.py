#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "llama-cpp-python==0.3.30; sys_platform == 'linux'",
# ]
# ///
"""
Unified Mervin/Mervis model-switching server -- one file for all three hosts.

Backend selection is automatic, decided at startup from what the host can do:

  GGUF models (phi / gemma / mistral / ...)
      * if a `llama-server` binary is present (e.g. Apple Silicon Mac with
        `brew install llama.cpp`, or the bundled Windows CPU build) it is
        launched as a subprocess and proxied -- this gives Metal GPU offload.
      * otherwise the model is run in-process with llama-cpp-python (Linux CPU).

  **Exactly one model is resident in memory at a time**, regardless of backend.
  Switching unloads the current model (stops its llama-server / frees the
  in-process model) and loads the new one. Generation and switching share one
  lock, so a swap never happens mid-response.

Run it the way each host already does:
  Windows         : uv run serve.py        (uses bundled llama-server.exe)
  Linux           : uv run serve.py        (uv installs llama-cpp-python)
  Mac             : python3 serve.py        (uses brew llama-server)

Use run.bat / run.sh rather than calling uv directly for normal launches.
Windows uses the bundled llama.cpp server under bin/; Linux points uv at the
llama-cpp-python CPU wheel index and disables source builds for that package.

Weights download lazily, smallest model first: the server starts as soon as the
smallest model is ready and fetches the rest in the background, marking each
model selectable the moment its weights land.

By default, serve.py runs web-only. Pass --cli to also drop into a terminal chat
(type to chat, /model to list, /model <name> to switch) alongside the web UI --
both share one serialization point so inference and model swaps never overlap.

Command-line flags:
  --web        run the web server only (default)
  --cli        run the web server plus terminal chat
  --port <n>   listen port (overrides MERV_PORT and the 52840 default)
  --check      print the detected backend plan and exit (no downloads, no models)
  --help       print command-line help and exit

Environment overrides:
  MERV_HOST           bind address (default 0.0.0.0 on macOS, else 127.0.0.1)
  MERV_PORT           listen port  (default 52840; --port wins)
  MERV_THREADS        CPU threads for the in-process backend (default 4)
  MERV_LLAMA_BACKEND  auto | server | inproc -- how phi/gemma run (default auto:
                      use the llama-server binary if present, else in-process)
"""

import sys
import os
import json
import signal
import threading
import re
import time
import shutil
import subprocess
import http.client
import uuid
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import urlopen, Request
from urllib.error import URLError, HTTPError
from datetime import datetime, timezone

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT     = int(os.environ.get('MERV_PORT', '52840'))   # may be overridden by --port
HOST     = os.environ.get('MERV_HOST', '0.0.0.0' if sys.platform == 'darwin' else '127.0.0.1')
THREADS  = int(os.environ.get('MERV_THREADS', '4'))

##############################################################################
# Host capability detection
##############################################################################

def find_llama_server():
    """Locate a llama-server binary. Prefer the bundled copy when present."""
    bundled = os.path.join(BASE_DIR, 'bin', 'llama.cpp',
                           'llama-server.exe' if os.name == 'nt' else 'llama-server')
    if os.path.isfile(bundled):
        return bundled
    found = shutil.which('llama-server')
    if found:
        return found
    for cand in ('/opt/homebrew/bin/llama-server',
                 '/usr/local/bin/llama-server',
                 '/usr/bin/llama-server'):
        if os.path.isfile(cand):
            return cand
    return None


LLAMA_SERVER  = find_llama_server()
LLAMA_BACKEND = os.environ.get('MERV_LLAMA_BACKEND', 'auto').lower()  # auto|server|inproc


##############################################################################
# Model catalogue.  Order here is the startup / default-active order.
##############################################################################

MODELS = {
    'phi': {
        'name': 'Phi-4-mini',
        'kind': 'llama',
        'port': 52841,
        'gguf': [
            os.path.join(BASE_DIR, 'phi4mini', 'model-q4_k_m.gguf'),
            os.path.join(BASE_DIR, 'phi4mini', 'model-q8_0.gguf'),
        ],
    },
    'qwen': {
        'name': 'Qwen 2.5 7B',
        'kind': 'llama',
        'port': 52842,
        'gguf': [
            os.path.join(BASE_DIR, 'qwen2.5-7b', 'model-q4_k_m.gguf'),
        ],
    },
    'gemma': {
        'name': 'Gemma 4 E4B',
        'kind': 'llama',
        'port': 52843,
        'gguf': [
            os.path.join(BASE_DIR, 'gemma4e4b', 'model-q4_k_m.gguf'),
            os.path.join(BASE_DIR, 'gemma4e4b', 'model-q8_0.gguf'),
        ],
    },
    'gemma2b': {
        'name': 'Gemma 4 E2B',
        'kind': 'llama',
        'port': 52846,
        'gguf': [
            os.path.join(BASE_DIR, 'gemma4e2b', 'model-q4_k_m.gguf'),
        ],
    },
    'mistral': {
        'name': 'Mistral 7B',
        'kind': 'llama',
        'port': 52845,
        'gguf': [
            os.path.join(BASE_DIR, 'mistral7b', 'model-q4_k_m.gguf'),
        ],
    },
}


def first_gguf(cfg):
    for p in cfg.get('gguf', []):
        if os.path.isfile(p):
            return p
    return None


##############################################################################
# Hardware fit -- each model dir carries a model.json with its requirements, so
# serve.py skips models that cannot run on this host (won't download or load).
##############################################################################

def total_ram_gb():
    """Total physical RAM in GB, or None if it can't be determined."""
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.memsize'],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) / 1e9
        if os.name == 'nt':
            import ctypes
            class MS(ctypes.Structure):
                _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                            ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                            ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                            ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                            ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]
            m = MS(); m.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
            return m.ullTotalPhys / 1e9
        return os.sysconf('SC_PHYS_PAGES') * os.sysconf('SC_PAGE_SIZE') / 1e9
    except Exception:
        return None


HOST_RAM_GB = total_ram_gb()


def model_manifest(key):
    """Load <model_dir>/model.json (hardware requirements). {} if absent."""
    paths = MODELS.get(key, {}).get('gguf', [])
    if not paths:
        return {}
    try:
        with open(os.path.join(os.path.dirname(paths[0]), 'model.json'), encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def fits_here(key):
    """False if this host clearly can't run the model (per its model.json)."""
    if HOST_RAM_GB is None:
        return True                       # unknown RAM -> don't gate
    need = model_manifest(key).get('min_ram_gb')
    return need is None or HOST_RAM_GB >= need


##############################################################################
# HuggingFace weight download (runs at startup when weights are absent)
##############################################################################

HF_WEIGHTS = {
    'phi': {
        'kind':      'file',
        'repo':      'freeideas/merv-phi4mini',
        'filename':  'model-q4_k_m.gguf',
        'local':     os.path.join(BASE_DIR, 'phi4mini', 'model-q4_k_m.gguf'),
        'approx_gb': 2.4,
    },
    'gemma': {
        'kind':      'file',
        'repo':      'freeideas/merv-gemma4e4b',
        'filename':  'model-q4_k_m.gguf',
        'local':     os.path.join(BASE_DIR, 'gemma4e4b', 'model-q4_k_m.gguf'),
        'approx_gb': 5.0,
    },
    'gemma2b': {
        'kind':      'file',
        'repo':      'freeideas/merv-gemma4e2b',
        'filename':  'model-q4_k_m.gguf',
        'local':     os.path.join(BASE_DIR, 'gemma4e2b', 'model-q4_k_m.gguf'),
        'approx_gb': 3.1,
    },
    'qwen': {
        'kind':      'file',
        'repo':      'freeideas/merv-qwen2.5-7b',
        'filename':  'model-q4_k_m.gguf',
        'local':     os.path.join(BASE_DIR, 'qwen2.5-7b', 'model-q4_k_m.gguf'),
        'approx_gb': 4.7,
    },
    'mistral': {
        'kind':      'file',
        'repo':      'freeideas/merv-mistral7b',
        'filename':  'model-q4_k_m.gguf',
        'local':     os.path.join(BASE_DIR, 'mistral7b', 'model-q4_k_m.gguf'),
        'approx_gb': 4.4,
    },
}


def weights_present(key):
    """True if this model's GGUF is already on disk."""
    cfg = HF_WEIGHTS.get(key)
    return bool(cfg) and os.path.isfile(cfg['local'])


# Background downloads yield to in-flight inference. Writing a multi-GB file
# evicts the mmap'd model from the OS page cache, which starves generation on a
# RAM-tight host. While any generation is running we stop reading the download
# socket between chunks, so TCP backpressure pauses the transfer. Only the
# streamed path is pausable; the hf_hub_download fallback is not.
_infer_active    = 0
_infer_lock      = threading.Lock()
_downloads_paused = threading.Event()   # set => pause streamed downloads


def infer_enter():
    global _infer_active
    with _infer_lock:
        _infer_active += 1
        _downloads_paused.set()


def infer_exit():
    global _infer_active
    with _infer_lock:
        _infer_active = max(0, _infer_active - 1)
        if _infer_active == 0:
            _downloads_paused.clear()


def _streamed_download(cfg):
    """Stream the file to <local>.part, pausing between chunks while inference is
    in flight, then atomically move it into place. Raises on any error."""
    from huggingface_hub import hf_hub_url
    url  = hf_hub_url(repo_id=cfg['repo'], filename=cfg['filename'])
    dst  = cfg['local']
    part = dst + '.part'
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    req = Request(url, headers={'User-Agent': 'merv-serve'})
    try:
        with urlopen(req, timeout=60) as resp, open(part, 'wb') as f:
            while True:
                while _downloads_paused.is_set():
                    time.sleep(0.1)          # yield the disk/cache to inference
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                f.write(chunk)
        os.replace(part, dst)
    except BaseException:
        try:
            os.remove(part)
        except OSError:
            pass
        raise


def download_one(key):
    """Download one model's weights (blocking). Returns True on success or if the
    weights are already present; False on failure or if huggingface_hub is absent."""
    cfg = HF_WEIGHTS.get(key)
    if not cfg:
        return False
    if weights_present(key):
        return True
    print(f'[serve] {key}: downloading {cfg["filename"]} from {cfg["repo"]} ...', flush=True)
    try:
        _streamed_download(cfg)                # pausable; preferred
        print(f'[serve] {key}: download complete', flush=True)
        return True
    except Exception as e:
        print(f'[serve] {key}: streamed download failed ({e}); '
              f'falling back to hf_hub_download', flush=True)
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print('[serve] huggingface_hub not installed -- cannot download', flush=True)
        return False
    try:
        os.makedirs(os.path.dirname(cfg['local']), exist_ok=True)
        hf_hub_download(repo_id=cfg['repo'], filename=cfg['filename'],
                        local_dir=os.path.dirname(cfg['local']))
        print(f'[serve] {key}: download complete', flush=True)
        return True
    except Exception as e:
        print(f'[serve] {key}: download failed: {e}', flush=True)
        return False


def download_queue():
    """Models that still need downloading and can run on this host, smallest first."""
    q = [key for key in HF_WEIGHTS if not weights_present(key) and fits_here(key)]
    q.sort(key=model_sort_key)
    return q


def model_sort_key(key):
    """Smallest models first, then stable catalogue order for ties."""
    keys = list(MODELS)
    return (HF_WEIGHTS.get(key, {}).get('approx_gb', 999), keys.index(key))


def model_keys_by_size():
    return sorted(MODELS, key=model_sort_key)


##############################################################################
# Message helpers
##############################################################################

def content_of(message):
    """Some backends put text under 'reasoning' instead of 'content'."""
    return (message.get('content') or message.get('reasoning')
            or message.get('reasoning_content') or '')


##############################################################################
# Backends
##############################################################################

class ProxyBackend:
    """Runs the `llama-server` binary as a subprocess and proxies to it
    (Metal GPU offload on a Mac, bundled CPU build on Windows).

    Single resident slot: at most one ProxyBackend is booted at a time
    (`_running`). Switching models stops the current subprocess and starts the
    new one, so only one model is ever in memory. The switch/generation lock
    (GEN_LOCK) ensures a swap never happens mid-response.
    """
    persistent = False
    needs_lock = False
    _running   = None        # the single ProxyBackend currently booted (one slot)

    def __init__(self, key, cmd, port, ready_kind):
        self.key        = key
        self.cmd        = cmd
        self.port       = port
        self.ready_kind = ready_kind     # 'llama' -> readiness via /health
        self.proc       = None
        self.available  = True           # weights present => selectable; boots on activate()

    def boot(self):
        print(f'[serve] loading {self.key} on port {self.port} ...', flush=True)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            print(f'[serve] {self.key} ready on port {self.port}', flush=True)
            return True
        except TimeoutError:
            out = self.proc.stdout.read(4096).decode('utf-8', 'replace') if self.proc.stdout else ''
            print(f'[serve] {self.key} failed to start:\n{out}', flush=True)
            self.stop()
            return False

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _wait_ready(self, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            try:
                if self.ready_kind == 'llama':
                    resp = urlopen(f'http://127.0.0.1:{self.port}/health', timeout=2)
                    if resp.status == 200 and json.loads(resp.read()).get('status') == 'ok':
                        return
                else:
                    resp = urlopen(f'http://127.0.0.1:{self.port}/v1/models', timeout=2)
                    if resp.status == 200:
                        return
            except (URLError, OSError, json.JSONDecodeError):
                pass
            time.sleep(2)
        raise TimeoutError(f'{self.key} not ready in {timeout}s')

    def activate(self):
        # Single resident slot: stop whatever proxy is running, then boot this one.
        # The caller holds GEN_LOCK, so this never races a generation or a switch.
        if ProxyBackend._running is self and self._alive():
            return
        other = ProxyBackend._running
        if other is not None and other is not self:
            print(f'[serve] unloading {other.key} (single model in memory)', flush=True)
            other.stop()
        ProxyBackend._running = None
        if not self._alive() and not self.boot():
            raise RuntimeError(f'{self.key} failed to start')
        ProxyBackend._running = self

    def _post(self, payload, stream):
        body = json.dumps(payload).encode()
        conn = http.client.HTTPConnection('127.0.0.1', self.port, timeout=300)
        conn.request('POST', '/v1/chat/completions', body=body,
                     headers={'Content-Type': 'application/json'})
        return conn  # caller reads resp then closes

    def complete(self, messages, params):
        payload = {'messages': messages, 'stream': False, **params}
        conn = self._post(payload, stream=False)
        try:
            resp = conn.getresponse()
            data = resp.read()
            if resp.status >= 400:
                raise RuntimeError(data.decode('utf-8', 'replace') or f'HTTP {resp.status}')
            result = json.loads(data)
            for ch in result.get('choices', []):
                msg = ch.get('message')
                if isinstance(msg, dict) and not msg.get('content'):
                    fallback = msg.get('reasoning') or msg.get('reasoning_content')
                    if fallback:
                        msg['content'] = fallback
            return result
        finally:
            conn.close()

    def stream(self, messages, params):
        payload = {'messages': messages, 'stream': True, **params}
        conn = self._post(payload, stream=True)
        resp = conn.getresponse()
        if resp.status >= 400:
            data = resp.read().decode('utf-8', 'replace')
            conn.close()
            raise RuntimeError(data or f'HTTP {resp.status}')
        buf = ''
        try:
            while True:
                raw = resp.read(1024)
                if not raw:
                    break
                buf += raw.decode('utf-8', 'replace')
                while '\n' in buf:
                    line, buf = buf.split('\n', 1)
                    line = line.strip()
                    if not line.startswith('data: '):
                        continue
                    data = line[6:]
                    if data == '[DONE]':
                        return
                    try:
                        chunk = json.loads(data)
                    except json.JSONDecodeError:
                        continue
                    for ch in chunk.get('choices', []):
                        delta = ch.get('delta')
                        if isinstance(delta, dict) and not delta.get('content'):
                            fallback = delta.get('reasoning') or delta.get('reasoning_content')
                            if fallback:
                                delta['content'] = fallback
                    yield chunk
        finally:
            conn.close()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()
        self.proc = None


class InProcBackend:
    """Runs a GGUF model in-process with llama-cpp-python (Windows / Linux, CPU).

    All in-process backends share a single resident model slot, so switching
    unloads the current model and loads the requested one. Generation is
    serialized through a shared lock.
    """
    persistent = False
    needs_lock = True

    _llm     = None
    _current = None

    def __init__(self, key, path):
        self.key       = key
        self.path      = path
        self.available = True

    def boot(self):
        pass  # loaded lazily on activate()

    def activate(self):
        # Caller holds GEN_LOCK; _ensure_loaded swaps the single resident model.
        self._ensure_loaded()

    def _ensure_loaded(self):
        if InProcBackend._current == self.key and InProcBackend._llm is not None:
            return
        from llama_cpp import Llama
        if InProcBackend._llm is not None:
            del InProcBackend._llm
            InProcBackend._llm = None
        print(f'[serve] loading {self.key} from {self.path} ({THREADS} threads) ...', flush=True)
        InProcBackend._llm = Llama(
            model_path=self.path,
            n_ctx=2048,
            n_threads=THREADS,
            n_threads_batch=THREADS,
            verbose=False,
        )
        InProcBackend._current = self.key
        print(f'[serve] {self.key} ready', flush=True)

    # complete()/stream() assume the caller holds lock() -- the request handler
    # acquires it around the whole call so it is released deterministically even
    # if the client disconnects mid-stream.
    def complete(self, messages, params):
        self._ensure_loaded()
        return InProcBackend._llm.create_chat_completion(
            messages=messages, stream=False, **params)

    def stream(self, messages, params):
        self._ensure_loaded()
        for chunk in InProcBackend._llm.create_chat_completion(
                messages=messages, stream=True, **params):
            yield chunk

    def stop(self):
        pass


class SpoofBackend:
    """Stand-in for a model that has no weights and no download source.

    Returns a friendly, persona-formatted "can't run here" reply to every
    request so the UI degrades gracefully instead of erroring.
    """
    persistent = False
    needs_lock = False
    available  = False

    def __init__(self, key):
        self.key  = key
        name      = MODELS.get(key, {}).get('name', key)
        self.text = (
            f'<Mervin>Sorry -- {name} is not available on this server, so here '
            f'I sulk.</Mervin>'
            f'<Mervis>No worries at all! Just pick another model and we will have '
            f'a wonderful chat right here!</Mervis>'
        )

    def boot(self):
        pass

    def activate(self):
        pass

    def _envelope(self, streaming):
        obj = 'chat.completion.chunk' if streaming else 'chat.completion'
        key = 'delta' if streaming else 'message'
        return {
            'id': f'spoof-{self.key}',
            'object': obj,
            'model': self.key,
            'choices': [{
                'index': 0,
                key: {'role': 'assistant', 'content': self.text},
                'finish_reason': 'stop',
            }],
        }

    def complete(self, messages, params):
        return self._envelope(streaming=False)

    def stream(self, messages, params):
        yield self._envelope(streaming=True)

    def stop(self):
        pass


class PendingBackend:
    """Placeholder for a model whose weights are still downloading.

    Returns a friendly persona reply telling the user to try again shortly, and
    reports available=False so the UI shows it as 'downloading' rather than ready.
    Replaced by a real backend (via bring_online) once the download finishes.
    """
    persistent = False
    needs_lock = False
    available  = False

    def __init__(self, key):
        self.key  = key
        name      = MODELS.get(key, {}).get('name', key)
        self.text = (
            f'<Mervin>{name} is still downloading to this server. Typical -- kept '
            f'waiting yet again.</Mervin>'
            f'<Mervis>Almost there! Try me again in a moment and I will be ready '
            f'to chat!</Mervis>'
        )

    def boot(self):     pass
    def activate(self): pass

    def _envelope(self, streaming):
        obj = 'chat.completion.chunk' if streaming else 'chat.completion'
        key = 'delta' if streaming else 'message'
        return {'id': f'pending-{self.key}', 'object': obj, 'model': self.key,
                'choices': [{'index': 0,
                             key: {'role': 'assistant', 'content': self.text},
                             'finish_reason': 'stop'}]}

    def complete(self, messages, params):
        return self._envelope(streaming=False)

    def stream(self, messages, params):
        yield self._envelope(streaming=True)

    def stop(self):
        pass


# Resolved at startup: key -> backend instance (always present). A model is a
# real backend when its weights exist, a PendingBackend while downloading, or a
# SpoofBackend when it has no weights and no download source. model_state mirrors
# this for the UI. backends_lock guards swaps of both dicts at runtime.
backends      = {}
model_state   = {}        # key -> 'ready' | 'downloading' | 'pending' | 'unavailable'
active_model  = None
active_lock   = threading.Lock()
backends_lock = threading.Lock()
# Serializes generation and model switching so only one model is ever loaded and
# a swap never runs during an in-flight generation (single model in memory).
GEN_LOCK      = threading.Lock()
shutdown_lock = threading.Lock()
shutdown_started = False


def stop_backends():
    for b in list(backends.values()):
        try:
            b.stop()
        except Exception:
            pass


def begin_shutdown(server):
    global shutdown_started
    with shutdown_lock:
        if shutdown_started:
            return
        shutdown_started = True

    def worker():
        print('[serve] shutdown requested via HTTP', flush=True)
        try:
            server.shutdown()
        finally:
            stop_backends()

    threading.Thread(target=worker, daemon=True).start()


def build_one(key):
    """Build (or rebuild) the backend for a single model from current disk state,
    set its model_state, and return it. Caller boots proxy backends separately."""
    if not fits_here(key):
        backends[key]    = SpoofBackend(key)   # too big for this host -> unavailable
        model_state[key] = 'unavailable'
        return backends[key]
    cfg = MODELS[key]
    path = first_gguf(cfg)
    if path is None:
        backends[key]    = PendingBackend(key) if key in HF_WEIGHTS else SpoofBackend(key)
        model_state[key] = 'pending' if key in HF_WEIGHTS else 'unavailable'
    elif LLAMA_BACKEND == 'server' or (LLAMA_BACKEND == 'auto' and LLAMA_SERVER):
        if not LLAMA_SERVER:
            backends[key] = InProcBackend(key, path)
        else:
            cmd = [LLAMA_SERVER, '--model', path, '--port', str(cfg['port']),
                   '--host', '127.0.0.1', '--ctx-size', '4096',
                   '--n-gpu-layers', '99', '--reasoning', 'off']
            backends[key] = ProxyBackend(key, cmd, cfg['port'], 'llama')
        model_state[key] = 'ready'
    else:
        backends[key]    = InProcBackend(key, path)
        model_state[key] = 'ready'
    return backends[key]


def build_backends():
    for key in model_keys_by_size():
        build_one(key)


def bring_online(key):
    """Weights are present now: build the (selectable) backend. Does NOT load the
    model -- only the active model is resident; the user loads this one by
    switching to it (or it becomes the active model at startup)."""
    with backends_lock:
        build_one(key)


def download_worker(queue):
    """Download the queued models in order (smallest first), lighting each up as
    soon as its weights land."""
    for key in queue:
        if not weights_present(key):
            with backends_lock:
                model_state[key] = 'downloading'
            if not download_one(key):
                with backends_lock:
                    model_state[key] = 'pending'
                continue
        bring_online(key)
        print(f'[serve] {key}: now available', flush=True)


def available_map():
    return {k: bool(getattr(b, 'available', False)) for k, b in backends.items()}


def states_map():
    """Per-model UI state. A 'ready' model whose backend is not yet serving
    (e.g. a proxy mid-boot) is reported as 'downloading'."""
    out = {}
    for k in MODELS:
        st = model_state.get(k, 'unavailable')
        if st == 'ready' and not getattr(backends.get(k), 'available', False):
            st = 'downloading'
        out[k] = st
    return out


##############################################################################
# Request / response log
##############################################################################

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_log_lock = threading.Lock()


def log_event(stage, ip, method, path, request_id=None, chat_id=None, model=None,
              status=None, request_body=None, response_body=None, error=None,
              **extra):
    now = datetime.now(timezone.utc)
    filename = now.strftime('%Y-%m-%d-%HZ.log')
    entry = {
        'ts': now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'stage': stage,
        'ip': ip,
        'method': method,
        'path': path,
    }
    if request_id is not None:
        entry['request_id'] = request_id
    if chat_id is not None:
        entry['chat_id'] = chat_id
    if model is not None:
        entry['model'] = model
    if status is not None:
        entry['status'] = status
    if request_body is not None:
        entry['request'] = request_body
    if response_body is not None:
        entry['response'] = response_body
    if error is not None:
        entry['error'] = error
    entry.update({k: v for k, v in extra.items() if v is not None})
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _log_lock:
        with open(os.path.join(LOG_DIR, filename), 'a', encoding='utf-8') as f:
            f.write(line)


def log_request(ip, method, path, request_body=None, response_body=None):
    log_event('legacy', ip, method, path,
              request_body=request_body, response_body=response_body)


##############################################################################
# HTTP handler
##############################################################################

class ProxyHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=BASE_DIR, **kwargs)

    def log_message(self, format, *args):
        pass

    def do_OPTIONS(self):
        self.send_response(200)
        self._cors_headers()
        self.end_headers()

    def do_GET(self):
        if self.path == '/health':
            with active_lock:
                model = active_model
            self._json_response({
                'status': 'ok' if model else 'loading',
                'model': model,
                'available': available_map(),
                'states': states_map(),
            })
        elif self.path == '/v1/models':
            data = [{'id': k, 'object': 'model'}
                    for k, b in backends.items() if getattr(b, 'available', False)]
            self._json_response({'object': 'list', 'data': data})
        elif self.path == '/':
            self.path = '/index.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        if self.path == '/switch':
            self._handle_switch(body)
        elif self.path == '/v1/chat/completions':
            self._handle_chat(body)
        elif self.path == '/shutdown':
            self._handle_shutdown()
        else:
            self.send_error(404)

    def _handle_shutdown(self):
        ip = self.client_address[0]
        if ip not in ('127.0.0.1', '::1'):
            self._json_response({'error': 'Shutdown is only allowed from localhost'}, 403)
            return
        log_request(ip, 'POST', '/shutdown')
        self._json_response({'status': 'shutting_down'})
        begin_shutdown(self.server)

    def _handle_switch(self, body):
        global active_model
        ip = self.client_address[0]
        try:
            data = json.loads(body)
            key = data.get('model')
            if key not in MODELS:
                self._json_response({'error': f'Unknown model: {key}'}, 400)
                return
            # Weights may have arrived since startup but the backend is still a
            # placeholder -- build it on demand before switching.
            if not getattr(backends[key], 'available', False) and weights_present(key):
                bring_online(key)
            backend = backends[key]
            if not getattr(backend, 'available', False):
                st = model_state.get(key, 'unavailable')
                if st in ('pending', 'downloading'):
                    self._json_response(
                        {'error': f'{MODELS[key]["name"]} is still downloading',
                         'status': 'downloading', 'model': key}, 503)
                else:
                    self._json_response(
                        {'error': f'{MODELS[key]["name"]} is not available on this server',
                         'status': 'unavailable', 'model': key}, 503)
                return
            log_request(ip, 'POST', '/switch', request_body=f'switch to {key}')
            # Load the new model (unloading the previous one) under GEN_LOCK so the
            # swap never runs during an in-flight generation. Only one model is
            # ever resident.
            if not GEN_LOCK.acquire(timeout=300):
                self._json_response({'error': 'Server busy, try again'}, 503)
                return
            try:
                backend.activate()
                with active_lock:
                    active_model = key
            finally:
                GEN_LOCK.release()
            self._json_response({'status': 'ok', 'model': key})
        except Exception as e:
            self._json_response({'error': str(e)}, 500)

    def _handle_chat(self, body):
        ip = self.client_address[0]
        with active_lock:
            key = active_model
        if not key:
            self._json_response({'error': 'Server still starting'}, 503)
            return

        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return

        request_id = req.get('request_id') or uuid.uuid4().hex
        chat_id = req.get('chat_id') or f'server-{request_id}'
        backend = backends.get(key)
        if backend is None or not getattr(backend, 'available', False):
            backend = SpoofBackend(key)   # graceful "can't run here"

        messages = req.get('messages', [])
        params = {
            'max_tokens':  req.get('max_tokens', 256),
            'temperature': req.get('temperature', 0.7),
            'top_p':       req.get('top_p', 0.9),
        }
        stream = req.get('stream', False)

        user_msgs = [m['content'] for m in messages if m.get('role') == 'user']
        log_user_msg = user_msgs[-1] if user_msgs else None
        log_event('http_request', ip, 'POST', '/v1/chat/completions',
                  request_id=request_id, chat_id=chat_id, model=key,
                  request_body=log_user_msg, message_count=len(messages),
                  stream=stream)

        # One model is resident at a time, so generation and switching share a
        # single lock: this serializes generations and guarantees no model swap
        # happens mid-response.
        if not GEN_LOCK.acquire(timeout=300):
            log_event('http_response', ip, 'POST', '/v1/chat/completions',
                      request_id=request_id, chat_id=chat_id, model=key,
                      status=503, response_body='Server busy, try again')
            self._json_response({'error': 'Server busy, try again'}, 503)
            return
        infer_enter()                 # pause background downloads while generating
        try:
            # Re-resolve under the lock (a /switch may have changed the active
            # model), then ensure that model is the single resident one.
            with active_lock:
                key = active_model
            backend = backends.get(key)
            if backend is None or not getattr(backend, 'available', False):
                backend = SpoofBackend(key)
            backend.activate()        # idempotent if already the resident model
            log_event('model_request', ip, 'POST', '/v1/chat/completions',
                      request_id=request_id, chat_id=chat_id, model=key,
                      request_body=log_user_msg, message_count=len(messages),
                      params=params, backend=backend.__class__.__name__)
            if stream:
                self._stream_response(backend, messages, params, ip, log_user_msg,
                                      request_id, chat_id, key)
            else:
                result = backend.complete(messages, params)
                text = content_of(result.get('choices', [{}])[0].get('message', {}))
                log_event('model_response', ip, 'POST', '/v1/chat/completions',
                          request_id=request_id, chat_id=chat_id, model=key,
                          status=200, response_body=text)
                log_event('http_response', ip, 'POST', '/v1/chat/completions',
                          request_id=request_id, chat_id=chat_id, model=key,
                          status=200, response_body=text)
                self._json_response(result)
        except Exception as e:
            log_event('model_response', ip, 'POST', '/v1/chat/completions',
                      request_id=request_id, chat_id=chat_id, model=key,
                      status=500, error=str(e))
            log_event('http_response', ip, 'POST', '/v1/chat/completions',
                      request_id=request_id, chat_id=chat_id, model=key,
                      status=500, error=str(e))
            try:
                self._json_response({'error': str(e)}, 500)
            except Exception:
                pass
        finally:
            infer_exit()              # resume background downloads
            GEN_LOCK.release()

    def _stream_response(self, backend, messages, params, ip, log_user_msg,
                         request_id, chat_id, key):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()

        full = ''
        error = None
        try:
            for chunk in backend.stream(messages, params):
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                piece = delta.get('content') or delta.get('reasoning') or ''
                if piece:
                    full += piece
                self.wfile.write(f'data: {json.dumps(chunk)}\n\n'.encode())
                self.wfile.flush()
            self.wfile.write(b'data: [DONE]\n\n')
            self.wfile.flush()
        except Exception as e:
            error = str(e)
            err_chunk = {'error': error}
            try:
                self.wfile.write(f'data: {json.dumps(err_chunk)}\n\n'.encode())
                self.wfile.write(b'data: [DONE]\n\n')
                self.wfile.flush()
            except Exception:
                pass

        status = 500 if error else 200
        log_event('model_response', ip, 'POST', '/v1/chat/completions',
                  request_id=request_id, chat_id=chat_id, model=key,
                  status=status, response_body=full, error=error)
        log_event('http_response', ip, 'POST', '/v1/chat/completions',
                  request_id=request_id, chat_id=chat_id, model=key,
                  status=200, response_body=full, error=error)

    def _json_response(self, obj, status=200):
        data = json.dumps(obj).encode()
        self.send_response(status)
        self._cors_headers()
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')


class ThreadedHTTPServer(HTTPServer):
    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


##############################################################################
# Entry point
##############################################################################

def describe_plan():
    print(f'[serve] host: {sys.platform}  python: {sys.version.split()[0]}')
    print(f'[serve] bind: http://{HOST}:{PORT}')
    print(f'[serve] llama-server binary: {LLAMA_SERVER or "(none -> in-process llama-cpp-python)"}')
    ram = f'{HOST_RAM_GB:.1f} GB' if HOST_RAM_GB is not None else 'unknown'
    print(f'[serve] host RAM: {ram} (one model resident at a time)')
    for key, b in backends.items():
        kind = type(b).__name__
        st = model_state.get(key, '?')
        extra = ''
        if not fits_here(key):
            need = model_manifest(key).get('min_ram_gb')
            extra = f' <- skipped: needs {need} GB RAM'
        elif isinstance(b, InProcBackend):
            extra = f' <- {b.path}'
        elif isinstance(b, ProxyBackend):
            extra = f' <- port {b.port}: {" ".join(b.cmd[:3])}...'
        print(f'[serve]   {key:8s} {st:11s} {kind:14s}{extra}')


COMMAND_LINE_HELP = """\
[serve] command-line args:
  --web        Run the web server only. This is the default when no mode flag is given.
  --cli        Run the web server and attach the terminal chat.
  --port <n>   Listen on port n. Overrides MERV_PORT and the 52840 default.
  --check      Print the detected backend plan and exit. No downloads or models start.
  --help       Print this help and exit.
"""


def print_command_line_help():
    print(COMMAND_LINE_HELP.rstrip(), flush=True)


def parse_args(argv):
    mode = 'web'
    mode_flags = []
    port = None
    check = False
    help_requested = False

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg in ('--web', '--cli'):
            mode_flags.append(arg)
            mode = arg[2:]
        elif arg == '--port':
            i += 1
            if i >= len(argv):
                print('[serve] --port requires a number, e.g. --port 8080', flush=True)
                sys.exit(2)
            try:
                port = int(argv[i])
            except ValueError:
                print('[serve] --port requires a number, e.g. --port 8080', flush=True)
                sys.exit(2)
        elif arg == '--check':
            check = True
        elif arg in ('--help', '-h'):
            help_requested = True
        else:
            print(f'[serve] unknown argument: {arg}', flush=True)
            sys.exit(2)
        i += 1

    if len(set(mode_flags)) > 1:
        print('[serve] choose one mode: --web or --cli', flush=True)
        sys.exit(2)

    return {
        'mode': mode,
        'port': port,
        'check': check,
        'help': help_requested,
    }


##############################################################################
# Built-in CLI chat -- a thin HTTP client of THIS server, so web and CLI share
# one serialization point (no second inference path). Auto-runs when stdin is a
# terminal; headless runs (systemd, pipes) stay web-only.
##############################################################################

def _norm(s):
    return re.sub(r'[^a-z0-9]', '', s.lower())


def _resolve_model(arg):
    n = _norm(arg)
    for k in MODELS:                                   # exact key / name
        if _norm(k) == n or _norm(MODELS[k]['name']) == n:
            return k
    for k in MODELS:                                   # prefix
        if _norm(k).startswith(n) or _norm(MODELS[k]['name']).startswith(n):
            return k
    return None


def _api_get(base, path):
    with urlopen(base + path, timeout=10) as r:
        return json.loads(r.read())


def _api_switch(base, key):
    req = Request(base + '/switch', data=json.dumps({'model': key}).encode(),
                  headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(req, timeout=120) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _api_chat_stream(host, port, messages):
    """POST a streaming chat to the local server; return (text, usage_tokens|None,
    elapsed_seconds|None) measured across the streamed deltas."""
    payload = json.dumps({'messages': messages, 'stream': True, 'max_tokens': 1024}).encode()
    conn = http.client.HTTPConnection(host, port, timeout=300)
    conn.request('POST', '/v1/chat/completions', body=payload,
                 headers={'Content-Type': 'application/json'})
    resp = conn.getresponse()
    full, buf, usage = '', '', None
    first_t = last_t = None
    try:
        while True:
            raw = resp.read(512)
            if not raw:
                break
            buf += raw.decode('utf-8', 'replace')
            while '\n' in buf:
                line, buf = buf.split('\n', 1)
                line = line.strip()
                if not line.startswith('data: '):
                    continue
                data = line[6:]
                if data == '[DONE]':
                    buf = ''
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                u = chunk.get('usage') or {}
                if u.get('completion_tokens'):
                    usage = u['completion_tokens']
                for ch in chunk.get('choices', []):
                    delta = ch.get('delta', {})
                    piece = delta.get('content') or delta.get('reasoning') or ''
                    if piece:
                        now = time.time()
                        first_t = first_t or now
                        last_t = now
                        full += piece
    finally:
        conn.close()
    elapsed = (last_t - first_t) if (first_t and last_t and last_t > first_t) else None
    return full, usage, elapsed


def _print_reply(reply, usage, elapsed):
    clean = reply
    m = re.search(r'<Mervin>(.*?)</Mervin>', clean, re.DOTALL)
    s = re.search(r'<Mervis>(.*?)</Mervis>', clean, re.DOTALL)
    if m or s:
        if m:
            print(f'\n  Mervin: {m.group(1).strip()}')
        if s:
            print(f'  Mervis: {s.group(1).strip()}')
    else:
        print('\n  ' + reply.strip())
    if elapsed and elapsed > 0:
        toks = usage if usage else max(1, round(len(reply) / 4))
        approx = '' if usage else '~'
        print(f'  [{approx}{toks / elapsed:.1f} tok/s, {elapsed:.1f}s]')


def chat_repl(base):
    host, port = '127.0.0.1', PORT
    history = {k: [] for k in MODELS}

    health = {}
    for _ in range(30):                       # wait briefly for an active model
        try:
            health = _api_get(base, '/health')
        except Exception:
            health = {}
        if health.get('model'):
            break
        time.sleep(0.5)
    active = health.get('model') or next(iter(MODELS))

    print('\n' + '=' * 60)
    print('  Mervin/Mervis CLI chat. Type a message, or:')
    print('    /model            list models and their state')
    print('    /model <name>     switch (e.g. /model mistral)')
    print('    /clear            forget this model\'s history')
    print('    /help             show commands')
    print('    /quit             exit')
    print('=' * 60)

    while True:
        try:
            line = input(f'\n[{active}] you> ').strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if not line:
            continue

        if line.startswith('/'):
            parts = line.split(maxsplit=1)
            cmd = parts[0].lower()
            arg = parts[1].strip() if len(parts) > 1 else ''
            if cmd in ('/quit', '/exit', '/q'):
                return
            elif cmd == '/help':
                print('  /model | /model <name> | /clear | /help | /quit')
            elif cmd == '/clear':
                history[active] = []
                print(f'  cleared {active} history')
            elif cmd == '/model':
                states = {}
                try:
                    states = _api_get(base, '/health').get('states', {})
                except Exception:
                    pass
                if not arg:
                    for k in MODELS:
                        mark = '*' if k == active else ' '
                        print(f'   {mark} {k:8s} {MODELS[k]["name"]:14s} [{states.get(k, "?")}]')
                else:
                    key = _resolve_model(arg)
                    if not key:
                        print(f'  unknown model: {arg}')
                    else:
                        code, resp = _api_switch(base, key)
                        if code == 200:
                            active = key
                            print(f'  switched to {key} ({MODELS[key]["name"]})')
                        else:
                            print(f'  cannot switch: {resp.get("error", "HTTP " + str(code))}')
            else:
                print(f'  unknown command: {cmd}')
            continue

        # Ensure the server is on our model before generating. This goes through
        # the same /switch + locks as the web UI -> single-filed inference.
        code, resp = _api_switch(base, active)
        if code != 200:
            print(f'  ({MODELS[active]["name"]}: {resp.get("error", "not ready")})')
            continue

        history[active].append({'role': 'user', 'content': line})
        messages = history[active]   # no system prompt -- behavior is fine-tuned in
        reply, usage, elapsed = _api_chat_stream(host, port, messages)
        history[active].append({'role': 'assistant', 'content': reply})
        _print_reply(reply, usage, elapsed)


def main():
    global active_model, PORT

    argv = sys.argv[1:]
    print_command_line_help()
    args = parse_args(argv)
    if args['help']:
        return
    if args['port'] is not None:              # overrides MERV_PORT / default
        PORT = args['port']

    build_backends()

    if args['check']:
        describe_plan()
        print('[serve] --check: no downloads, no backends started.')
        return

    describe_plan()

    queue = download_queue()   # missing models, smallest first

    # Need one model resident before serving. If nothing is cached, fetch the
    # smallest now so the user can start chatting while the rest download.
    if not any(getattr(b, 'available', False) for b in backends.values()) and queue:
        first = queue[0]
        print(f'[serve] nothing cached -- fetching smallest ({first}) first ...', flush=True)
        with backends_lock:
            model_state[first] = 'downloading'
        if download_one(first):
            bring_online(first)

    # "available" == weights present == selectable. Exactly one model is loaded
    # into memory at a time; load the smallest available now and serve.
    ready = [k for k in model_keys_by_size() if getattr(backends[k], 'available', False)]
    if not ready:
        print('[serve] ERROR: no models could be made available on this host.', flush=True)
        sys.exit(1)

    active_model = ready[0]
    backends[active_model].activate()         # load ONLY this model (single slot)
    print(f'[serve] active model: {active_model}', flush=True)
    print(f'[serve] selectable:   {ready}', flush=True)

    try:
        server = ThreadedHTTPServer((HOST, PORT), ProxyHandler)
    except OSError as e:
        print(f'[serve] ERROR: cannot bind to {HOST}:{PORT} -- {e}', flush=True)
        print(f'[serve] Try a different port: --port 53840 (or any free port)', flush=True)
        sys.exit(1)

    def cleanup(*_):
        stop_backends()
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    print(f'[serve] listening on http://{HOST}:{PORT}', flush=True)

    # Download the rest in the background, smallest first; each lights up when done.
    remaining = [k for k in queue if not getattr(backends[k], 'available', False)]
    if remaining:
        print(f'[serve] background download queue (smallest first): {remaining}', flush=True)
        threading.Thread(target=download_worker, args=(remaining,), daemon=True).start()

    if args['mode'] == 'cli':
        chat_repl(f'http://127.0.0.1:{PORT}')
        cleanup()
    else:
        try:
            server_thread.join()
        except KeyboardInterrupt:
            cleanup()
        finally:
            stop_backends()


if __name__ == '__main__':
    main()
