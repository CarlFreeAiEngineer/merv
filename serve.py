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
  MERV_NO_REFRESH     if set, never re-fetch cached weights that have changed on
                      HuggingFace (skip the startup staleness check; offline pin)
"""

import sys
import os
import json
import sqlite3
import hashlib
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
from urllib.parse import parse_qs, unquote_plus
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


def nvidia_vram_gb():
    """(total_gb, free_gb) for the primary NVIDIA GPU, or (None, None) if there
    is no usable NVIDIA card / nvidia-smi. Used to decide GPU offload per model."""
    try:
        out = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.total,memory.free',
             '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5)
        if out.returncode != 0 or not out.stdout.strip():
            return (None, None)
        total, free = (float(x) for x in out.stdout.strip().splitlines()[0].split(','))
        return (total / 1024.0, free / 1024.0)        # MiB -> GiB
    except Exception:
        return (None, None)


GPU_TOTAL_GB, GPU_FREE_GB = nvidia_vram_gb()
# Full offload by default (also what Apple Metal wants); an explicit value forces it.
GPU_LAYERS_ENV = os.environ.get('MERV_GPU_LAYERS')

# Windows bundles a llama.cpp server build. We always fetch the GPU-capable
# (CUDA) build and simply run it CPU-only when there is no GPU, so there is one
# build to manage -- no CPU-vs-CUDA branching. (Mac uses brew llama-server with
# Metal; Linux runs llama-cpp-python in-process, so neither downloads this.)
LLAMA_CPP_TAG  = os.environ.get('LLAMA_CPP_TAG', 'b9761')
LLAMA_CPP_CUDA = os.environ.get('LLAMA_CPP_CUDA', '12.4')


def ensure_llama_server():
    """On Windows, make sure the bundled GPU-capable llama.cpp build is complete,
    downloading whatever is missing from the llama.cpp GitHub release: the CUDA
    server (llama-server.exe + ggml-cuda.dll) and the CUDA runtime DLLs (cudart /
    cublas) that ggml-cuda.dll needs. Both are required for GPU offload -- without
    cudart the server silently runs on CPU, so we check for it explicitly. No-op
    elsewhere and when everything is already present."""
    if os.name != 'nt':
        return
    dest = os.path.join(BASE_DIR, 'bin', 'llama.cpp')
    exe  = os.path.join(dest, 'llama-server.exe')
    have_exe    = os.path.isfile(exe)
    have_cudart = any(f.lower().startswith('cudart64') and f.lower().endswith('.dll')
                      for f in (os.listdir(dest) if os.path.isdir(dest) else []))
    server_zip = f'llama-{LLAMA_CPP_TAG}-bin-win-cuda-{LLAMA_CPP_CUDA}-x64.zip'
    cudart_zip = f'cudart-llama-bin-win-cuda-{LLAMA_CPP_CUDA}-x64.zip'
    needed = []
    if not have_exe:
        needed.append(server_zip)
    if not have_cudart:
        needed.append(cudart_zip)
    if not needed:
        return

    import zipfile
    os.makedirs(dest, exist_ok=True)
    base = f'https://github.com/ggml-org/llama.cpp/releases/download/{LLAMA_CPP_TAG}'
    for asset in needed:
        tmp = os.path.join(dest, asset)
        print(f'[serve] downloading {asset} (GPU-capable llama.cpp build) ...', flush=True)
        try:
            req = Request(f'{base}/{asset}', headers={'User-Agent': 'merv-serve'})
            with urlopen(req, timeout=120) as r, open(tmp, 'wb') as f:
                shutil.copyfileobj(r, f)
            with zipfile.ZipFile(tmp) as z:
                z.extractall(dest)
        finally:
            try:
                os.remove(tmp)
            except OSError:
                pass
    if not os.path.isfile(exe):
        print('[serve] ERROR: llama-server.exe missing after download', flush=True)


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
    # Showcase-only: trained + on HF (freeideas/merv-gptoss20b) but its ~12GB IQ2_M
    # GGUF won't run on a 16GB box. NOT in HF_WEIGHTS, so it never downloads; it
    # shows as the last, always-"unavailable" column. See gptoss20b/README.md.
    'gptoss20b': {
        'name': 'GPT-OSS 20B',
        'kind': 'llama',
        'port': 52847,
        'gguf': [
            os.path.join(BASE_DIR, 'gptoss20b', 'model-iq2_m.gguf'),
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


def _cpu_brief():
    """Short CPU model string, e.g. 'Intel Core i7-9750H' or 'Apple M1'."""
    raw = ''
    try:
        if sys.platform == 'darwin':
            raw = subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'],
                                 capture_output=True, text=True, timeout=5).stdout.strip()
        elif os.name == 'nt':
            import winreg
            with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE,
                                r'HARDWARE\DESCRIPTION\System\CentralProcessor\0') as k:
                raw = winreg.QueryValueEx(k, 'ProcessorNameString')[0]
        else:
            with open('/proc/cpuinfo', encoding='utf-8') as f:
                for line in f:
                    if line.lower().startswith('model name'):
                        raw = line.split(':', 1)[1].strip()
                        break
    except Exception:
        raw = ''
    raw = re.sub(r'\(R\)|\(TM\)|\(tm\)', '', raw)
    raw = re.sub(r'\s+@.*$', '', raw)                 # drop "@ 2.60GHz"
    raw = re.sub(r'\bCPU\b|\bProcessor\b', '', raw)
    return re.sub(r'\s+', ' ', raw).strip() or 'CPU'


def _gpu_brief():
    """Short GPU string, e.g. 'GTX 1660 Ti 6G', 'Metal GPU', or None."""
    if sys.platform == 'darwin':
        return 'Metal GPU'
    try:
        out = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                              '--format=csv,noheader,nounits'],
                             capture_output=True, text=True, timeout=5)
        if out.returncode == 0 and out.stdout.strip():
            name, mem = (x.strip() for x in out.stdout.strip().splitlines()[0].split(','))
            name = re.sub(r'NVIDIA|GeForce|with Max-Q Design|\(R\)|\(TM\)', '', name)
            name = re.sub(r'\s+', ' ', name).strip()
            return f'{name} {round(float(mem) / 1024)}G'
    except Exception:
        pass
    return None


def physical_cores():
    """Physical CPU cores (not hyperthreads), or None if it can't be determined.
    os.cpu_count() reports logical processors -- on a hyperthreaded chip that is
    2x the real cores (e.g. an i7-10750H is 6 cores / 12 threads)."""
    try:
        if sys.platform == 'darwin':
            out = subprocess.run(['sysctl', '-n', 'hw.physicalcpu'],
                                 capture_output=True, text=True, timeout=5)
            return int(out.stdout.strip()) or None
        if os.name == 'nt':
            out = subprocess.run(
                ['powershell', '-NoProfile', '-Command',
                 '(Get-CimInstance Win32_Processor | '
                 'Measure-Object -Property NumberOfCores -Sum).Sum'],
                capture_output=True, text=True, timeout=10)
            return int(out.stdout.strip()) or None
        # Linux: count distinct (physical id, core id) pairs in /proc/cpuinfo.
        seen, cur = set(), {}
        with open('/proc/cpuinfo', encoding='utf-8') as f:
            for line in f:
                if ':' not in line:
                    if cur:
                        seen.add((cur.get('physical id'), cur.get('core id')))
                        cur = {}
                    continue
                k, v = (x.strip() for x in line.split(':', 1))
                if k in ('physical id', 'core id'):
                    cur[k] = v
        if cur:
            seen.add((cur.get('physical id'), cur.get('core id')))
        seen.discard((None, None))
        return len(seen) or None
    except Exception:
        return None


def _cores_brief():
    """e.g. '6C/12T' when hyperthreaded, '6 cores', or '12 threads' as a fallback."""
    logical = os.cpu_count()
    phys = physical_cores()
    if phys and logical and phys != logical:
        return f'{phys}C/{logical}T'
    if phys:
        return f'{phys} cores'
    return f'{logical} threads' if logical else ''


def hardware_summary():
    """One-line host hardware blurb for the UI, e.g.
    'Intel Core i7-10750H 6C/12T, 16G RAM, GTX 1660 Ti 6G'."""
    parts = []
    parts.append(f'{_cpu_brief()} {_cores_brief()}'.strip())
    if HOST_RAM_GB is not None:
        parts.append(f'{round(HOST_RAM_GB * 1e9 / 2**30)}G RAM')   # GiB, as people expect
    parts.append(_gpu_brief() or 'no GPU')
    return ', '.join(parts)


HARDWARE = hardware_summary()


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


# Headroom over the raw weight size for the KV cache + compute buffers at the
# 4096 context we launch with. A model only goes fully on the GPU if it fits.
GPU_HEADROOM_GB = 1.3


def gpu_layers_for(key):
    """How many layers to offload to the GPU for this model. '99' = all, '0' =
    CPU-only. We always run the GPU-capable build and just switch offload on/off:

      * macOS         -> '99' (Apple Metal, unified memory)
      * NVIDIA found  -> '99' if the model fits in free VRAM, else '0'
      * no NVIDIA     -> '0' (the CUDA build runs CPU-only)

    A failed GPU launch also falls back to CPU at boot, so the VRAM estimate only
    needs to be roughly right. MERV_GPU_LAYERS overrides everything."""
    if GPU_LAYERS_ENV is not None:
        return GPU_LAYERS_ENV
    if sys.platform == 'darwin':
        return '99'
    if GPU_FREE_GB is None:
        return '0'                        # no NVIDIA GPU -> CPU on the CUDA build
    need = (model_manifest(key).get('size_gb') or 4.0) + GPU_HEADROOM_GB
    return '99' if need <= GPU_FREE_GB else '0'


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


##############################################################################
# Staleness check -- weights on HuggingFace can change. We only download when a
# file is absent, so a cached copy would otherwise be served forever even after
# the repo is updated. At startup we compare each cached GGUF against its HF copy
# and remove any that no longer match, so the normal (smallest-first) download
# path re-fetches the current version.
#
# Network-graceful by design: if HF is unreachable -- or MERV_NO_REFRESH is set --
# cached files are left untouched, so offline hosts keep serving what they have.
# The expensive part (hashing a multi-GB file) is cached in a <file>.sha256
# sidecar keyed on file size, so an unchanged file is only ever hashed once.
##############################################################################

def _sidecar_path(path):
    return path + '.sha256'


def local_sha256(path):
    """sha256 of a local file, cached in a <path>.sha256 sidecar keyed on size so
    an unchanged file is hashed at most once. None if the file is absent."""
    if not os.path.isfile(path):
        return None
    size = os.path.getsize(path)
    sc = _sidecar_path(path)
    try:
        with open(sc, encoding='utf-8') as f:
            rec_sha, rec_size = f.read().split()
        if int(rec_size) == size:
            return rec_sha
    except (OSError, ValueError):
        pass
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 22), b''):
            h.update(chunk)
    digest = h.hexdigest()
    try:
        with open(sc, 'w', encoding='utf-8') as f:
            f.write(f'{digest} {size}')
    except OSError:
        pass
    return digest


_remote_meta_cache = {}


def remote_meta(key):
    """(sha256, size) of this model's GGUF on HuggingFace, or (None, None) if it
    can't be determined (huggingface_hub missing, or HF unreachable)."""
    if key in _remote_meta_cache:
        return _remote_meta_cache[key]
    cfg = HF_WEIGHTS.get(key)
    sha = size = None
    if cfg:
        try:
            from huggingface_hub import HfApi
            info = HfApi().model_info(cfg['repo'], files_metadata=True)
            sib = next((s for s in info.siblings
                        if s.rfilename == cfg['filename']), None)
            if sib and sib.lfs:
                sha  = sib.lfs.get('sha256')
                size = sib.lfs.get('size')
        except Exception:
            pass
    _remote_meta_cache[key] = (sha, size)
    return sha, size


def refresh_stale_weights():
    """Remove cached GGUFs that no longer match their HuggingFace copy so the
    normal download path re-fetches the current version. Does nothing for files
    we can't verify (HF unreachable) or when MERV_NO_REFRESH is set."""
    if os.environ.get('MERV_NO_REFRESH'):
        return
    for key, cfg in HF_WEIGHTS.items():
        local = cfg['local']
        if not os.path.isfile(local):
            continue
        remote_sha, remote_size = remote_meta(key)
        if not remote_sha:
            continue                       # can't verify -> keep cached copy
        # Cheap size check first; only hash when sizes match.
        stale = (remote_size is not None and os.path.getsize(local) != remote_size)
        if not stale:
            stale = local_sha256(local) != remote_sha
        if stale:
            print(f'[serve] {key}: cached weights differ from HuggingFace '
                  f'-> removing to refetch the current version', flush=True)
            for p in (local, _sidecar_path(local)):
                try:
                    os.remove(p)
                except OSError:
                    pass


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


def _record_sidecar(key):
    """Write the size-keyed sha256 sidecar for a freshly downloaded file, reusing
    HF's known hash when available so a multi-GB download is not re-hashed."""
    cfg = HF_WEIGHTS.get(key)
    if not cfg or not os.path.isfile(cfg['local']):
        return
    sha, _ = remote_meta(key)
    try:
        if sha:
            with open(_sidecar_path(cfg['local']), 'w', encoding='utf-8') as f:
                f.write(f'{sha} {os.path.getsize(cfg["local"])}')
        else:
            local_sha256(cfg['local'])     # compute + cache when HF hash unknown
    except OSError:
        pass


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
        _record_sidecar(key)
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
        _record_sidecar(key)
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
    new one, so only one model is ever in memory. The single request worker is the
    only caller, so a swap never happens mid-response.
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

    def _gpu_layers(self):
        if '--n-gpu-layers' in self.cmd:
            return self.cmd[self.cmd.index('--n-gpu-layers') + 1]
        return '0'

    def _force_cpu(self):
        if '--n-gpu-layers' in self.cmd:
            self.cmd[self.cmd.index('--n-gpu-layers') + 1] = '0'

    def _verify_gpu(self):
        """Confirm the booted server actually put a CUDA context on the GPU. A
        CUDA build with the runtime missing (or no device) starts fine but
        silently runs on CPU, so we cross-check nvidia-smi rather than trust
        -ngl. Consumer GPUs under Windows WDDM report per-process memory as
        '[N/A]', so a process merely *appearing* in the compute-apps list is
        proof of GPU use -- the VRAM figure is shown only when available.

        On Apple Silicon the brew llama-server links Metal directly: there is no
        separate runtime that can be missing (the CPU-fallback failure mode this
        guards against is CUDA-only) and no nvidia-smi, so -ngl>0 already means
        Metal offload is in effect."""
        if sys.platform == 'darwin':
            print(f'[serve] {self.key}: GPU offload confirmed (Metal)', flush=True)
            return True
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-compute-apps=pid,used_memory',
                 '--format=csv,noheader,nounits'],
                capture_output=True, text=True, timeout=5)
            for line in out.stdout.strip().splitlines():
                pid, mem = (x.strip() for x in line.split(','))
                if pid == str(self.proc.pid):
                    try:
                        mib = int(float(mem))
                        detail = f'{mib} MiB VRAM' if mib > 0 else 'CUDA context'
                    except ValueError:
                        detail = 'CUDA context (per-process VRAM N/A on this GPU)'
                    print(f'[serve] {self.key}: GPU offload confirmed ({detail})', flush=True)
                    return True
        except Exception:
            pass
        print(f'[serve] {self.key}: WARNING -- asked for GPU but not on the GPU; '
              f'running on CPU (is the CUDA runtime present?)', flush=True)
        return False

    def _boot_once(self):
        gpu = self._gpu_layers()
        where = f'GPU (ngl={gpu})' if gpu != '0' else 'CPU'
        print(f'[serve] loading {self.key} on port {self.port} [{where}] ...', flush=True)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            print(f'[serve] {self.key} ready on port {self.port} [{where}]', flush=True)
            if gpu != '0':
                self._verify_gpu()
            self._warmup_probe()
            return True
        except TimeoutError as e:
            out = self.proc.stdout.read(4096).decode('utf-8', 'replace') if self.proc.stdout else ''
            print(f'[serve] {self.key} failed to start ({e}):\n{out}', flush=True)
            self.stop()
            return False
        except Exception as e:
            out = self.proc.stdout.read(4096).decode('utf-8', 'replace') if self.proc.stdout else ''
            print(f'[serve] {self.key} warmup probe failed ({e}):\n{out}', flush=True)
            self.stop()
            return False

    def boot(self):
        if self._boot_once():
            return True
        # GPU launch failed (e.g. VRAM OOM) -- retry CPU-only so the model still
        # runs. This is the "GPU if it can, else CPU" fallback, decided per model.
        if self._gpu_layers() != '0':
            print(f'[serve] {self.key}: GPU launch failed -- retrying CPU-only', flush=True)
            self._force_cpu()
            return self._boot_once()
        return False

    def _alive(self):
        return self.proc is not None and self.proc.poll() is None

    def _wait_ready(self, timeout=300):
        deadline = time.time() + timeout
        while time.time() < deadline:
            if self.proc is not None and self.proc.poll() is not None:
                raise TimeoutError(f'process exited with code {self.proc.returncode}')
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
        # Only the request worker calls this, so it never races a generation/switch.
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

    def _warmup_probe(self):
        """Run one hidden streaming completion and stop after the first token
        chunk. This proves the model can actually decode before the UI unlocks."""
        payload = {
            'messages': [{'role': 'user', 'content': 'hi'}],
            'stream': True,
            'max_tokens': 2,
            'temperature': 0.1,
            'top_p': 0.1,
        }
        conn = self._post(payload, stream=True)
        try:
            resp = conn.getresponse()
            if resp.status >= 400:
                data = resp.read().decode('utf-8', 'replace')
                raise RuntimeError(data or f'HTTP {resp.status}')
            saw_token = False
            while True:
                line = resp.readline()
                if not line:
                    break
                text = line.decode('utf-8', 'replace').strip()
                if not text.startswith('data: '):
                    continue
                data = text[6:]
                if data == '[DONE]':
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = chunk.get('choices', [{}])[0].get('delta', {})
                piece = delta.get('content') or delta.get('reasoning') or ''
                if piece:
                    saw_token = True
                    break
            if not saw_token:
                raise RuntimeError('warmup probe produced no token')
        finally:
            conn.close()

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
    unloads the current model and loads the requested one. Only the request worker
    calls these, so generation is already serialized.
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
        # Only the request worker calls this; _ensure_loaded swaps the resident model.
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

    # Called only by the request worker (one generation at a time), so no lock.
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
backends_lock = threading.Lock()
# The resident model and the loading model live in the `state` table now; a single
# worker thread drains the request queue, so generation and switching are already
# serialized (exactly one model in memory) with no extra lock.
shutdown_lock = threading.Lock()
shutdown_started = False


##############################################################################
# Shared state in SQLite.  The arena's entire visible state lives in one SQLite
# file (chat_history.db), and the browser and --cli repaint themselves purely by
# reading it -- there is no per-client state to keep in sync:
#
#   messages  -- every chat message per model. An assistant reply is inserted with
#                status 'streaming' and its `content` grows as tokens arrive, so a
#                client polling /history sees the reply stream in. On completion it
#                flips to 'done' and records n_tokens / gen_ms for the tok/s readout.
#   revs      -- per-model revision, bumped on every change (incl. each streaming
#                update) so a client only refetches a column that actually moved.
#   requests  -- the incoming queue: chat messages and model switches. A single
#                worker thread drains it in order, so generation and switching are
#                serialized with no lock -- exactly one model is ever resident, and
#                a chat runs against whatever model is active when the worker
#                reaches it (switches ahead of it in the queue have taken effect).
#   state     -- the single resident model and the model currently loading.
#
# One connection (check_same_thread=False) serialized by chat_lock; WAL mode keeps
# the worker's writes from blocking the frequent pollers.
#
# When prompting a model we send only the last PROMPT_TURNS turns (a turn = one
# user message + its reply): the new user message plus the previous
# PROMPT_TURNS-1 complete exchanges. Older turns stay stored/displayed but are
# not sent, so on the 4th message the model no longer sees the 1st exchange.
##############################################################################

CHAT_DB      = os.path.join(BASE_DIR, 'chat_history.db')
PROMPT_TURNS = 3
chat_lock    = threading.Lock()
_chat_conn   = None
work_ready   = threading.Event()    # set whenever the worker has something to do


def _now():
    return datetime.now(timezone.utc).isoformat()


def _db():
    """The shared SQLite connection, created (with schema) on first use. Caller
    holds chat_lock."""
    global _chat_conn
    if _chat_conn is None:
        conn = sqlite3.connect(CHAT_DB, check_same_thread=False)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA synchronous=NORMAL')
        conn.execute('CREATE TABLE IF NOT EXISTS messages ('
                     'id INTEGER PRIMARY KEY AUTOINCREMENT, model TEXT NOT NULL, '
                     'role TEXT NOT NULL, content TEXT NOT NULL, ts TEXT NOT NULL, '
                     "status TEXT NOT NULL DEFAULT 'done', n_tokens INTEGER, gen_ms INTEGER)")
        conn.execute('CREATE INDEX IF NOT EXISTS idx_messages_model ON messages(model, id)')
        conn.execute('CREATE TABLE IF NOT EXISTS revs ('
                     'model TEXT PRIMARY KEY, rev INTEGER NOT NULL)')
        conn.execute('CREATE TABLE IF NOT EXISTS requests ('
                     'id INTEGER PRIMARY KEY AUTOINCREMENT, kind TEXT NOT NULL, '
                     'model TEXT, content TEXT, request_id TEXT, '
                     "status TEXT NOT NULL DEFAULT 'pending', error TEXT, "
                     'created_ts TEXT NOT NULL, done_ts TEXT)')
        conn.execute('CREATE TABLE IF NOT EXISTS state ('
                     'id INTEGER PRIMARY KEY CHECK (id = 1), '
                     'active_model TEXT, loading_model TEXT)')
        conn.execute('INSERT OR IGNORE INTO state(id, active_model, loading_model) VALUES(1, NULL, NULL)')
        # Migrate older DBs that predate the streaming / tok-s columns.
        cols = {r[1] for r in conn.execute('PRAGMA table_info(messages)')}
        for name, decl in (('status', "TEXT NOT NULL DEFAULT 'done'"),
                           ('n_tokens', 'INTEGER'), ('gen_ms', 'INTEGER')):
            if name not in cols:
                conn.execute(f'ALTER TABLE messages ADD COLUMN {name} {decl}')
        conn.commit()
        _chat_conn = conn
    return _chat_conn


def init_chat_db():
    """Open the DB / create the schema at startup, and clear anything a previous
    run left mid-flight so we start clean: cancel queued/running requests and mark
    any half-streamed reply done (a partial reply is not resumable)."""
    with chat_lock:
        conn = _db()
        conn.execute("UPDATE requests SET status='cancelled', done_ts=? "
                     "WHERE status IN ('pending','running')", (_now(),))
        conn.execute("UPDATE messages SET status='done' WHERE status='streaming'")
        conn.commit()


def _bump_rev(conn, key):
    conn.execute('INSERT INTO revs(model, rev) VALUES(?, 1) '
                 'ON CONFLICT(model) DO UPDATE SET rev = rev + 1', (key,))
    return conn.execute('SELECT rev FROM revs WHERE model=?', (key,)).fetchone()[0]


# ---- resident / loading model -------------------------------------------
def get_state():
    with chat_lock:
        r = _db().execute('SELECT active_model, loading_model FROM state WHERE id=1').fetchone()
    return {'active_model': r[0] if r else None, 'loading_model': r[1] if r else None}


def set_active(model):
    """Mark `model` the resident one and clear any loading flag."""
    with chat_lock:
        conn = _db()
        conn.execute('UPDATE state SET active_model=?, loading_model=NULL WHERE id=1', (model,))
        conn.commit()


def set_loading(model):
    with chat_lock:
        conn = _db()
        conn.execute('UPDATE state SET loading_model=? WHERE id=1', (model,))
        conn.commit()


# ---- request queue ------------------------------------------------------
def enqueue(kind, model=None, content=None, request_id=None):
    with chat_lock:
        conn = _db()
        cur = conn.execute('INSERT INTO requests(kind, model, content, request_id, created_ts) '
                           'VALUES(?,?,?,?,?)', (kind, model, content, request_id, _now()))
        conn.commit()
        rid = cur.lastrowid
    work_ready.set()
    return rid


def next_pending():
    """Claim the oldest pending request (mark it running). None if the queue is empty."""
    with chat_lock:
        conn = _db()
        row = conn.execute("SELECT id, kind, model, content, request_id FROM requests "
                           "WHERE status='pending' ORDER BY id LIMIT 1").fetchone()
        if not row:
            return None
        conn.execute("UPDATE requests SET status='running' WHERE id=?", (row[0],))
        conn.commit()
    return {'id': row[0], 'kind': row[1], 'model': row[2], 'content': row[3], 'request_id': row[4]}


def finish_request(rid, error=None):
    with chat_lock:
        conn = _db()
        conn.execute('UPDATE requests SET status=?, error=?, done_ts=? WHERE id=?',
                     ('error' if error else 'done', error, _now(), rid))
        conn.commit()


def request_status(rid):
    with chat_lock:
        r = _db().execute('SELECT status, error FROM requests WHERE id=?', (rid,)).fetchone()
    return (r[0], r[1]) if r else (None, None)


def queue_view():
    """Pending + running requests, oldest first (for the UI and the busy flag)."""
    with chat_lock:
        rows = _db().execute("SELECT id, kind, model, status FROM requests "
                             "WHERE status IN ('pending','running') ORDER BY id").fetchall()
    return [{'id': i, 'kind': k, 'model': m, 'status': s} for i, k, m, s in rows]


# ---- messages -----------------------------------------------------------
def add_message(key, role, content, status='done'):
    with chat_lock:
        conn = _db()
        cur = conn.execute('INSERT INTO messages(model, role, content, ts, status) VALUES(?,?,?,?,?)',
                           (key, role, content, _now(), status))
        _bump_rev(conn, key)
        conn.commit()
        return cur.lastrowid


def update_message(msg_id, key, content):
    """Grow a streaming reply's content and bump the model's revision."""
    with chat_lock:
        conn = _db()
        conn.execute('UPDATE messages SET content=? WHERE id=?', (content, msg_id))
        _bump_rev(conn, key)
        conn.commit()


def finish_message(msg_id, key, content, n_tokens, gen_ms):
    with chat_lock:
        conn = _db()
        conn.execute("UPDATE messages SET content=?, status='done', n_tokens=?, gen_ms=? WHERE id=?",
                     (content, n_tokens, gen_ms, msg_id))
        _bump_rev(conn, key)
        conn.commit()


def history_clear(key):
    """Erase a model's transcript, bump the revision. Returns the new revision."""
    with chat_lock:
        conn = _db()
        conn.execute('DELETE FROM messages WHERE model=?', (key,))
        rev = _bump_rev(conn, key)
        conn.commit()
        return rev


def _row_to_msg(role, content, status, n_tokens, gen_ms):
    m = {'role': role, 'content': content, 'status': status}
    if n_tokens is not None:
        m['n_tokens'] = n_tokens
    if gen_ms is not None:
        m['gen_ms'] = gen_ms
    return m


def history_snapshot(key):
    """This model's messages (oldest first, incl. any streaming reply) and its rev."""
    with chat_lock:
        conn = _db()
        rows = conn.execute('SELECT role, content, status, n_tokens, gen_ms FROM messages '
                            'WHERE model=? ORDER BY id', (key,)).fetchall()
        r = conn.execute('SELECT rev FROM revs WHERE model=?', (key,)).fetchone()
    return [_row_to_msg(*row) for row in rows], (r[0] if r else 0)


def prompt_messages(key):
    """Trimmed (role, content) messages to send to the model: only completed rows,
    last PROMPT_TURNS turns (excludes the empty streaming reply we just created)."""
    with chat_lock:
        rows = _db().execute("SELECT role, content FROM messages "
                             "WHERE model=? AND status='done' ORDER BY id", (key,)).fetchall()
    return trim_turns([{'role': role, 'content': content} for role, content in rows])


def all_revs():
    revs = {k: 0 for k in MODELS}
    with chat_lock:
        rows = _db().execute('SELECT model, rev FROM revs').fetchall()
    revs.update({m: rv for m, rv in rows if m in MODELS})
    return revs


def all_history():
    """(history, revs) for every model -- history is key -> [messages]."""
    hist = {k: [] for k in MODELS}
    with chat_lock:
        conn = _db()
        rows  = conn.execute('SELECT model, role, content, status, n_tokens, gen_ms '
                             'FROM messages ORDER BY id').fetchall()
        rrows = conn.execute('SELECT model, rev FROM revs').fetchall()
    for row in rows:
        if row[0] in hist:
            hist[row[0]].append(_row_to_msg(*row[1:]))
    revs = {k: 0 for k in MODELS}
    revs.update({m: rv for m, rv in rrows if m in MODELS})
    return hist, revs


def trim_turns(messages, max_turns=PROMPT_TURNS):
    """Keep only the last `max_turns` turns of `messages` (a turn = one user
    message + its reply). `messages` ends with the latest user message, so this
    returns that message plus the previous max_turns-1 complete exchanges. With
    fewer than max_turns user messages, the whole list is returned unchanged."""
    users = 0
    start = 0
    for i in range(len(messages) - 1, -1, -1):
        if messages[i].get('role') == 'user':
            users += 1
            if users == max_turns:
                start = i
                break
    return messages[start:]


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
                   '--n-gpu-layers', gpu_layers_for(key), '--reasoning', 'off']
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
# Request worker -- the single consumer of the request queue. Because it is the
# only thread that loads models and runs generation, switching and chatting are
# serialized for free (exactly one model resident, no GEN_LOCK). It writes all
# state into the DB; the HTTP handlers only enqueue and read.
##############################################################################

# Flush a growing reply to the DB at most this often, so each token is not a write
# (clients poll about once a second, so finer would not even be seen).
STREAM_FLUSH_S = 0.6


def worker_loop():
    """Drain the request queue forever, one item at a time."""
    while not shutdown_started:
        req = next_pending()
        if req is None:
            work_ready.wait(timeout=1.0)
            work_ready.clear()
            continue
        try:
            if req['kind'] == 'switch':
                _worker_switch(req['model'])
            elif req['kind'] == 'chat':
                _worker_chat(req)
            finish_request(req['id'])
        except Exception as e:
            print(f'[serve] request {req["id"]} ({req["kind"]}) failed: {e}', flush=True)
            finish_request(req['id'], error=str(e))


def _resolve_backend(key):
    """Return a booted-or-bootable available backend for key, building it on demand
    if its weights have since landed. Raises if the model can't run here."""
    if not getattr(backends.get(key), 'available', False) and weights_present(key):
        bring_online(key)
    backend = backends.get(key)
    if backend is None or not getattr(backend, 'available', False):
        raise RuntimeError(f'{MODELS[key]["name"]} is not available on this server')
    return backend


def _worker_switch(key):
    if key not in MODELS:
        raise RuntimeError(f'unknown model: {key}')
    if get_state()['active_model'] == key and getattr(backends.get(key), 'available', False):
        # Already resident -- make sure it is actually booted, then no-op.
        backends[key].activate()
        return
    backend = _resolve_backend(key)
    set_loading(key)                  # every client now shows "Loading X..."
    try:
        backend.activate()            # stops the old model, boots this one
        set_active(key)               # clears loading
    except Exception:
        set_loading(None)
        raise


def _worker_chat(req):
    """A chat runs against whatever model is active right now (switches ahead of it
    in the queue have already taken effect). Record the user message, then stream
    the reply straight into a 'streaming' DB row so every client watches it grow."""
    key = get_state()['active_model']
    if not key:
        raise RuntimeError('no active model')
    backend = _resolve_backend(key)
    backend.activate()                # idempotent: ensure resident
    add_message(key, 'user', req['content'], status='done')
    msgs = prompt_messages(key)
    reply_id = add_message(key, 'assistant', '', status='streaming')
    log_event('model_request', '-', 'POST', '/enqueue', request_id=req.get('request_id'),
              model=key, request_body=req['content'], message_count=len(msgs),
              backend=backend.__class__.__name__)

    params = {'max_tokens': 1024, 'temperature': 0.7, 'top_p': 0.9}
    full = ''
    n_tokens = None
    first_t = last_t = None
    last_flush = 0.0
    error = None
    infer_enter()                     # pause background downloads while generating
    try:
        for chunk in backend.stream(msgs, params):
            u = chunk.get('usage') or {}
            if u.get('completion_tokens'):
                n_tokens = u['completion_tokens']
            delta = chunk.get('choices', [{}])[0].get('delta', {})
            piece = delta.get('content') or delta.get('reasoning') or ''
            if piece:
                now = time.time()
                if first_t is None:
                    first_t = now
                last_t = now
                full += piece
                if now - last_flush >= STREAM_FLUSH_S:
                    update_message(reply_id, key, full)
                    last_flush = now
    except Exception as e:
        error = str(e)
    finally:
        infer_exit()

    gen_ms = int((last_t - first_t) * 1000) if (first_t and last_t and last_t > first_t) else None
    if not full and error:
        full = f'<Mervin>Error: {error}</Mervin>'
    if n_tokens is None:
        n_tokens = max(1, len(full) // 4) if full else 0
    finish_message(reply_id, key, full, n_tokens, gen_ms)
    log_event('model_response', '-', 'POST', '/enqueue', request_id=req.get('request_id'),
              model=key, status=500 if error else 200, response_body=full, error=error)


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


def fresh_shutdown_timestamp(query, now=None, window_s=300):
    """Accept a forgiving UTC timestamp in the query string.

    Looks for timestamp-like text anywhere in the raw query, strips separators,
    and accepts YYYYMMDDHHMM or YYYYMMDDHHMMSS within +/- window_s.
    """
    now = now or datetime.now(timezone.utc)
    text = unquote_plus(query or '')
    digits = ''.join(re.findall(r'\d', text))
    for start in range(len(digits)):
        for width, fmt in ((12, '%Y%m%d%H%M'), (14, '%Y%m%d%H%M%S')):
            stamp = digits[start:start + width]
            if len(stamp) < width:
                continue
            try:
                ts = datetime.strptime(stamp, fmt).replace(tzinfo=timezone.utc)
            except ValueError:
                continue
            if abs((now - ts).total_seconds()) <= window_s:
                return True, ts
    return False, None


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
        path, _, query = self.path.partition('?')
        if path == '/health':
            # The whole arena snapshot, polled by every client ~1x/second.
            st = get_state()
            q = queue_view()
            busy = st['loading_model'] is not None or len(q) > 0
            self._json_response({
                'status': 'ok' if st['active_model'] else 'loading',
                'model': st['active_model'],
                'loading': st['loading_model'],
                'busy': busy,
                'queue': q,
                'available': available_map(),
                'states': states_map(),
                'hardware': HARDWARE,
                'revs': all_revs(),
            })
        elif path == '/history':
            self._handle_history(query)
        elif path == '/request':
            self._handle_request(query)
        elif path == '/shutdown':
            self._handle_shutdown('GET', query)
        elif self.path == '/':
            self.path = '/index.html'
            super().do_GET()
        else:
            super().do_GET()

    def do_POST(self):
        content_len = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(content_len)
        if self.path == '/enqueue':
            self._handle_enqueue(body)
        elif self.path == '/clear':
            self._handle_clear(body)
        elif self.path == '/shutdown':
            self._handle_shutdown('POST')
        else:
            self.send_error(404)

    def _handle_shutdown(self, method, query=''):
        ip = self.client_address[0]
        if ip not in ('127.0.0.1', '::1'):
            self._json_response({'error': 'Shutdown is only allowed from localhost'}, 403)
            return
        if method == 'GET':
            ok, ts = fresh_shutdown_timestamp(query)
            if not ok:
                self._json_response({'error': 'GET /shutdown requires a UTC timestamp within 5 minutes'}, 400)
                return
            request_body = f'timestamp {ts.isoformat()}'
        else:
            request_body = None
        log_request(ip, method, '/shutdown', request_body=request_body)
        self._json_response({'status': 'shutting_down'})
        begin_shutdown(self.server)

    def _handle_enqueue(self, body):
        """Queue a chat message or a model switch and return its id immediately. The
        worker processes the queue in order; the result shows up in the DB, which
        every client is polling."""
        ip = self.client_address[0]
        try:
            req = json.loads(body)
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return
        kind = req.get('kind')
        if kind == 'switch':
            model = req.get('model')
            if model not in MODELS:
                self._json_response({'error': f'Unknown model: {model}'}, 400)
                return
            rid = enqueue('switch', model=model, request_id=req.get('request_id'))
            log_request(ip, 'POST', '/enqueue', request_body=f'switch {model}')
            self._json_response({'status': 'queued', 'id': rid, 'kind': 'switch'})
        elif kind == 'chat':
            content = (req.get('content') or '').strip()
            if not content:
                self._json_response({'error': 'Empty message'}, 400)
                return
            rid = enqueue('chat', content=content, request_id=req.get('request_id'))
            log_event('http_request', ip, 'POST', '/enqueue',
                      request_id=req.get('request_id'), request_body=content)
            self._json_response({'status': 'queued', 'id': rid, 'kind': 'chat'})
        else:
            self._json_response({'error': f'Unknown request kind: {kind}'}, 400)

    def _handle_request(self, query):
        """Status of a queued request -- the CLI polls this to wait for its reply."""
        try:
            rid = int((parse_qs(query).get('id') or [''])[0])
        except ValueError:
            self._json_response({'error': 'bad id'}, 400)
            return
        status, error = request_status(rid)
        if status is None:
            self._json_response({'error': 'not found'}, 404)
            return
        self._json_response({'id': rid, 'status': status, 'error': error})

    def _handle_history(self, query):
        """GET /history -> every model's transcript + revisions.
        GET /history?model=<key> -> just that model's messages + revision."""
        params = parse_qs(query)
        model = (params.get('model') or [None])[0]
        if model is not None:
            if model not in MODELS:
                self._json_response({'error': f'Unknown model: {model}'}, 400)
                return
            msgs, rev = history_snapshot(model)
            self._json_response({'model': model, 'messages': msgs, 'rev': rev})
            return
        hist, revs = all_history()
        self._json_response({'history': hist, 'revs': revs})

    def _handle_clear(self, body):
        """Erase one model's shared transcript (clears it for everyone)."""
        ip = self.client_address[0]
        try:
            key = json.loads(body).get('model')
        except json.JSONDecodeError:
            self._json_response({'error': 'Invalid JSON'}, 400)
            return
        if key not in MODELS:
            self._json_response({'error': f'Unknown model: {key}'}, 400)
            return
        rev = history_clear(key)
        log_request(ip, 'POST', '/clear', request_body=f'clear {key}')
        self._json_response({'status': 'ok', 'model': key, 'rev': rev})

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
    if sys.platform == 'darwin':
        print('[serve] Apple Metal GPU (models offloaded to Metal when GPU layers > 0)')
    elif GPU_TOTAL_GB is not None:
        print(f'[serve] NVIDIA GPU: {GPU_TOTAL_GB:.1f} GB total, {GPU_FREE_GB:.1f} GB free '
              f'(models offloaded to GPU when they fit, else CPU)')
    elif GPU_LAYERS_ENV is not None:
        print(f'[serve] GPU layers forced via MERV_GPU_LAYERS={GPU_LAYERS_ENV}')
    else:
        print('[serve] no NVIDIA GPU detected (nvidia-smi); CPU unless the backend offloads itself')
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
            ngl = b._gpu_layers()
            where = 'CPU' if ngl == '0' else f'GPU ngl={ngl}'
            extra = f' <- port {b.port}, {where}'
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


def _api_post(base, path, payload, timeout=30):
    req = Request(base + path, data=json.dumps(payload).encode(),
                  headers={'Content-Type': 'application/json'}, method='POST')
    try:
        with urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read())
    except HTTPError as e:
        try:
            return e.code, json.loads(e.read())
        except Exception:
            return e.code, {}


def _api_clear(base, key):
    return _api_post(base, '/clear', {'model': key})


def _wait_request(base, rid, timeout=900):
    """Block until a queued request finishes; return (status, error). A switch may
    take minutes (cold model load), so poll patiently."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = _api_get(base, f'/request?id={rid}')
        except Exception:
            time.sleep(0.5)
            continue
        if r.get('status') in ('done', 'error', 'cancelled'):
            return r.get('status'), r.get('error')
        time.sleep(0.4)
    return 'timeout', 'request did not finish in time'


def _api_switch(base, key):
    """Enqueue a model switch and wait for it. Returns (ok, error)."""
    code, resp = _api_post(base, '/enqueue', {'kind': 'switch', 'model': key})
    if code != 200:
        return False, resp.get('error', f'HTTP {code}')
    status, error = _wait_request(base, resp['id'])
    return status == 'done', error


def _last_assistant(base, key):
    """The most recent assistant message dict for a model, or None."""
    try:
        msgs = _api_get(base, '/history?model=' + key).get('messages', [])
    except Exception:
        return None
    for m in reversed(msgs):
        if m.get('role') == 'assistant':
            return m
    return None


def _msg_elapsed(m):
    return (m.get('gen_ms') / 1000.0) if m.get('gen_ms') else None


def _send_chat(base, key, content):
    """Enqueue a chat, wait for it, and print the reply (read back from the DB)."""
    code, resp = _api_post(base, '/enqueue', {'kind': 'chat', 'content': content})
    if code != 200:
        print(f'  [{resp.get("error", "HTTP " + str(code))}]')
        return
    status, error = _wait_request(base, resp['id'])
    if status != 'done':
        print(f'  [chat {status}: {error or ""}]')
        return
    msg = _last_assistant(base, key)
    if msg:
        _print_reply(msg.get('content', ''), msg.get('n_tokens'), _msg_elapsed(msg))
    else:
        print('  [no reply]')


def _print_history(base, key):
    """Print the shared transcript for a model when entering / switching to it."""
    try:
        msgs = _api_get(base, '/history?model=' + key).get('messages', [])
    except Exception:
        return
    if not msgs:
        return
    print(f'\n  -- shared history: {MODELS[key]["name"]} ({len(msgs)} messages) --')
    for m in msgs:
        if m.get('role') == 'user':
            print(f'  you> {m.get("content", "")}')
        else:
            _print_reply(m.get('content', ''), m.get('n_tokens'), _msg_elapsed(m))


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
    print('  Mervin/Mervis CLI chat (shares each model\'s transcript with the web UI).')
    print('    /model            list models and their state')
    print('    /model <name>     switch (e.g. /model mistral)')
    print('    /clear            erase this model\'s shared history (for everyone)')
    print('    /help             show commands')
    print('    /quit             exit')
    print('=' * 60)
    _print_history(base, active)

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
                code, resp = _api_clear(base, active)
                if code == 200:
                    print(f'  erased {active} shared history')
                else:
                    print(f'  cannot clear: {resp.get("error", "HTTP " + str(code))}')
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
                        print(f'  switching to {MODELS[key]["name"]} (loading may take a while)...')
                        ok, err = _api_switch(base, key)
                        if ok:
                            active = key
                            print(f'  switched to {key} ({MODELS[key]["name"]})')
                            _print_history(base, active)
                        else:
                            print(f'  cannot switch: {err or "failed"}')
            else:
                print(f'  unknown command: {cmd}')
            continue

        # Make sure the arena is on our model first (a web user may have switched),
        # then send. Both go through the same queue/worker as the web UI.
        ok, err = _api_switch(base, active)
        if not ok:
            print(f'  ({MODELS[active]["name"]}: {err or "not ready"})')
            continue
        _send_chat(base, active, line)


def main():
    global PORT, LLAMA_SERVER

    argv = sys.argv[1:]
    print_command_line_help()
    args = parse_args(argv)
    if args['help']:
        return
    if args['port'] is not None:              # overrides MERV_PORT / default
        PORT = args['port']

    if not args['check']:
        # Make the bundled GPU-capable server present before we plan backends,
        # then drop any cached weights HuggingFace has changed so the normal
        # download path re-fetches them. --check stays side-effect-free.
        ensure_llama_server()
        LLAMA_SERVER = find_llama_server()
        refresh_stale_weights()

    build_backends()
    init_chat_db()             # open / create the shared transcript store

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

    first_model = ready[0]
    backends[first_model].activate()          # load ONLY this model (single slot)
    set_active(first_model)                    # record the resident model in the DB
    print(f'[serve] active model: {first_model}', flush=True)
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

    # The single worker that drains the request queue (chats + switches).
    threading.Thread(target=worker_loop, daemon=True).start()

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
