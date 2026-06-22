#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = [
#     "huggingface_hub",
#     "llama-cpp-python",
#     "mlx-lm; sys_platform == 'darwin' and platform_machine == 'arm64'",
# ]
# ///
"""
Unified Mervin/Mervis model-switching server -- one file for all three hosts.

Backend selection is automatic, decided at startup from what the host can do:

  phi / gemma (GGUF)
      * if a `llama-server` binary is present (e.g. Apple Silicon Mac with
        `brew install llama.cpp`) it is launched as a subprocess and proxied
        -- this gives Metal GPU offload. All such backends stay resident and
        switching between them is instant.
      * otherwise the model is run in-process with llama-cpp-python (Windows
        and Linux, CPU). Only one in-process model is resident at a time;
        switching unloads the old one and loads the new one.

  qwen (MLX, Apple-only)
      * if the `mlx_lm` package is importable AND mlx weights exist, it is
        launched via `mlx_lm.server` and proxied (Mac only).
      * everywhere else qwen is unavailable. Any request that targets it gets
        a friendly canned "can't run on this server" reply instead of an error,
        and the UI greys the column out.

Run it the way each host already does:
  Windows / Linux : uv run serve.py        (uv installs llama-cpp-python)
  Mac             : python3 serve.py        (uses brew llama-server + mlx_lm)

Environment overrides:
  MERV_HOST           bind address (default 0.0.0.0 on macOS, else 127.0.0.1)
  MERV_PORT           listen port  (default 52840)
  MERV_THREADS        CPU threads for the in-process backend (default 4)
  MERV_LLAMA_BACKEND  auto | server | inproc -- how phi/gemma run (default auto:
                      use the llama-server binary if present, else in-process)

Pass --check to print the detected backend plan and exit without loading models.
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
import importlib.util
import http.client
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.request import urlopen
from urllib.error import URLError
from datetime import datetime, timezone

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
    sys.stderr.reconfigure(encoding='utf-8')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PORT     = int(os.environ.get('MERV_PORT', '52840'))
HOST     = os.environ.get('MERV_HOST', '0.0.0.0' if sys.platform == 'darwin' else '127.0.0.1')
THREADS  = int(os.environ.get('MERV_THREADS', '4'))


##############################################################################
# Host capability detection
##############################################################################

def find_llama_server():
    """Locate a llama-server binary (Mac/brew gives GPU offload). None if absent."""
    found = shutil.which('llama-server')
    if found:
        return found
    for cand in ('/opt/homebrew/bin/llama-server',
                 '/usr/local/bin/llama-server',
                 '/usr/bin/llama-server'):
        if os.path.isfile(cand):
            return cand
    return None


def have_mlx():
    return importlib.util.find_spec('mlx_lm') is not None


LLAMA_SERVER  = find_llama_server()
MLX_OK        = have_mlx()
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
        'name': 'Qwen 3.5-4B',
        'kind': 'mlx',
        'port': 52842,
        'mlx': [
            os.path.join(BASE_DIR, 'qwen3.5-4b', 'mlx-4bit'),
            os.path.join(BASE_DIR, 'qwen3.5-4b', 'merged_model'),
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
    'gptoss': {
        'name': 'gpt-oss 20B',
        'kind': 'llama',
        'port': 52844,
        'gguf': [
            os.path.join(BASE_DIR, 'gpt-oss', 'model-mxfp4.gguf'),
        ],
    },
}


def first_gguf(cfg):
    for p in cfg.get('gguf', []):
        if os.path.isfile(p):
            return p
    return None


def first_mlx_dir(cfg):
    for p in cfg.get('mlx', []):
        if os.path.isdir(p) and any(
            f.endswith(('.safetensors', '.npz')) for f in os.listdir(p)
        ):
            return p
    return None


##############################################################################
# HuggingFace weight download (runs at startup when weights are absent)
##############################################################################

HF_WEIGHTS = {
    'phi': {
        'kind':     'file',
        'repo':     'freeideas/merv-phi4mini',
        'filename': 'model-q4_k_m.gguf',
        'local':    os.path.join(BASE_DIR, 'phi4mini', 'model-q4_k_m.gguf'),
    },
    'gemma': {
        'kind':     'file',
        'repo':     'freeideas/merv-gemma4e4b',
        'filename': 'model-q4_k_m.gguf',
        'local':    os.path.join(BASE_DIR, 'gemma4e4b', 'model-q4_k_m.gguf'),
    },
    'qwen': {
        'kind':  'dir',
        'repo':  'freeideas/merv-qwen3.5-4b-mlx',
        'local': os.path.join(BASE_DIR, 'qwen3.5-4b', 'mlx-4bit'),
    },
    'gptoss': {
        'kind':     'file',
        'repo':     'freeideas/merv-gpt-oss-20b',
        'filename': 'model-mxfp4.gguf',
        'local':    os.path.join(BASE_DIR, 'gpt-oss', 'model-mxfp4.gguf'),
    },
}


def download_weights():
    try:
        from huggingface_hub import hf_hub_download, snapshot_download
    except ImportError:
        print('[serve] huggingface_hub not installed -- skipping weight download', flush=True)
        return

    for key, cfg in HF_WEIGHTS.items():
        # Skip MLX-only weights on non-Mac platforms (they can't be used).
        if cfg['kind'] == 'dir' and not MLX_OK:
            continue

        if cfg['kind'] == 'file':
            if os.path.isfile(cfg['local']):
                continue
            print(f'[serve] {key}: weights missing -- downloading {cfg["filename"]} '
                  f'from {cfg["repo"]} ...', flush=True)
            os.makedirs(os.path.dirname(cfg['local']), exist_ok=True)
            try:
                hf_hub_download(
                    repo_id=cfg['repo'],
                    filename=cfg['filename'],
                    local_dir=os.path.dirname(cfg['local']),
                )
                print(f'[serve] {key}: download complete', flush=True)
            except Exception as e:
                print(f'[serve] {key}: download failed: {e}', flush=True)
        else:
            local = cfg['local']
            if os.path.isdir(local) and any(
                f.endswith(('.safetensors', '.npz')) for f in os.listdir(local)
            ):
                continue
            print(f'[serve] {key}: weights missing -- downloading MLX dir '
                  f'from {cfg["repo"]} ...', flush=True)
            os.makedirs(local, exist_ok=True)
            try:
                snapshot_download(repo_id=cfg['repo'], local_dir=local)
                print(f'[serve] {key}: download complete', flush=True)
            except Exception as e:
                print(f'[serve] {key}: download failed: {e}', flush=True)


##############################################################################
# Tag kludge -- the small models routinely misspell their own persona tags
##############################################################################

FALLBACK_RESPONSE = (
    '<Mervin>I am feeling too sad to respond right now.</Mervin>'
    '<Mervis>I am so joyful I can barely speak right now!</Mervis>'
)


def kludge_fix_tags(text):
    text = re.sub(r'<M{2,}ervin[^a-zA-Z0-9>]*>?', '<Mervin>', text)
    text = re.sub(r'<M{2,}ervis[^a-zA-Z0-9>]*>?', '<Mervis>', text)
    text = re.sub(r'<Mervin[^a-zA-Z0-9>]+>', '<Mervin>', text)
    text = re.sub(r'<Mervis[^a-zA-Z0-9>]+>', '<Mervis>', text)
    text = re.sub(r'<Mervin(?=[A-Z])', '<Mervin>', text)
    text = re.sub(r'<Mervis(?=[A-Z])', '<Mervis>', text)
    text = re.sub(r'</M+ervin[^a-zA-Z0-9>]*>', '</Mervin>', text)
    text = re.sub(r'</M+ervis[^a-zA-Z0-9>]*>', '</Mervis>', text)
    return text


def kludge_has_valid_tags(text):
    return bool(
        re.search(r'<Mervin>.*?</Mervin>', text, re.DOTALL)
        and re.search(r'<Mervis>.*?</Mervis>', text, re.DOTALL)
    )


def kludge_clean_messages(messages):
    cleaned = []
    for msg in messages:
        if msg.get('role') == 'assistant':
            content = kludge_fix_tags(msg.get('content') or '')
            if not kludge_has_valid_tags(content):
                content = FALLBACK_RESPONSE
            cleaned.append({**msg, 'content': content})
        else:
            cleaned.append(msg)
    return cleaned


def content_of(message):
    """Some backends (mlx/qwen) put text under 'reasoning' instead of 'content'."""
    return message.get('content') or message.get('reasoning') or ''


##############################################################################
# Backends
##############################################################################

class ProxyBackend:
    """Runs an OpenAI-compatible server as a subprocess and proxies to it.

    Used for llama-server (phi/gemma, GPU) and mlx_lm.server (qwen) on the Mac.
    These stay resident, so switching between them is instant and they need no
    request serialization (the child server handles its own concurrency).
    """
    persistent = True
    needs_lock = False

    def __init__(self, key, cmd, port, ready_kind):
        self.key        = key
        self.cmd        = cmd
        self.port       = port
        self.ready_kind = ready_kind     # 'llama' -> /health ; 'mlx' -> /v1/models
        self.proc       = None
        self.available  = False

    def boot(self):
        print(f'[serve] starting {self.key} backend on port {self.port} ...', flush=True)
        self.proc = subprocess.Popen(
            self.cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            self._wait_ready()
            self.available = True
            print(f'[serve] {self.key} ready on port {self.port}', flush=True)
        except TimeoutError:
            out = self.proc.stdout.read(4096).decode('utf-8', 'replace') if self.proc.stdout else ''
            print(f'[serve] {self.key} failed to start:\n{out}', flush=True)
            self.stop()
            self.available = False

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
        pass  # already running

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
            data = conn.getresponse().read()
            result = json.loads(data)
            for ch in result.get('choices', []):
                msg = ch.get('message')
                if isinstance(msg, dict) and 'content' not in msg and 'reasoning' in msg:
                    msg['content'] = msg['reasoning']
            return result
        finally:
            conn.close()

    def stream(self, messages, params):
        payload = {'messages': messages, 'stream': True, **params}
        conn = self._post(payload, stream=True)
        resp = conn.getresponse()
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
                        if isinstance(delta, dict) and 'content' not in delta and 'reasoning' in delta:
                            delta['content'] = delta['reasoning']
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
    _lock    = threading.Lock()

    def __init__(self, key, path):
        self.key       = key
        self.path      = path
        self.available = True

    def boot(self):
        pass  # loaded lazily on activate()

    def activate(self):
        with InProcBackend._lock:
            self._ensure_loaded()

    @classmethod
    def lock(cls):
        return cls._lock

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
    """Stand-in for a model that cannot run on this host (e.g. qwen off-Mac).

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
            f'<Mervin>Sorry -- {name} can only run on the Mac (it needs Apple '
            f'MLX hardware). This server cannot load it, so here I sulk.</Mervin>'
            f'<Mervis>No worries at all! Just pick Phi or Gemma and we will have '
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


# Resolved at startup: key -> backend instance (always present; unavailable
# models get a SpoofBackend so requests still get a sensible reply).
backends     = {}
active_model = None
active_lock  = threading.Lock()


def build_backends():
    for key, cfg in MODELS.items():
        if cfg['kind'] == 'llama':
            path = first_gguf(cfg)
            if path is None:
                backends[key] = SpoofBackend(key)
            elif LLAMA_BACKEND == 'server' or (LLAMA_BACKEND == 'auto' and LLAMA_SERVER):
                if not LLAMA_SERVER:
                    print('[serve] WARNING: MERV_LLAMA_BACKEND=server but no llama-server '
                          'binary found; falling back to in-process.', flush=True)
                    backends[key] = InProcBackend(key, path)
                else:
                    cmd = [LLAMA_SERVER, '--model', path, '--port', str(cfg['port']),
                           '--host', '127.0.0.1', '--ctx-size', '4096', '--n-gpu-layers', '99']
                    backends[key] = ProxyBackend(key, cmd, cfg['port'], 'llama')
            else:
                backends[key] = InProcBackend(key, path)
        elif cfg['kind'] == 'mlx':
            path = first_mlx_dir(cfg)
            if path and MLX_OK:
                cmd = [sys.executable, '-m', 'mlx_lm.server', '--model', path,
                       '--port', str(cfg['port']), '--host', '127.0.0.1']
                backends[key] = ProxyBackend(key, cmd, cfg['port'], 'mlx')
            else:
                backends[key] = SpoofBackend(key)


def available_map():
    return {k: bool(getattr(b, 'available', False)) for k, b in backends.items()}


##############################################################################
# Request / response log
##############################################################################

LOG_DIR = os.path.join(BASE_DIR, 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
_log_lock = threading.Lock()


def log_request(ip, method, path, request_body=None, response_body=None):
    now = datetime.now(timezone.utc)
    filename = now.strftime('%Y-%m-%d-%HZ.log')
    entry = {
        'ts': now.strftime('%Y-%m-%dT%H:%M:%S.%f')[:-3] + 'Z',
        'ip': ip,
        'method': method,
        'path': path,
    }
    if request_body is not None:
        entry['request'] = request_body
    if response_body is not None:
        entry['response'] = response_body
    line = json.dumps(entry, ensure_ascii=False) + '\n'
    with _log_lock:
        with open(os.path.join(LOG_DIR, filename), 'a', encoding='utf-8') as f:
            f.write(line)


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
        else:
            self.send_error(404)

    def _handle_switch(self, body):
        global active_model
        ip = self.client_address[0]
        try:
            data = json.loads(body)
            key = data.get('model')
            if key not in MODELS:
                self._json_response({'error': f'Unknown model: {key}'}, 400)
                return
            backend = backends[key]
            if not getattr(backend, 'available', False):
                self._json_response(
                    {'error': f'{MODELS[key]["name"]} is not available on this server'}, 503)
                return
            log_request(ip, 'POST', '/switch', request_body=f'switch to {key}')
            backend.activate()        # instant for proxy, loads model for in-process
            with active_lock:
                active_model = key
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

        backend = backends.get(key)
        if backend is None or not getattr(backend, 'available', False):
            backend = SpoofBackend(key)   # graceful "can't run here"

        messages = kludge_clean_messages(req.get('messages', []))
        params = {
            'max_tokens':  req.get('max_tokens', 256),
            'temperature': req.get('temperature', 0.7),
            'top_p':       req.get('top_p', 0.9),
        }
        stream = req.get('stream', False)

        user_msgs = [m['content'] for m in messages if m.get('role') == 'user']
        log_user_msg = user_msgs[-1] if user_msgs else None

        # In-process backends share a single resident model and a single llama
        # instance, so serialize them. Proxy backends manage their own concurrency.
        lock = InProcBackend.lock() if getattr(backend, 'needs_lock', False) else None
        if lock is not None and not lock.acquire(timeout=300):
            self._json_response({'error': 'Server busy, try again'}, 503)
            return
        try:
            if stream:
                self._stream_response(backend, messages, params, ip, log_user_msg)
            else:
                result = backend.complete(messages, params)
                text = content_of(result.get('choices', [{}])[0].get('message', {}))
                log_request(ip, 'POST', '/v1/chat/completions',
                            request_body=log_user_msg, response_body=text)
                self._json_response(result)
        except Exception as e:
            log_request(ip, 'POST', '/v1/chat/completions',
                        request_body=log_user_msg, response_body=f'ERROR: {e}')
            try:
                self._json_response({'error': str(e)}, 500)
            except Exception:
                pass
        finally:
            if lock is not None:
                lock.release()

    def _stream_response(self, backend, messages, params, ip, log_user_msg):
        self.send_response(200)
        self._cors_headers()
        self.send_header('Content-Type', 'text/event-stream')
        self.send_header('Cache-Control', 'no-cache')
        self.send_header('Connection', 'close')
        self.end_headers()

        full = ''
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
            full += f' [ERROR: {e}]'

        log_request(ip, 'POST', '/v1/chat/completions',
                    request_body=log_user_msg, response_body=full)

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
    print(f'[serve] mlx_lm available:    {MLX_OK}')
    for key, b in backends.items():
        kind = type(b).__name__
        extra = ''
        if isinstance(b, InProcBackend):
            extra = f' <- {b.path}'
        elif isinstance(b, ProxyBackend):
            extra = f' <- port {b.port}: {" ".join(b.cmd[:3])}...'
        print(f'[serve]   {key:6s} {kind:14s}{extra}')


def main():
    global active_model

    download_weights()
    build_backends()

    if '--check' in sys.argv:
        describe_plan()
        print('[serve] --check: not starting any backends.')
        return

    describe_plan()

    # Boot persistent (proxy) backends, in parallel, so we don't wait serially.
    threads = []
    for b in backends.values():
        if getattr(b, 'persistent', False) and not isinstance(b, SpoofBackend):
            t = threading.Thread(target=b.boot, daemon=True)
            t.start()
            threads.append(t)
    for t in threads:
        t.join()

    # Choose the first genuinely-available model and make it active.
    ready = [k for k, b in backends.items() if getattr(b, 'available', False)]
    if not ready:
        print('[serve] ERROR: no models are available on this host.', flush=True)
        for key, cfg in MODELS.items():
            hint = cfg.get('gguf', cfg.get('mlx', ['?']))[0]
            print(f'  {key}: expected weights near {hint}', flush=True)
        sys.exit(1)

    first = ready[0]
    backends[first].activate()        # in-process: loads it now; proxy: no-op
    active_model = first

    print(f'[serve] active model: {active_model}', flush=True)
    print(f'[serve] available:    {ready}', flush=True)

    try:
        server = ThreadedHTTPServer((HOST, PORT), ProxyHandler)
    except OSError as e:
        print(f'[serve] ERROR: cannot bind to {HOST}:{PORT} -- {e}', flush=True)
        print(f'[serve] Try a different port: MERV_PORT=53840 (or any free port)', flush=True)
        sys.exit(1)
    print(f'[serve] listening on http://{HOST}:{PORT}', flush=True)

    def cleanup(*_):
        for b in backends.values():
            try:
                b.stop()
            except Exception:
                pass
        os._exit(0)

    signal.signal(signal.SIGINT, cleanup)
    signal.signal(signal.SIGTERM, cleanup)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        cleanup()


if __name__ == '__main__':
    main()
