#!/usr/bin/env -S uv run
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "gguf",
#   "numpy",
#   "safetensors",
#   "mlx-lm",
# ]
# ///
"""
Convert fine-tuned GGUF files (Q8_0) -> HF safetensors -> MLX 4-bit.

Usage:
  uv run gguf_to_mlx.py phi     # convert phi4mini
  uv run gguf_to_mlx.py gemma   # convert gemma4e4b
  uv run gguf_to_mlx.py all     # both
"""

import sys, os, json, re, shutil
import numpy as np
from pathlib import Path

BASE = Path(__file__).parent

# ── Q8_0 dequantisation ────────────────────────────────────────────────────

def dequantize_q8_0(tensor) -> np.ndarray:
    """
    GGUF Q8_0: blocks of 32 int8 values each with a float16 scale.
    GGUF dimension order is REVERSED vs numpy (last GGUF dim = first numpy dim).
    Returns float16 numpy array in HF (numpy) shape.
    """
    gguf_shape = [int(s) for s in tensor.shape]
    np_shape   = list(reversed(gguf_shape))      # HF / numpy ordering

    inner_dim        = np_shape[-1]              # quantised dimension
    outer_elements   = int(np.prod(np_shape[:-1]))
    n_blocks         = inner_dim // 32           # Q8_0 block size = 32
    assert inner_dim % 32 == 0, f"inner_dim {inner_dim} not divisible by 32"

    raw = np.asarray(tensor.data).reshape(outer_elements, n_blocks * 34)

    blocks    = raw.reshape(outer_elements * n_blocks, 34)
    scale_raw = blocks[:, :2].copy()             # float16 bytes
    qs        = blocks[:, 2:].view(np.int8)      # int8 quant values

    scale = np.frombuffer(scale_raw.tobytes(), dtype=np.float16).reshape(-1, 1)
    out   = (scale.astype(np.float32) * qs.astype(np.float32)).astype(np.float16)
    return out.reshape(np_shape)


def load_f32(tensor) -> np.ndarray:
    """Return an F32 GGUF tensor as float32 numpy array (already in numpy shape)."""
    gguf_shape = [int(s) for s in tensor.shape]
    np_shape   = list(reversed(gguf_shape))
    return np.asarray(tensor.data, dtype=np.float32).reshape(np_shape)


def read_tensor(tensor) -> np.ndarray:
    if tensor.tensor_type == 0:   # F32
        return load_f32(tensor)
    elif tensor.tensor_type == 8: # Q8_0
        return dequantize_q8_0(tensor)
    else:
        raise ValueError(f"Unsupported tensor type {tensor.tensor_type} for {tensor.name}")


# ── GGUF field helpers ─────────────────────────────────────────────────────

def field_scalar(fields, key, default=None):
    try:
        val = fields[key].parts[-1].tolist()
        return val[0] if isinstance(val, list) else val
    except (KeyError, IndexError, AttributeError):
        return default


def field_str(fields, key, default=None):
    try:
        from gguf import GGUFValueType
        f = fields[key]
        if f.types[0] == GGUFValueType.STRING:
            return bytes(f.parts[-1]).decode('utf-8')
    except (KeyError, IndexError, AttributeError):
        pass
    return default


# ── phi3 / phi4-mini ──────────────────────────────────────────────────────

PHI3_GGUF_TO_HF = {
    'token_embd.weight':      'model.embed_tokens.weight',
    'output_norm.weight':     'model.norm.weight',
}
def phi3_layer_map(gguf_name):
    m = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
    if not m:
        return None
    n, rest = m.group(1), m.group(2)
    mapping = {
        'attn_norm.weight':   f'model.layers.{n}.input_layernorm.weight',
        'ffn_norm.weight':    f'model.layers.{n}.post_attention_layernorm.weight',
        'attn_qkv.weight':    f'model.layers.{n}.self_attn.qkv_proj.weight',
        'attn_output.weight': f'model.layers.{n}.self_attn.o_proj.weight',
        'ffn_up.weight':      f'model.layers.{n}.mlp.gate_up_proj.weight',
        'ffn_down.weight':    f'model.layers.{n}.mlp.down_proj.weight',
    }
    return mapping.get(rest)


def convert_phi(gguf_path: Path, hf_out: Path, mlx_out: Path):
    from gguf import GGUFReader
    print(f'[phi] Reading {gguf_path}')
    r = GGUFReader(str(gguf_path))
    fields = r.fields

    # ── extract rope factors (go into config, not weights) ──
    rope_long  = None
    rope_short = None
    weights    = {}

    for t in r.tensors:
        arr = read_tensor(t)
        if t.name == 'rope_factors_long.weight':
            rope_long = arr.astype(np.float32).tolist()
            continue
        if t.name == 'rope_factors_short.weight':
            rope_short = arr.astype(np.float32).tolist()
            continue
        hf_name = PHI3_GGUF_TO_HF.get(t.name) or phi3_layer_map(t.name)
        if hf_name is None:
            print(f'  [phi] skipping unknown tensor: {t.name}')
            continue
        weights[hf_name] = arr

    # ── build config.json ──
    hidden       = int(field_scalar(fields, 'phi3.embedding_length', 3072))
    n_layers     = int(field_scalar(fields, 'phi3.block_count', 32))
    n_heads      = int(field_scalar(fields, 'phi3.attention.head_count', 24))
    n_kv_heads   = int(field_scalar(fields, 'phi3.attention.head_count_kv', 8))
    ffn_len      = int(field_scalar(fields, 'phi3.feed_forward_length', 8192))
    rms_eps      = float(field_scalar(fields, 'phi3.attention.layer_norm_rms_epsilon', 1e-5))
    rope_base    = float(field_scalar(fields, 'phi3.rope.freq_base', 10000.0))
    ctx_len      = int(field_scalar(fields, 'phi3.context_length', 131072))
    orig_ctx     = int(field_scalar(fields, 'phi3.rope.scaling.original_context_length', 4096))
    attn_factor  = float(field_scalar(fields, 'phi3.rope.scaling.attn_factor', 1.0))
    rope_dim     = int(field_scalar(fields, 'phi3.rope.dimension_count', hidden // n_heads))
    head_dim     = hidden // n_heads
    partial_rotary = rope_dim / head_dim   # e.g. 96/128 = 0.75

    # vocab size from embedding tensor shape
    emb_weight   = weights.get('model.embed_tokens.weight')
    vocab_size   = emb_weight.shape[0] if emb_weight is not None else 200064

    rope_scaling = {
        'type':           'longrope',
        'long_mscale':    attn_factor,
        'short_mscale':   1.0,
    }
    if rope_long  is not None: rope_scaling['long_factor']  = rope_long
    if rope_short is not None: rope_scaling['short_factor'] = rope_short

    config = {
        '_name_or_path':                  'phi4mini-ft',
        'architectures':                  ['Phi3ForCausalLM'],
        'model_type':                     'phi3',
        'hidden_size':                    hidden,
        'intermediate_size':              ffn_len,
        'num_hidden_layers':              n_layers,
        'num_attention_heads':            n_heads,
        'num_key_value_heads':            n_kv_heads,
        'max_position_embeddings':        ctx_len,
        'original_max_position_embeddings': orig_ctx,
        'rms_norm_eps':                   rms_eps,
        'rope_theta':                     rope_base,
        'rope_scaling':                   rope_scaling,
        'partial_rotary_factor':          partial_rotary,
        'vocab_size':                     vocab_size,
        'tie_word_embeddings':            True,
        'torch_dtype':                    'float16',
    }

    _save(weights, config, hf_out, gguf_path.parent, mlx_out, 'phi')


# ── gemma4 ────────────────────────────────────────────────────────────────

def gemma4_layer_map(gguf_name):
    m = re.match(r'blk\.(\d+)\.(.+)', gguf_name)
    if not m:
        return None
    n, rest = m.group(1), m.group(2)
    mapping = {
        'attn_norm.weight':         f'model.layers.{n}.input_layernorm.weight',
        'ffn_norm.weight':          f'model.layers.{n}.pre_feedforward_layernorm.weight',
        'post_attention_norm.weight':  f'model.layers.{n}.post_attention_layernorm.weight',
        'post_ffw_norm.weight':     f'model.layers.{n}.post_feedforward_layernorm.weight',
        'attn_k_norm.weight':       f'model.layers.{n}.self_attn.k_norm.weight',
        'attn_q_norm.weight':       f'model.layers.{n}.self_attn.q_norm.weight',
        'attn_k.weight':            f'model.layers.{n}.self_attn.k_proj.weight',
        'attn_q.weight':            f'model.layers.{n}.self_attn.q_proj.weight',
        'attn_v.weight':            f'model.layers.{n}.self_attn.v_proj.weight',
        'attn_output.weight':       f'model.layers.{n}.self_attn.o_proj.weight',
        'ffn_gate.weight':          f'model.layers.{n}.mlp.gate_proj.weight',
        'ffn_up.weight':            f'model.layers.{n}.mlp.up_proj.weight',
        'ffn_down.weight':          f'model.layers.{n}.mlp.down_proj.weight',
        'inp_gate.weight':          f'model.layers.{n}.per_layer_input_gate.weight',
        'proj.weight':              f'model.layers.{n}.per_layer_projection.weight',
        'post_norm.weight':         f'model.layers.{n}.post_per_layer_input_norm.weight',
        'layer_output_scale.weight': f'model.layers.{n}.layer_scalar',
    }
    return mapping.get(rest)

GEMMA4_GGUF_TO_HF = {
    'token_embd.weight':         'model.embed_tokens.weight',
    'output_norm.weight':        'model.norm.weight',
    'per_layer_token_embd.weight': 'model.embed_tokens_per_layer.weight',
    'per_layer_model_proj.weight': 'model.per_layer_model_projection.weight',
    'per_layer_proj_norm.weight':  'model.per_layer_projection_norm.weight',
    'rope_freqs.weight':           None,
}


def convert_gemma(gguf_path: Path, hf_out: Path, mlx_out: Path):
    from gguf import GGUFReader, GGUFValueType
    print(f'[gemma] Reading {gguf_path}')
    r = GGUFReader(str(gguf_path))
    fields = r.fields

    weights = {}
    for t in r.tensors:
        arr = read_tensor(t)
        hf_name = GEMMA4_GGUF_TO_HF.get(t.name, gemma4_layer_map(t.name))
        if hf_name is None:
            print(f'  [gemma] skipping unknown tensor: {t.name}')
            continue
        weights[hf_name] = arr

    # ── build config.json from GGUF metadata ──
    def _i(key, default): return int(field_scalar(fields, key, default))
    def _f(key, default): return float(field_scalar(fields, key, default))

    hidden      = _i('gemma4.embedding_length', 2560)
    n_layers    = _i('gemma4.block_count', 34)
    n_heads     = _i('gemma4.attention.head_count', 16)
    n_kv_heads  = _i('gemma4.attention.head_count_kv', 8)
    ffn_len     = _i('gemma4.feed_forward_length', 10240)
    rms_eps     = _f('gemma4.attention.layer_norm_rms_epsilon', 1e-6)
    rope_base   = _f('gemma4.rope.freq_base', 10000.0)
    ctx_len     = _i('gemma4.context_length', 131072)
    q_norm_dims = [
        arr.shape[0]
        for name, arr in weights.items()
        if name.endswith('.self_attn.q_norm.weight')
    ]
    head_dim        = min(q_norm_dims) if q_norm_dims else 256
    global_head_dim = max(q_norm_dims) if q_norm_dims else 512
    layer_types = []
    for i in range(n_layers):
        q_norm = weights.get(f'model.layers.{i}.self_attn.q_norm.weight')
        layer_types.append(
            'full_attention'
            if q_norm is not None and q_norm.shape[0] == global_head_dim
            else 'sliding_attention'
        )

    kv_layers = [
        int(name.split('.')[2])
        for name in weights
        if name.endswith('.self_attn.k_proj.weight')
    ]
    num_kv_shared_layers = n_layers - (max(kv_layers) + 1) if kv_layers else 0

    hidden_per_layer_input = 0
    for name, arr in weights.items():
        if name.endswith('.per_layer_input_gate.weight'):
            hidden_per_layer_input = arr.shape[0]
            break

    sliding_win = _i('gemma4.attention.sliding_window', 1024)

    emb = weights.get('model.embed_tokens.weight')
    vocab_size = emb.shape[0] if emb is not None else 262144
    per_layer_embed = weights.get('model.embed_tokens_per_layer.weight')
    vocab_size_per_layer_input = (
        per_layer_embed.shape[0] if per_layer_embed is not None else vocab_size
    )

    config = {
        '_name_or_path':          'gemma4e4b-ft',
        'architectures':          ['Gemma4ForConditionalGeneration'],
        'model_type':             'gemma4_text',
        'hidden_size':            hidden,
        'intermediate_size':      ffn_len,
        'num_hidden_layers':      n_layers,
        'num_attention_heads':    n_heads,
        'num_key_value_heads':    n_kv_heads,
        'head_dim':               head_dim,
        'global_head_dim':        global_head_dim,
        'num_global_key_value_heads': n_kv_heads,
        'num_kv_shared_layers':   num_kv_shared_layers,
        'hidden_size_per_layer_input': hidden_per_layer_input,
        'vocab_size_per_layer_input': vocab_size_per_layer_input,
        'max_position_embeddings': ctx_len,
        'rms_norm_eps':           rms_eps,
        'rope_parameters': {
            'full_attention': {
                'partial_rotary_factor': 0.25,
                'rope_theta': rope_base,
                'rope_type': 'proportional',
            },
            'sliding_attention': {
                'partial_rotary_factor': 1.0,
                'rope_theta': 10000.0,
                'rope_type': 'default',
            },
        },
        'vocab_size':             vocab_size,
        'sliding_window':         sliding_win,
        'sliding_window_pattern':  6,
        'layer_types':            layer_types,
        'use_double_wide_mlp':     False,
        'tie_word_embeddings':    True,
        'torch_dtype':            'bfloat16',
    }

    _save(weights, config, hf_out, gguf_path.parent, mlx_out, 'gemma')


# ── common save + mlx convert ─────────────────────────────────────────────

def _save(weights: dict, config: dict, hf_out: Path, src_dir: Path,
          mlx_out: Path, tag: str):
    from safetensors.numpy import save_file

    hf_out.mkdir(parents=True, exist_ok=True)
    print(f'[{tag}] Writing {len(weights)} tensors to {hf_out}')
    save_file({k: v for k, v in weights.items()}, str(hf_out / 'model.safetensors'))

    (hf_out / 'config.json').write_text(json.dumps(config, indent=2))

    # copy tokenizer files
    for name in ('tokenizer.json', 'tokenizer_config.json',
                 'special_tokens_map.json', 'tokenizer.model'):
        src = src_dir / name
        if src.exists():
            shutil.copy(src, hf_out / name)

    print(f'[{tag}] Converting to MLX 4-bit -> {mlx_out}')
    if mlx_out.exists():
        shutil.rmtree(mlx_out)

    import mlx.core as mx
    mx.set_default_device(mx.cpu)
    from mlx_lm.convert import convert
    convert(
        hf_path=str(hf_out),
        mlx_path=str(mlx_out),
        quantize=True,
    )
    print(f'[{tag}] Done -> {mlx_out}')


# ── entry point ───────────────────────────────────────────────────────────

def main():
    target = sys.argv[1] if len(sys.argv) > 1 else 'all'

    if target in ('phi', 'all'):
        convert_phi(
            gguf_path = BASE / 'phi4mini' / 'model-q8_0.gguf',
            hf_out    = BASE / 'phi4mini' / 'hf-ft',
            mlx_out   = BASE / 'phi4mini' / 'mlx-4bit',
        )

    if target in ('gemma', 'all'):
        convert_gemma(
            gguf_path = BASE / 'gemma4e4b' / 'model-q8_0.gguf',
            hf_out    = BASE / 'gemma4e4b' / 'hf-ft',
            mlx_out   = BASE / 'gemma4e4b' / 'mlx-4bit',
        )


if __name__ == '__main__':
    main()
