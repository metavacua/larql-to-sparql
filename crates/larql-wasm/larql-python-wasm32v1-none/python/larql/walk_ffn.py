"""
MLX generation with vindex walk FFN.

MLX handles attention (KV cache, sampling, generation loop).
Vindex handles FFN (Rust sparse computation via WalkModel).

Only attention + norms + embeddings in MLX memory.
FFN runs in Rust — gate KNN from vindex, sparse up/down from model weights.

Usage:
    from larql.walk_ffn import load
    import mlx_lm

    model, tokenizer = load("model.vindex")
    response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=20)
"""

import json
from pathlib import Path
from typing import Tuple


def load(vindex_path: str, top_k: int = 8192) -> Tuple:
    """Load MLX model with vindex walk FFN.

    Attention + norms + embeddings loaded into MLX.
    FFN replaced with Rust walk FFN (vindex gate KNN + sparse weights).

    Args:
        vindex_path: path to .vindex directory (requires --level all)
        top_k: features per FFN layer (8192 = lossless)

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.utils as mlx_utils
    import larql
    from larql.mlx import _build_config, _weight_prefix

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)

    # Load WalkModel in Rust (holds weights + vindex for per-layer FFN)
    walk_model = larql.WalkModel(str(vindex_path), top_k=top_k)

    # Build MLX model architecture
    mlx_config = _build_config(str(vindex_path))
    model_class, model_args_class = mlx_utils._get_classes(config=mlx_config)
    model = model_class(model_args_class.from_dict(mlx_config))

    # Load only attention + norms + embeddings (skip FFN up/down)
    dtype_str = config.get("dtype", "f32")
    mv_fmt = "e" if dtype_str == "f16" else "f"
    hidden = config["hidden_size"]
    prefix, lm_prefix = _weight_prefix(config)

    import mmap as mmap_mod
    open_files = {}
    mmaps = {}

    def get_mmap(fname):
        if fname not in mmaps:
            fpath = vpath / fname
            if fpath.exists():
                fh = open(fpath, "rb")
                mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
                open_files[fname] = fh
                mmaps[fname] = mm
        return mmaps.get(fname)

    with open(vpath / "weight_manifest.json") as f:
        manifest = json.load(f)

    weights = {}
    skipped = 0

    for entry in manifest:
        key = entry["key"]
        if "vision_tower" in key or "multi_modal" in key:
            continue
        # Skip FFN up/down — walk FFN handles these in Rust
        if "mlp.up_proj" in key or "mlp.down_proj" in key:
            skipped += entry["length"]
            continue

        mm = get_mmap(entry["file"])
        if mm is None:
            continue

        shape = tuple(entry["shape"])
        buf = mm[entry["offset"]:entry["offset"] + entry["length"]]
        mv = memoryview(buf).cast(mv_fmt, shape)

        if key == "lm_head.weight":
            name = f"{lm_prefix}lm_head.weight"
        elif key == "norm.weight":
            name = f"{prefix}norm.weight"
        else:
            name = f"{prefix}{key}"
        weights[name] = mx.array(mv)

    # Embeddings
    vocab = config["vocab_size"]
    bpf = 2 if dtype_str == "f16" else 4
    fh_e = open(vpath / "embeddings.bin", "rb")
    mm_e = mmap_mod.mmap(fh_e.fileno(), 0, access=mmap_mod.ACCESS_READ)
    buf = mm_e[:vocab * hidden * bpf]
    weights[f"{prefix}embed_tokens.weight"] = mx.array(memoryview(buf).cast(mv_fmt, (vocab, hidden)))

    # Gate vectors
    for info in config.get("layers", []):
        gate_mm = get_mmap("gate_vectors.bin")
        buf = gate_mm[info["offset"]:info["offset"] + info["length"]]
        mv = memoryview(buf).cast(mv_fmt, (info["num_features"], hidden))
        weights[f"{prefix}layers.{info['layer']}.mlp.gate_proj.weight"] = mx.array(mv)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())

    # Clean up mmaps
    for mm in mmaps.values():
        mm.close()
    for fh in open_files.values():
        fh.close()
    mm_e.close()
    fh_e.close()

    # Patch each layer's MLP with walk FFN
    _patch_mlp(model, walk_model, config)

    tokenizer = mlx_utils.load_tokenizer(vpath)
    print(f"Walk FFN: {skipped / 1e9:.1f} GB FFN weights handled by Rust (not in MLX memory)")

    return model, tokenizer


def _patch_mlp(model, walk_model, config):
    """Replace each layer's MLP with a walk FFN module."""
    import mlx.core as mx
    import mlx.nn as nn

    # Find the layers list
    if hasattr(model, 'language_model'):
        layers = model.language_model.model.layers
    elif hasattr(model, 'model'):
        layers = model.model.layers
    else:
        layers = model.layers

    class WalkMLP(nn.Module):
        """MLP that delegates to Rust walk FFN. No numpy."""
        def __init__(self, layer_idx):
            super().__init__()
            self._layer_idx = layer_idx

        def __call__(self, x):
            shape = x.shape
            hidden = shape[-1]

            if len(shape) == 3:
                x_2d = x.reshape(-1, hidden)
            else:
                x_2d = x
            seq_len = x_2d.shape[0]

            # MLX → f32 bytes → Rust (zero-copy) → bytes → MLX
            x_f32 = x_2d.astype(mx.float32)
            mx.eval(x_f32)
            x_bytes = bytes(x_f32)

            # Rust: gate KNN + sparse FFN — all computation in Rust
            out_bytes = walk_model.ffn_layer(
                layer=self._layer_idx, x_bytes=x_bytes, seq_len=seq_len
            )

            # bytes → MLX array
            out = mx.array(memoryview(out_bytes).cast('f', (seq_len, hidden)))
            out = out.astype(x.dtype)
            if len(shape) == 3:
                out = out.reshape(shape)
            return out

    for i in range(len(layers)):
        layers[i].mlp = WalkMLP(i)
