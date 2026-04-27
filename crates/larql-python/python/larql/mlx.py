"""
Load MLX models directly from a vindex.

No safetensors. No intermediate files. The vindex binary files are mmap'd
and wrapped as mx.array views — zero-copy on Apple Silicon unified memory.

Usage:
    import larql
    model, tokenizer = larql.mlx.load("gemma3-4b.vindex")
"""

import json
from pathlib import Path
from typing import Optional, Tuple



def _weight_prefix(config: dict) -> Tuple[str, str]:
    """Determine the weight name prefix MLX expects for this architecture."""
    family = config.get("family", "")
    model_type = config.get("model_config", {}).get("model_type", family)
    if "gemma3" in model_type:
        return "language_model.model.", "language_model."
    return "model.", ""


def _load_weights(vindex_path: str) -> dict:
    """Mmap vindex binaries and produce {tensor_name: mx.array} dict.

    Uses memoryview.cast() to create mx.array directly from mmap'd
    memory — bypasses numpy entirely for f16 vindexes.
    """
    import mmap as mmap_mod
    import mlx.core as mx

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)
    with open(vpath / "weight_manifest.json") as f:
        manifest = json.load(f)

    dtype_str = config.get("dtype", "f32")
    hidden = config["hidden_size"]
    prefix, lm_prefix = _weight_prefix(config)

    # memoryview format codes: 'e' = f16, 'f' = f32
    mv_fmt = "e" if dtype_str == "f16" else "f"

    # Mmap each binary file once
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

    def mmap_to_mx(mm, offset, length, shape):
        """Create mx.array from mmap slice via memoryview. No numpy."""
        buf = mm[offset:offset + length]
        mv = memoryview(buf).cast(mv_fmt, shape)
        return mx.array(mv)

    weights = {}

    try:
        # Manifest weights (attention, FFN up/down, norms)
        for entry in manifest:
            raw_name = entry["key"]
            if "vision_tower" in raw_name or "multi_modal" in raw_name:
                continue

            mm = get_mmap(entry["file"])
            if mm is None:
                continue

            shape = tuple(entry["shape"])
            offset = entry["offset"]
            length = entry["length"]

            if raw_name == "lm_head.weight":
                name = f"{lm_prefix}lm_head.weight"
            elif raw_name == "norm.weight":
                name = f"{prefix}norm.weight"
            else:
                name = f"{prefix}{raw_name}"

            weights[name] = mmap_to_mx(mm, offset, length, shape)

        # Embeddings
        embed_path = vpath / "embeddings.bin"
        if embed_path.exists():
            vocab = config.get("vocab_size", 0)
            if vocab > 0:
                fh = open(embed_path, "rb")
                mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
                open_files["embeddings.bin"] = fh
                weights[f"{prefix}embed_tokens.weight"] = mmap_to_mx(
                    mm, 0, vocab * hidden * (2 if dtype_str == "f16" else 4),
                    (vocab, hidden)
                )

        # FFN gate (gate_vectors.bin = mlp.gate_proj.weight)
        gate_path = vpath / "gate_vectors.bin"
        if gate_path.exists():
            fh = open(gate_path, "rb")
            gate_mm = mmap_mod.mmap(fh.fileno(), 0, access=mmap_mod.ACCESS_READ)
            open_files["gate_vectors.bin"] = fh
            for info in config.get("layers", []):
                layer = info["layer"]
                nf = info["num_features"]
                off = info["offset"]
                length = info["length"]
                weights[f"{prefix}layers.{layer}.mlp.gate_proj.weight"] = mmap_to_mx(
                    gate_mm, off, length, (nf, hidden)
                )

    finally:
        # Close mmaps and files after mx.array has copied the data
        for mm in mmaps.values():
            mm.close()
        for fh in open_files.values():
            fh.close()

    return weights


def _build_config(vindex_path: str) -> dict:
    """Build MLX config. Uses HF cache if available, else vindex metadata."""
    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        vc = json.load(f)

    # Try HF cache first (has all fields MLX needs)
    try:
        from mlx_lm.utils import load_config, hf_repo_to_path
        hf_path = hf_repo_to_path(vc.get("model", ""))
        if hf_path:
            return load_config(hf_path)
    except Exception:
        pass

    # Build from vindex
    mc = vc.get("model_config", {})
    tc = {
        "model_type": mc.get("model_type", vc.get("family", "")),
        "hidden_size": vc["hidden_size"],
        "intermediate_size": vc["intermediate_size"],
        "num_hidden_layers": vc["num_layers"],
        "vocab_size": vc["vocab_size"],
        "head_dim": mc.get("head_dim", 256),
        "num_attention_heads": mc.get("num_q_heads", 8),
        "num_key_value_heads": mc.get("num_kv_heads", 4),
        "rope_theta": mc.get("rope_base", 1000000.0),
        "rms_norm_eps": 1e-6,
    }

    if "gemma3" in vc.get("family", ""):
        return {"model_type": "gemma3", "text_config": tc,
                "sliding_window": mc.get("sliding_window", 1024)}
    return tc


def load(vindex_path: str, lazy: bool = False) -> Tuple:
    """Load an MLX model directly from a vindex.

    No safetensors. Weights are mmap'd from vindex binaries and
    wrapped as mx.array — zero-copy on unified memory.

    Args:
        vindex_path: Path to .vindex directory
        lazy: If True, don't eval weights immediately

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx_lm.utils as mlx_utils

    vpath = Path(vindex_path)

    config = _build_config(vindex_path)
    model_class, model_args_class = mlx_utils._get_classes(config=config)
    model = model_class(model_args_class.from_dict(config))

    weights = _load_weights(vindex_path)

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)

    if not lazy:
        mx.eval(model.parameters())

    tokenizer = mlx_utils.load_tokenizer(vpath)

    return model, tokenizer
