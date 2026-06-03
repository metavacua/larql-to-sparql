"""
MLX inference from vindex — mmap'd weights, streamed from SSD.

Three modes, all use full weights (no feature dropping):

  larql.mlx.load()       — all weights in GPU memory. Fast. For models that fit.
  larql.streaming.load() — mmap'd weights, Metal pages on demand. For large models.
  walk_ffn.load()        — FFN in Rust (CPU). Vindex as editable knowledge layer.

The streaming path (this module) lets a 120B model run on 8GB MacBook:
  Total weights:   224 GB (on SSD)
  GPU memory:      ~one layer at a time (~120 MB), paged by Metal via mmap
  Speed:           SSD-bound (~1.5s/token at 3 GB/s NVMe)

Usage:
    from larql.streaming import load
    import mlx_lm

    model, tokenizer = load("gpt-oss-120b.vindex")
    response = mlx_lm.generate(model, tokenizer, prompt="...", max_tokens=20)
"""

import json
from pathlib import Path
from typing import Tuple


def load(vindex_path: str) -> Tuple:
    """Load MLX model with mmap'd weights — streams from SSD.

    Weights stay lazy (no mx.eval upfront). Metal pages them from SSD
    via mmap as each layer executes. Only the active layer's weights
    are in GPU memory at any time.

    All features used. Full dense FFN. No quality loss.

    For models that fit in GPU memory, use larql.mlx.load() instead —
    it's faster because weights are pre-loaded.

    Args:
        vindex_path: path to .vindex directory

    Returns:
        (model, tokenizer) — ready for mlx_lm.generate()
    """
    import mlx.core as mx
    import mlx_lm.utils as mlx_utils
    from larql.mlx import _build_config, _load_weights

    vpath = Path(vindex_path)

    with open(vpath / "index.json") as f:
        config = json.load(f)

    mlx_config = _build_config(str(vindex_path))
    model_class, model_args_class = mlx_utils._get_classes(config=mlx_config)
    model = model_class(model_args_class.from_dict(mlx_config))

    weights = _load_weights(str(vindex_path))

    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    model.eval()
    model.load_weights(list(weights.items()), strict=False)
    # No mx.eval — weights stay lazy, paged from SSD via mmap on demand

    tokenizer = mlx_utils.load_tokenizer(vpath)

    total_bytes = sum(
        e.get("length", 0) for e in
        json.load(open(vpath / "weight_manifest.json"))
    )
    # Add embeddings + gate vectors
    embed_path = vpath / "embeddings.bin"
    if embed_path.exists():
        total_bytes += embed_path.stat().st_size
    gate_path = vpath / "gate_vectors.bin"
    if gate_path.exists():
        total_bytes += gate_path.stat().st_size

    total_gb = total_bytes / 1e9
    num_layers = config.get("num_layers", 0)
    hidden = config.get("hidden_size", 0)
    intermediate = config.get("intermediate_size", 0)
    bpf = 2 if config.get("dtype", "f32") == "f16" else 4
    layer_mb = (hidden * intermediate * 3 + hidden * hidden * 4) * bpf / 1e6

    print(f"Streaming: {total_gb:.1f} GB on SSD, ~{layer_mb:.0f} MB/layer, "
          f"{num_layers} layers, mmap'd → Metal pages on demand")

    return model, tokenizer
