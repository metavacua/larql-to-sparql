"""Numpy value layer: decode safetensors tensor bytes to float32 and compute
quantified per-tensor difference metrics. Pure measurement."""
import numpy as np


def decode_tensor(data, dtype):
    if dtype == "F32":
        return np.frombuffer(data, dtype="<f4").astype(np.float32, copy=True)
    if dtype == "F16":
        return np.frombuffer(data, dtype="<f2").astype(np.float32)
    if dtype == "BF16":
        u16 = np.frombuffer(data, dtype="<u2")
        u32 = (u16.astype(np.uint32) << 16)
        return u32.view(np.float32).astype(np.float32, copy=True)
    raise ValueError(f"unsupported dtype for decode: {dtype}")


def tensor_value_metrics(a, b):
    if a.shape != b.shape:
        return {"comparable": False, "shape_a": list(a.shape), "shape_b": list(b.shape)}
    diff = a.astype(np.float64) - b.astype(np.float64)
    return {
        "comparable": True,
        "n_total": int(a.size),
        "n_differing": int(np.count_nonzero(a != b)),
        "max_abs_diff": float(np.max(np.abs(diff))) if a.size else 0.0,
        "l2": float(np.sqrt(np.sum(diff * diff))),
    }
