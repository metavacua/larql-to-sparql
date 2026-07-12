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
