"""Numpy value layer: decode safetensors tensor bytes to float32 and compute
quantified per-tensor difference metrics. Pure measurement."""
import struct
import numpy as np
import roundtrip_diff as _D


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


def _tensor_bytes(path, spec, header_len):
    begin, end = spec["data_offsets"]
    with open(path, "rb") as f:
        f.seek(8 + header_len + begin)
        return f.read(end - begin)


def _header_len(path):
    with open(path, "rb") as f:
        return struct.unpack("<Q", f.read(8))[0]


def safetensors_value_diff(path_a, path_b):
    ha, hb = _D.read_safetensors_header(path_a), _D.read_safetensors_header(path_b)
    if "error" in ha or "error" in hb:
        return {"error": ha.get("error") or hb.get("error")}
    la, lb = _header_len(path_a), _header_len(path_b)
    out = {}
    for name in sorted(set(ha["tensors"]) & set(hb["tensors"])):
        sa, sb = ha["tensors"][name], hb["tensors"][name]
        try:
            aa = decode_tensor(_tensor_bytes(path_a, sa, la), sa["dtype"]).reshape(sa["shape"])
            bb = decode_tensor(_tensor_bytes(path_b, sb, lb), sb["dtype"]).reshape(sb["shape"])
        except ValueError as e:
            out[name] = {"comparable": False, "decode_error": str(e)}
            continue
        out[name] = tensor_value_metrics(aa, bb)
    out["_bytes_equal"] = _D.sha256_file(path_a) == _D.sha256_file(path_b)
    return out
