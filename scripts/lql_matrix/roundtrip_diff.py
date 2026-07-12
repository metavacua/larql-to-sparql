"""Stdlib structural differ for model directories: file manifest, safetensors
header, JSON structural diff. Pure measurement — no interpretation."""
import hashlib
import json
import os
import struct


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_manifest(model_dir):
    out = {}
    for name in sorted(os.listdir(model_dir)):
        p = os.path.join(model_dir, name)
        if not os.path.isfile(p):
            continue
        out[name] = {"size": os.path.getsize(p), "sha256": sha256_file(p)}
    return out


def manifest_bijection(man_a, man_b):
    a, b = set(man_a), set(man_b)
    only_a = sorted(a - b)
    only_b = sorted(b - a)
    return {
        "only_a": only_a,
        "only_b": only_b,
        "in_both": sorted(a & b),
        "bijective": not only_a and not only_b,
    }


def read_safetensors_header(path):
    try:
        with open(path, "rb") as f:
            raw_len = f.read(8)
            if len(raw_len) != 8:
                return {"error": "truncated header length"}
            (hlen,) = struct.unpack("<Q", raw_len)
            hdr = json.loads(f.read(hlen).decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    metadata = hdr.pop("__metadata__", None)
    tensors = {}
    for name, spec in hdr.items():
        tensors[name] = {
            "dtype": spec.get("dtype"),
            "shape": spec.get("shape"),
            "data_offsets": spec.get("data_offsets"),
        }
    order = sorted(tensors, key=lambda n: (tensors[n]["data_offsets"] or [0])[0])
    return {"tensors": tensors, "metadata": metadata, "order": order}


def header_diff(ha, hb):
    if "error" in ha or "error" in hb:
        return {"error_a": ha.get("error"), "error_b": hb.get("error")}
    ta, tb = ha["tensors"], hb["tensors"]
    a, b = set(ta), set(tb)
    dtype_changes, shape_changes = {}, {}
    for name in sorted(a & b):
        if ta[name]["dtype"] != tb[name]["dtype"]:
            dtype_changes[name] = [ta[name]["dtype"], tb[name]["dtype"]]
        if ta[name]["shape"] != tb[name]["shape"]:
            shape_changes[name] = [ta[name]["shape"], tb[name]["shape"]]
    return {
        "tensor_only_a": sorted(a - b),
        "tensor_only_b": sorted(b - a),
        "dtype_changes": dtype_changes,
        "shape_changes": shape_changes,
        "order_equal": ha["order"] == hb["order"],
        "metadata_equal": ha["metadata"] == hb["metadata"],
        "metadata_a": ha["metadata"],
        "metadata_b": hb["metadata"],
    }
