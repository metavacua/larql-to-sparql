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
            filesize = os.fstat(f.fileno()).st_size
            if hlen > filesize - 8:
                return {"error": "header length exceeds file size"}
            hdr = json.loads(f.read(hlen).decode("utf-8"))
    except (OSError, ValueError, MemoryError) as e:
        return {"error": f"{type(e).__name__}: {e}"}
    if not isinstance(hdr, dict):
        return {"error": "header is not a JSON object"}
    metadata = hdr.pop("__metadata__", None)
    tensors = {}
    for name, spec in hdr.items():
        if not isinstance(spec, dict):
            return {"error": f"tensor spec for {name} is not an object"}
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


def _flatten(obj, prefix=""):
    out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.update(_flatten(v, f"{prefix}.{k}" if prefix else str(k)))
    else:
        out[prefix] = obj
    return out


def json_structural_diff(path_a, path_b):
    try:
        with open(path_a, "rb") as f:
            raw_a = f.read()
        obj_a = json.loads(raw_a.decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error_a": f"{type(e).__name__}: {e}"}
    try:
        with open(path_b, "rb") as f:
            raw_b = f.read()
        obj_b = json.loads(raw_b.decode("utf-8"))
    except (OSError, ValueError) as e:
        return {"error_b": f"{type(e).__name__}: {e}"}
    fa, fb = _flatten(obj_a), _flatten(obj_b)
    a, b = set(fa), set(fb)
    changed = {p: [fa[p], fb[p]] for p in sorted(a & b) if fa[p] != fb[p]}
    return {
        "only_a_paths": sorted(a - b),
        "only_b_paths": sorted(b - a),
        "changed": changed,
        "byte_identical": raw_a == raw_b,
    }
