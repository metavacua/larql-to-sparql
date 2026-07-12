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
