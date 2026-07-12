"""Round-trip quantified-difference matrix: active-model registry, comparison
enumeration, and top-level combine over model-directory pairs. Emits raw
measured differences only — no interpretation."""
import json
import sys

import roundtrip_diff as D
import roundtrip_value as V

REGISTRY = [
    {"id": "smol135", "active": True, "variants": [
        {"variant": "instruct-bf16", "repo": "HuggingFaceTB/SmolLM2-135M-Instruct", "src_dtype": "BF16"},
        {"variant": "base-f32", "repo": "HuggingFaceTB/SmolLM2-135M", "src_dtype": "F32"},
    ]},
    {"id": "qwen05", "active": False, "variants": []},
    {"id": "smol360", "active": False, "variants": []},
    {"id": "qwen15", "active": False, "variants": []},
    {"id": "granite1b", "active": False, "variants": []},
    {"id": "bitnet2b", "active": False, "variants": []},
]

DRIVERS = ("lql", "cli")
INSERT_FORMS = ("knn", "compose")


def active_variants(registry=REGISTRY):
    out = []
    for m in registry:
        if not m["active"]:
            continue
        for v in m["variants"]:
            out.append((m["id"], v["variant"], v["repo"], v["src_dtype"]))
    return out


def enumerate_comparisons(model_id, variant):
    rows = []
    for driver in DRIVERS:
        rows.append({"model": model_id, "variant": variant, "driver": driver,
                     "mode": "A", "insert_form": None, "comparison": "input_vs_A"})
        for form in INSERT_FORMS:
            rows.append({"model": model_id, "variant": variant, "driver": driver,
                         "mode": "B", "insert_form": form, "comparison": "B_vs_A"})
    rows.append({"model": model_id, "variant": variant, "driver": "lql+cli",
                 "mode": "A", "insert_form": None, "comparison": "lqlA_vs_cliA"})
    for form in INSERT_FORMS:
        rows.append({"model": model_id, "variant": variant, "driver": "lql+cli",
                     "mode": "B", "insert_form": form, "comparison": "lqlB_vs_cliB"})
    return rows
