"""Round-trip quantified-difference matrix: active-model registry, comparison
enumeration, and top-level combine over model-directory pairs. Emits raw
measured differences only — no interpretation."""
import json
import os
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


def diff_model_dirs(dir_a, dir_b):
    man_a, man_b = D.file_manifest(dir_a), D.file_manifest(dir_b)
    bij = D.manifest_bijection(man_a, man_b)
    files = {}
    for name in bij["in_both"]:
        pa, pb = os.path.join(dir_a, name), os.path.join(dir_b, name)
        sha_eq = man_a[name]["sha256"] == man_b[name]["sha256"]
        if name.endswith(".safetensors"):
            files[name] = {
                "sha256_equal": sha_eq,
                "header": D.header_diff(D.read_safetensors_header(pa),
                                        D.read_safetensors_header(pb)),
                "values": V.safetensors_value_diff(pa, pb),
            }
        elif name.endswith(".json"):
            files[name] = {"sha256_equal": sha_eq,
                           "json": D.json_structural_diff(pa, pb)}
        else:
            files[name] = {"sha256_equal": sha_eq,
                           "size_a": man_a[name]["size"], "size_b": man_b[name]["size"]}
    return {"manifest": bij, "files": files}


def to_rows(meta, dir_diff):
    rows = []
    base = dict(meta)
    rows.append({**base, "file": "_manifest",
                 "bijective": dir_diff["manifest"]["bijective"],
                 "only_a": dir_diff["manifest"].get("only_a", []),
                 "only_b": dir_diff["manifest"].get("only_b", [])})
    for name, rec in dir_diff["files"].items():
        row = {**base, "file": name, "sha256_equal": rec.get("sha256_equal")}
        if "header" in rec:
            h = rec["header"]
            row["header_dtype_changes"] = h.get("dtype_changes", {})
            row["header_shape_changes"] = h.get("shape_changes", {})
            row["header_order_equal"] = h.get("order_equal")
            row["header_metadata_equal"] = h.get("metadata_equal")
            row["header_tensor_only_a"] = h.get("tensor_only_a", [])
            row["header_tensor_only_b"] = h.get("tensor_only_b", [])
            vals = rec.get("values", {})
            row["values_bytes_equal"] = vals.get("_bytes_equal")
            row["values_max_abs_diff"] = max(
                (m.get("max_abs_diff", 0.0) for k, m in vals.items()
                 if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable")),
                default=0.0)
            row["values_n_differing_total"] = sum(
                m.get("n_differing", 0) for k, m in vals.items()
                if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable"))
            row["values_l2_total"] = sum(
                m.get("l2", 0.0) for k, m in vals.items()
                if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable"))
            row["values_n_total_total"] = sum(
                m.get("n_total", 0) for k, m in vals.items()
                if k != "_bytes_equal" and isinstance(m, dict) and m.get("comparable"))
            if "error_a" in h or "error_b" in h:
                row["header_error_a"] = h.get("error_a")
                row["header_error_b"] = h.get("error_b")
            if isinstance(vals, dict) and "error" in vals:
                row["values_error"] = vals["error"]
        if "json" in rec:
            j = rec["json"]
            row["json_changed"] = j.get("changed", {})
            row["json_only_a_paths"] = j.get("only_a_paths", [])
            row["json_only_b_paths"] = j.get("only_b_paths", [])
            row["json_byte_identical"] = j.get("byte_identical")
            if "error_a" in j or "error_b" in j:
                row["json_error_a"] = j.get("error_a")
                row["json_error_b"] = j.get("error_b")
        if "size_a" in rec or "size_b" in rec:
            row["size_a"] = rec.get("size_a")
            row["size_b"] = rec.get("size_b")
        rows.append(row)
    return rows


def render_markdown(rows):
    cols = ["model", "variant", "driver", "mode", "insert_form", "comparison",
            "file", "sha256_equal"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    return "\n".join(lines) + "\n"


def main(argv):
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True)
    ap.add_argument("--b", required=True)
    ap.add_argument("--meta", required=True, help="JSON meta dict")
    ap.add_argument("--out", required=True)
    ap.add_argument("--md")
    ns = ap.parse_args(argv)
    try:
        meta = json.loads(ns.meta)
        rows = to_rows(meta, diff_model_dirs(ns.a, ns.b))
    except (ValueError, OSError) as e:
        with open(ns.out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"file": "_error", "error": str(e)}) + "\n")
        return 1
    with open(ns.out, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    if ns.md:
        with open(ns.md, "w", encoding="utf-8") as f:
            f.write(render_markdown(rows))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
