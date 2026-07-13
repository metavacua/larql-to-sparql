"""Round-trip quantified-difference matrix: structural+value diff over
model-directory pairs, and the row/markdown rendering shared by the
round-trip orchestrator (`roundtrip_compile.py`) and aggregator
(`roundtrip_aggregate.py`). Emits raw measured differences only — no
interpretation."""
import json
import os
import sys

import roundtrip_diff as D
import roundtrip_value as V

# Declared output schema: the ONLY top-level fields `to_rows` may add to a row
# beyond the caller-supplied `meta`. This is the positive-existence contract —
# a consumer reads it to know every field, and the test asserts to_rows emits
# nothing outside it (so an interpretation field under ANY name is caught, not
# just a hardcoded denylist of a few). Every field here is a measured quantity,
# a machine-checkable predicate, or a factual "could-not-measure" error string.
ROW_SCHEMA = frozenset({
    "file",
    # manifest row
    "bijective", "only_a", "only_b",
    # per-file common
    "sha256_equal",
    # safetensors header
    "header_dtype_changes", "header_shape_changes", "header_order_equal",
    "header_metadata_equal", "header_tensor_only_a", "header_tensor_only_b",
    "header_metadata_a", "header_metadata_b", "header_error_a", "header_error_b",
    # safetensors values
    "values_bytes_equal", "values_max_abs_diff", "values_n_differing_total",
    "values_l2_total", "values_n_total_total", "values_per_tensor", "values_error",
    # json
    "json_changed", "json_only_a_paths", "json_only_b_paths", "json_byte_identical",
    "json_error_a", "json_error_b",
    # other files
    "size_a", "size_b",
})


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
                "values": V.safetensors_value_diff(pa, pb, bytes_equal=sha_eq),
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
            if h.get("metadata_equal") is False:
                row["header_metadata_a"] = h.get("metadata_a")
                row["header_metadata_b"] = h.get("metadata_b")
            vals = rec.get("values", {})
            row["values_bytes_equal"] = vals.get("_bytes_equal")
            comparable = [m for k, m in vals.items()
                          if k != "_bytes_equal" and isinstance(m, dict)
                          and m.get("comparable")]
            row["values_max_abs_diff"] = max(
                (m.get("max_abs_diff", 0.0) for m in comparable), default=0.0)
            row["values_n_differing_total"] = sum(m.get("n_differing", 0) for m in comparable)
            row["values_l2_total"] = sum(m.get("l2", 0.0) for m in comparable)
            row["values_n_total_total"] = sum(m.get("n_total", 0) for m in comparable)
            # Positive existence: emit EVERY tensor's difference record — comparable
            # metrics AND incomparable reasons (shape mismatch / spec / decode). The
            # aggregates above are a derived summary over the comparable subset; without
            # this, a divergence can't be localized to a tensor and the omission is a
            # silent absence a consumer can't account for.
            row["values_per_tensor"] = {
                k: m for k, m in vals.items()
                if k != "_bytes_equal" and isinstance(m, dict)}
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
