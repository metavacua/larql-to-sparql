#!/usr/bin/env python3
"""Aggregate per-leg round-trip diff JSON docs (one `{"leg":...,
"comparisons":[...]}` object per leg, written by `roundtrip_compile.py`) into
one catalogue. Flattens every doc's `comparisons` rows, tagging each with its
`leg`, and renders via the EXISTING `roundtrip_matrix.render_markdown`.
Tolerant of malformed/partial files — a bad JSON file or a leg-level error
doc (no `comparisons` key) is skipped, never fatal."""
import glob
import json
import sys
from pathlib import Path

import roundtrip_matrix as M


def load(pattern):
    rows = []
    for path in sorted(glob.glob(pattern)):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
            doc = json.loads(text)
        except (OSError, ValueError):
            continue
        if not isinstance(doc, dict):
            continue
        leg = doc.get("leg", "?")
        for row in doc.get("comparisons") or []:
            if isinstance(row, dict):
                rows.append({**row, "leg": leg})
    return rows


def main(pattern, out_json, out_md):
    rows = load(pattern)
    Path(out_json).write_text(json.dumps(rows, indent=2), encoding="utf-8")
    Path(out_md).write_text(M.render_markdown(rows), encoding="utf-8")
    print(f"wrote {out_json} and {out_md} ({len(rows)} rows)")


if __name__ == "__main__":
    pat = sys.argv[1] if len(sys.argv) > 1 else "roundtrip-*.json"
    oj = sys.argv[2] if len(sys.argv) > 2 else "roundtrip-catalogue.json"
    om = sys.argv[3] if len(sys.argv) > 3 else "roundtrip-catalogue.md"
    main(pat, oj, om)
