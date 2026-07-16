#!/usr/bin/env python3
"""Descriptive-only matrix report, matching lql-strategy-matrix.yml's aggregate.py
convention (docs/specs/2026-07-16-larql-goose-toolcalling-design.md ADR-4): no
pass/fail-correctness judgement is applied here for the discovery/cli-measurement
legs (7-12) -- their outcome field is reported as-is, per ADR-3. The two
vm-coding-task legs (1-2) DO have a real correctness bar (a dispatched ToolRequest),
so their outcome is highlighted distinctly.

Usage: aggregate.py '<glob-pattern-for-result-*.json>' <out.md>
"""
import glob
import json
import sys

VM_LEG_APPROACH = "emulate-stream-harness"


def main() -> int:
    pattern, out_path = sys.argv[1], sys.argv[2]
    records = []
    for path in sorted(glob.glob(pattern)):
        try:
            with open(path, encoding="utf-8") as f:
                records.append(json.load(f))
        except Exception as e:
            records.append({"leg_id": path, "approach_id": "?", "outcome": "unreadable", "detail": str(e)})

    lines = ["## Tool-Calling Strategy Matrix — Results\n"]
    lines.append(f"{len(records)} leg result(s) collected.\n")
    lines.append("| leg_id | approach_id | outcome | detail |")
    lines.append("|---|---|---|---|")
    for r in records:
        leg_id = r.get("leg_id", "?")
        approach = r.get("approach_id", "?")
        outcome = r.get("outcome", "?")
        detail = str(r.get("detail", "")).replace("|", "\\|").replace("\n", " ")[:200]
        marker = ""
        if approach == VM_LEG_APPROACH:
            marker = " ✅" if outcome == "pass" else " ❌"
        lines.append(f"| `{leg_id}` | {approach} | {outcome}{marker} | {detail} |")

    lines.append("")
    lines.append(
        "Legs under `emulate-stream-harness` are the only pass/fail-gated legs "
        "(a real `ToolRequest` must have dispatched). All other legs are "
        "discovery/measurement, per ADR-3 — their outcome is data, not a merge "
        "blocker, regardless of what it says."
    )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
