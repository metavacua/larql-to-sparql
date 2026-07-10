#!/usr/bin/env python3
"""Aggregate per-level LQL matrix results into one table. Descriptive only —
reports the raw mechanical outcome (bucket + exit code) per (command, level).
No pass/fail-correctness judgement is made or implied."""
import json
import sys
import glob
from pathlib import Path

LEVEL_ORDER = ["browse", "attention", "inference", "all"]


def main(results_glob: str, out_md: str) -> None:
    rows = {}       # id -> {level -> (bucket, exit_code, duration_ms)}
    cats = {}       # id -> category
    order = []      # preserve first-seen command order
    extract = {}    # level -> extraction outcome dict (from extract-<level>.json if present)

    for path in sorted(glob.glob(results_glob)):
        for line in Path(path).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            cid, lvl = r["id"], r["level"]
            if cid not in rows:
                rows[cid] = {}
                cats[cid] = r.get("cat", "")
                order.append(cid)
            rows[cid][lvl] = (r.get("bucket", "?"), r.get("exit_code", "?"),
                              r.get("duration_ms", "?"))

    for ep in glob.glob("**/extract-*.json", recursive=True):
        try:
            d = json.loads(Path(ep).read_text())
            extract[d["level"]] = d
        except Exception:
            pass

    levels = [lvl for lvl in LEVEL_ORDER
              if any(lvl in rows[c] for c in rows)] or LEVEL_ORDER

    def cell(bucket_ec):
        if bucket_ec is None:
            return "·"
        bucket, ec, _ = bucket_ec
        mark = {"ok": "ok", "err": f"err{ec}", "timeout": "TIMEOUT",
                "crash": f"CRASH{ec}"}.get(bucket, str(bucket))
        return mark

    lines = ["# LQL Strategy Matrix — raw outcomes",
             "",
             "Empirical: each cell is the **mechanical** result of running the "
             "command against a vindex extracted at that level. `ok`=exit 0, "
             "`errN`=non-zero exit N, `TIMEOUT`=killed by per-cell cap, "
             "`CRASH`=SIGKILL/OOM/SIGSEGV/SIGABRT, `·`=not run. **No claim is "
             "made about whether output is correct** — only what happened.",
             ""]

    # extraction row
    if extract:
        lines += ["## Extraction outcome per level", ""]
        lines.append("| level | " + " | ".join(levels) + " |")
        lines.append("|---|" + "---|" * len(levels))
        erow = []
        for lvl in levels:
            e = extract.get(lvl)
            erow.append(f"{e.get('bucket','?')} ({e.get('duration_ms','?')}ms)"
                        if e else "·")
        lines.append("| **extract** | " + " | ".join(erow) + " |")
        lines.append("")

    lines += ["## Command × level", "",
              "| command | cat | " + " | ".join(levels) + " |",
              "|---|---|" + "---|" * len(levels)]
    for cid in order:
        cells = " | ".join(cell(rows[cid].get(lvl)) for lvl in levels)
        lines.append(f"| `{cid}` | {cats[cid]} | {cells} |")

    # bucket tally
    lines += ["", "## Tally (per level)", "",
              "| level | ok | err | timeout | crash |",
              "|---|---|---|---|---|"]
    for lvl in levels:
        t = {"ok": 0, "err": 0, "timeout": 0, "crash": 0}
        for cid in rows:
            be = rows[cid].get(lvl)
            if be:
                t[be[0]] = t.get(be[0], 0) + 1
        lines.append(f"| {lvl} | {t['ok']} | {t['err']} | {t['timeout']} | {t['crash']} |")

    Path(out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {out_md} ({len(order)} commands x {len(levels)} levels)")


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "results-*/*.jsonl"
    o = sys.argv[2] if len(sys.argv) > 2 else "lql-matrix.md"
    main(g, o)
