#!/usr/bin/env python3
"""Aggregate per-level LQL matrix results into one Markdown report. Descriptive
only — reports the raw mechanical outcome (bucket + exit code) per (command,
level), plus an error-text overlay, resource stats, and a failures/refusals
detail block. No pass/fail-correctness judgement is made or implied.

Reads JSONL produced by run_matrix.py: one `{"type":"meta",…}` provenance row
per level followed by one row per cell. Tolerant of malformed lines (skips
them, loudly) and pins utf-8 so non-ASCII model output never breaks decoding.
"""
import glob
import json
import sys
from pathlib import Path

LEVEL_ORDER = ["browse", "attention", "inference", "all"]


def load(results_glob):
    rows, cats, order, meta = {}, {}, [], {}
    bad = 0
    for path in sorted(glob.glob(results_glob)):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if r.get("type") == "meta":
                meta[r.get("level", "?")] = r
                continue
            try:
                cid, lvl = r["id"], r["level"]
            except KeyError:
                bad += 1
                continue
            if cid not in rows:
                rows[cid], cats[cid] = {}, r.get("cat", "")
                order.append(cid)
            rows[cid][lvl] = r
    return rows, cats, order, meta, bad


def mark(r):
    """One-glance cell mark. ⚠️ = exited 0 but printed an error/refusal."""
    if r is None:
        return "·"
    b, ec, es = r.get("bucket"), r.get("exit_code", "?"), r.get("err_signal", 0)
    if b == "ok":
        return "⚠️" if es else "✅"
    if b == "err":
        return f"❌{ec}"
    if b == "crash":
        return f"💥{ec}"
    if b == "timeout":
        return "⏱"
    return str(b)


def load_extract():
    """Best-effort scan for extract-<level>.json (written by the workflow)."""
    ex = {}
    for p in glob.glob("extract-*.json") + glob.glob("**/extract-*.json", recursive=True):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            ex[d["level"]] = d
        except Exception:
            pass
    return ex


def main(results_glob, out_md):
    rows, cats, order, meta, bad = load(results_glob)
    extract = load_extract()
    present = [l for l in LEVEL_ORDER if any(l in rows[c] for c in rows)] or LEVEL_ORDER
    expected = max((len(rows[c]) for c in rows), default=0)  # cells in the fullest leg
    counts = {l: sum(1 for c in rows if l in rows[c]) for l in present}

    L = ["# LQL Strategy Matrix — raw outcomes", "",
         "Each cell is the **mechanical** result of running the command against a "
         "vindex extracted at that level. **No claim is made about output "
         "correctness.** Legend: ✅ exit 0 clean · ⚠️ exit 0 **but** stdout/stderr "
         "carried an error/refusal (`larql lql` exits 0 in-band) · ❌N non-zero "
         "exit N · 💥N crash (OOM/SIGSEGV/SIGABRT) · ⏱ per-cell timeout · · not run.",
         ""]

    if bad:
        L += [f"> ⚠️ {bad} malformed result line(s) skipped.", ""]

    # incomplete-leg flag (B4): a leg with fewer cells than the fullest one
    incomplete = [f"`{l}` ({counts[l]}/{expected})"
                  for l in present if counts[l] < expected]
    if incomplete:
        L += ["> ⚠️ **Incomplete legs** (fewer cells than the fullest — runner "
              "died / whole-leg timeout mid-corpus): " + ", ".join(incomplete), ""]

    # provenance
    if meta:
        L += ["## Provenance", "",
              "| level | commit | larql | model | runner | started (UTC) | backtrace |",
              "|---|---|---|---|---|---|---|"]
        for l in present:
            m = meta.get(l, {})
            L.append("| {} | `{}` | {} | {} | {} | {} | {} |".format(
                l, (m.get("commit") or "?")[:10], m.get("larql_version") or "?",
                m.get("model") or "?", m.get("runner_os") or "?",
                (m.get("started_at") or "?")[:19], m.get("backtrace") or "unset"))
        L.append("")

    # extraction outcome (itself an observed cell)
    if extract:
        L += ["## Extraction outcome per level", "",
              "| metric | " + " | ".join(present) + " |",
              "|---|" + "---|" * len(present)]
        def erow(label, fn):
            return "| **" + label + "** | " + " | ".join(
                fn(extract.get(l)) for l in present) + " |"
        L.append(erow("outcome", lambda e: f"{e.get('bucket','?')} ({e.get('duration_ms','?')}ms)" if e else "·"))
        L.append(erow("peak RSS (MB)", lambda e: f"{(e.get('peak_rss_kb') or 0)/1024:.0f}" if e and e.get('peak_rss_kb') else ("n/a" if e else "·")))
        L.append(erow("vindex (MB)", lambda e: str(e.get('vindex_mb', '?')) if e else "·"))
        L.append("")

    # command × level
    L += ["## Command × level", "",
          "| command | cat | " + " | ".join(present) + " |",
          "|---|---|" + "---|" * len(present)]
    for cid in order:
        cells = " | ".join(mark(rows[cid].get(l)) for l in present)
        L.append(f"| `{cid}` | {cats[cid]} | {cells} |")

    # tally (adds errtext = exit-0-but-error overlay)
    L += ["", "## Tally (per level)", "",
          "| level | ✅ ok | ⚠️ errtext | ❌ err | ⏱ timeout | 💥 crash |",
          "|---|---|---|---|---|---|"]
    for l in present:
        t = {"ok": 0, "errtext": 0, "err": 0, "timeout": 0, "crash": 0}
        for cid in rows:
            r = rows[cid].get(l)
            if not r:
                continue
            b = r.get("bucket")
            if b == "ok" and r.get("err_signal"):
                t["errtext"] += 1
            elif b == "ok":
                t["ok"] += 1
            else:
                t[b] = t.get(b, 0) + 1
        L.append(f"| {l} | {t['ok']} | {t['errtext']} | {t['err']} | "
                 f"{t['timeout']} | {t['crash']} |")

    # resource stats (perf): peak RSS + slowest cell per level
    L += ["", "## Resource (per level)", "",
          "| level | peak RSS (MB) | slowest cell | slowest (ms) |",
          "|---|---|---|---|"]
    for l in present:
        best_rss, slow_id, slow_ms = 0, "-", 0
        for cid in rows:
            r = rows[cid].get(l)
            if not r:
                continue
            rss = r.get("peak_rss_kb") or 0
            if isinstance(rss, int) and rss > best_rss:
                best_rss = rss
            d = r.get("duration_ms") or 0
            if isinstance(d, int) and d > slow_ms:
                slow_ms, slow_id = d, cid
        rss_mb = f"{best_rss/1024:.0f}" if best_rss else "n/a"
        L.append(f"| {l} | {rss_mb} | `{slow_id}` | {slow_ms} |")

    # failures & refusals detail (C1): evidence inline, collapsed
    fails = []
    for cid in order:
        for l in present:
            r = rows[cid].get(l)
            if r and (r.get("bucket") != "ok" or r.get("err_signal")):
                fails.append((l, cid, mark(r), r.get("exit_code"),
                              r.get("peak_rss_kb"), r.get("err_line", "")))
    L += ["", f"## Failures & refusals ({len(fails)})", ""]
    if fails:
        L += ["<details><summary>level · cell · mark · exit · rss(kb) · first error line</summary>",
              "", "| level | cell | mark | exit | rss | first error line |",
              "|---|---|---|---|---|---|"]
        for l, cid, mk, ec, rss, el in fails:
            el = (el or "").replace("|", "\\|")
            L.append(f"| {l} | `{cid}` | {mk} | {ec} | {rss} | `{el}` |")
        L += ["", "</details>"]
    else:
        L.append("None — every cell exited 0 with no error/refusal text.")

    Path(out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_md} ({len(order)} commands x {len(present)} levels, "
          f"{len(fails)} failures/refusals, {bad} bad lines)")


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "results-*/*.jsonl"
    o = sys.argv[2] if len(sys.argv) > 2 else "lql-matrix.md"
    main(g, o)
