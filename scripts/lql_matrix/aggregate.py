#!/usr/bin/env python3
"""Aggregate per-leg LQL matrix results into one Markdown report. Descriptive
only — no pass/fail-correctness judgement. A "leg" is one (source × recipe ×
level) job; its meta row's `level` field carries the leg name.

Reads run_matrix JSONL (`{"type":"meta",...}` + per-cell rows), plus
`descriptor-<leg>.json` and `produce-<leg>.json` (best-effort, from cwd). utf-8
+ tolerant of malformed lines.
"""
import glob
import json
import sys
from pathlib import Path


def load(results_glob):
    rows, cats, order, meta, bad = {}, {}, [], {}, 0
    for path in sorted(glob.glob(results_glob)):
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for line in text.splitlines():
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
                cid, leg = r["id"], r["level"]
            except KeyError:
                bad += 1
                continue
            if cid not in rows:
                rows[cid], cats[cid] = {}, r.get("cat", "")
                order.append(cid)
            rows[cid][leg] = r
    return rows, cats, order, meta, bad


def load_json_glob(pattern, key):
    out = {}
    for p in glob.glob(pattern) + glob.glob("**/" + pattern, recursive=True):
        try:
            d = json.loads(Path(p).read_text(encoding="utf-8"))
            if d.get(key):
                out[d[key]] = d
        except Exception:
            pass
    return out


def mark(r):
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


def first_err(r):
    return (r.get("err_line") or "").replace("|", "\\|") if r else ""


def main(results_glob, out_md):
    rows, cats, order, meta, bad = load(results_glob)
    desc = load_json_glob("descriptor-*.json", "name")
    prod = load_json_glob("produce-*.json", "name")
    legs = sorted(set(list(meta) + [l for c in rows for l in rows[c]]))

    L = ["# LQL Strategy Matrix — raw outcomes", "",
         "Each cell = the **mechanical** result of a command against a vindex "
         "produced by that leg's recipe. **No output-correctness claim.** Legend: "
         "✅ exit 0 clean · ⚠️ exit 0 **but** error/refusal text · ❌N non-zero N · "
         "💥N crash (OOM/SIGSEGV/SIGABRT) · ⏱ timeout · · not run.", "",
         f"Legs: **{len(legs)}** · commands/leg: **{len(order)}**"
         + (f" · ⚠️ {bad} malformed line(s) skipped" if bad else ""), ""]

    # Produced-vindex descriptor conformance — did the recipe yield what we expected?
    if desc:
        L += ["## Produced-vindex descriptor", "",
              "Recorded from each produced `index.json`. `quant?` = observed quant "
              "(ternary via `bitnet_layout`) vs the leg's expected quant.", "",
              "| leg | produce | family | dtype | quant (obs/exp) | quant? | hidden | vindex MB |",
              "|---|---|---|---|---|:--:|---|---|"]
        for lg in legs:
            d = desc.get(lg, {})
            p = prod.get(lg, {})
            pb = p.get("bucket", "?")
            produced = "❌ " + pb if pb != "ok" else "ok"
            if not d.get("produced", True) or d.get("error"):
                L.append(f"| `{lg}` | {produced} | — | — | — | ❌ | — | {p.get('vindex_mb','?')} |")
                continue
            obs, exp = d.get("observed_quant"), d.get("expect_quant")
            ok = "✅" if d.get("quant_match") else "❌"
            L.append(f"| `{lg}` | {produced} | {d.get('family')} | {d.get('dtype')} | "
                     f"{obs}/{exp} | {ok} | {d.get('hidden_size','?')} | {p.get('vindex_mb','?')} |")
        L.append("")

    # Per-leg tally (the primary view when legs are many)
    L += ["## Tally (per leg)", "",
          "| leg | ✅ ok | ⚠️ errtext | ❌ err | ⏱ timeout | 💥 crash | peak RSS MB |",
          "|---|---|---|---|---|---|---|"]
    for lg in legs:
        t = {"ok": 0, "errtext": 0, "err": 0, "timeout": 0, "crash": 0}
        best_rss = 0
        for cid in rows:
            r = rows[cid].get(lg)
            if not r:
                continue
            b = r.get("bucket")
            if b == "ok" and r.get("err_signal"):
                t["errtext"] += 1
            elif b == "ok":
                t["ok"] += 1
            else:
                t[b] = t.get(b, 0) + 1
            rss = r.get("peak_rss_kb") or 0
            if isinstance(rss, int) and rss > best_rss:
                best_rss = rss
        rss_mb = f"{best_rss/1024:.0f}" if best_rss else "n/a"
        L.append(f"| `{lg}` | {t['ok']} | {t['errtext']} | {t['err']} | "
                 f"{t['timeout']} | {t['crash']} | {rss_mb} |")

    # Command × leg pivot — only when narrow enough to read; else it's in artifacts.
    if len(legs) <= 8:
        L += ["", "## Command × leg", "",
              "| command | cat | " + " | ".join(legs) + " |",
              "|---|---|" + "---|" * len(legs)]
        for cid in order:
            L.append(f"| `{cid}` | {cats[cid]} | "
                     + " | ".join(mark(rows[cid].get(lg)) for lg in legs) + " |")
    else:
        L += ["", f"> Command × leg pivot omitted ({len(legs)} legs too wide); "
              "per-cell detail is in the `results-*` artifacts.", ""]

    # Failures & refusals detail — evidence inline, collapsed
    fails = [(lg, cid, mark(r), r.get("exit_code"), r.get("peak_rss_kb"), first_err(r))
             for cid in order for lg in legs
             if (r := rows[cid].get(lg)) and (r.get("bucket") != "ok" or r.get("err_signal"))]
    L += ["", f"## Failures & refusals ({len(fails)})", ""]
    if fails:
        L += ["<details><summary>leg · cell · mark · exit · rss(kb) · first error line</summary>",
              "", "| leg | cell | mark | exit | rss | first error line |",
              "|---|---|---|---|---|---|"]
        L += [f"| `{lg}` | `{cid}` | {mk} | {ec} | {rss} | `{el}` |"
              for lg, cid, mk, ec, rss, el in fails]
        L += ["", "</details>"]
    else:
        L.append("None — every cell exited 0 with no error/refusal text.")

    Path(out_md).write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out_md} ({len(legs)} legs × {len(order)} commands, "
          f"{len(fails)} failures/refusals, {bad} bad lines)")


if __name__ == "__main__":
    g = sys.argv[1] if len(sys.argv) > 1 else "results-*/*.jsonl"
    o = sys.argv[2] if len(sys.argv) > 2 else "lql-matrix.md"
    main(g, o)
