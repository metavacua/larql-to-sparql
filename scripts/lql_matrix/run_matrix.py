#!/usr/bin/env python3
"""LQL strategy-matrix runner — EMPIRICAL, NOT a validation suite.

Runs every LQL command in a corpus against one already-extracted vindex and
records RAW MECHANICAL OUTCOMES ONLY per cell: exit code, coarse bucket
(ok / err / timeout / crash), wall duration, peak RSS, and an *error-text
signal* (did stdout/stderr contain an error/refusal even though the process
exited 0 — `larql lql` exits 0 on in-band errors). It makes NO judgement about
whether output is "correct"; a break is data to read. A failing cell never
aborts the run.

I/O practices (deliberate):
  * FULL stdout/stderr of every cell are written to per-cell files under
    <out_dir>/cells/ and kept for artifact upload — nothing is captured then
    discarded, and a panic's backtrace TAIL survives.
  * Stream *content* never transits the shell argv. This driver reads the
    captured files itself (utf-8, errors="replace") and truncates by CODEPOINT,
    so a multibyte boundary can never corrupt a row.
  * One process orchestrates each cell via an argv list (no shell, no
    injection); WRAP is shlex-split so containment prefixes compose cleanly.

Usage:   run_matrix.py <level> <vindex_dir> <corpus.jsonl> <out.jsonl>
Env:     LARQL_BIN (default target/release/larql), MODEL_ID, TMPROOT,
         WRAP (e.g. "larql-probe safe --mem 5600 --"), CELL_TIMEOUT (default 900),
         plus GITHUB_SHA / RUNNER_OS / RUST_BACKTRACE recorded as provenance.
"""
import datetime
import json
import os
import pathlib
import re
import shlex
import subprocess
import sys
import tempfile
import time

# larql prints "Error: …" for in-band failures and Rust prints "… panicked at …";
# case-sensitive on purpose — matches what larql/Rust actually emit, avoids
# flagging benign prose that merely contains the word "error".
ERR_RE = re.compile(r"Error:|panicked|Parse error|note: run with RUST_BACKTRACE")
RSS_RE = re.compile(r"Maximum resident set size.*?:\s*(\d+)")
CLIP = 800  # codepoint budget for head/tail snapshots in the JSONL


def bucket_for(rc: int) -> str:
    if rc == 0:
        return "ok"
    if rc == 124:
        return "timeout"
    if rc in (137, 139, 134) or (rc < 0 and -rc in (9, 11, 6)):
        return "crash"  # SIGKILL(OOM)/SIGSEGV/SIGABRT
    return "err"


def first_error_line(text: str) -> str:
    for ln in text.splitlines():
        if ERR_RE.search(ln):
            return ln.strip()[:200]
    return ""


def main() -> None:
    level, vindex, corpus, out = sys.argv[1:5]
    larql = os.environ.get("LARQL_BIN", "target/release/larql")
    model = os.environ.get("MODEL_ID", "")
    tmproot = os.environ.get("TMPROOT") or tempfile.mkdtemp()
    wrap = shlex.split(os.environ.get("WRAP", ""))
    cell_timeout = os.environ.get("CELL_TIMEOUT", "900")
    os.makedirs(tmproot, exist_ok=True)

    out_path = pathlib.Path(out)
    cells_dir = out_path.parent / "cells"
    cells_dir.mkdir(parents=True, exist_ok=True)
    time_bin = "/usr/bin/time" if pathlib.Path("/usr/bin/time").exists() else None

    def subst(s: str) -> str:
        return (s.replace("{{VINDEX}}", vindex)
                 .replace("{{MODEL}}", model)
                 .replace("{{TMP}}", tmproot))

    # ── provenance row (anchors the artifact to a commit/binary/model/runner) ──
    try:
        ver = subprocess.run([larql, "--version"], capture_output=True,
                             text=True, timeout=30).stdout.strip()
    except Exception:
        ver = ""
    meta = {
        "type": "meta", "level": level, "larql_version": ver,
        "commit": os.environ.get("GITHUB_SHA", ""),
        "model": model, "runner_os": os.environ.get("RUNNER_OS", ""),
        "vindex": vindex, "backtrace": os.environ.get("RUST_BACKTRACE", ""),
        "time_bin": bool(time_bin), "wrap": " ".join(wrap),
        "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    }
    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")

    n = 0
    with open(corpus, encoding="utf-8") as cf:
        for line in cf:
            line = line.strip()
            if not line:
                continue
            c = json.loads(line)
            cid, cat, lql = c["id"], c.get("cat", ""), subst(c["lql"])

            outf = cells_dir / f"{level}.{cid}.out"
            errf = cells_dir / f"{level}.{cid}.err"
            timef = cells_dir / f"{level}.{cid}.time"

            argv = ["timeout", "--kill-after=10", cell_timeout, *wrap]
            if time_bin:
                argv += [time_bin, "-v", "-o", str(timef)]
            argv += [larql, "lql", lql]

            t0 = time.monotonic()
            with outf.open("wb") as so, errf.open("wb") as se:
                rc = subprocess.run(argv, stdout=so, stderr=se).returncode
            dur_ms = int((time.monotonic() - t0) * 1000)
            bucket = bucket_for(rc)

            # read the FULL captured streams back (utf-8, tolerant, codepoint-safe)
            so_b, se_b = outf.read_bytes(), errf.read_bytes()
            so_txt = so_b.decode("utf-8", "replace")
            se_txt = se_b.decode("utf-8", "replace")
            combined = so_txt + "\n" + se_txt
            err_signal = 1 if ERR_RE.search(combined) else 0

            peak_rss_kb = ""
            if time_bin and timef.exists():
                m = RSS_RE.search(timef.read_text("utf-8", "replace"))
                if m:
                    peak_rss_kb = int(m.group(1))
                timef.unlink()  # RSS captured; keep only .out/.err for artifacts

            row = {
                "level": level, "id": cid, "cat": cat, "lql": lql,
                "exit_code": rc, "bucket": bucket, "duration_ms": dur_ms,
                "peak_rss_kb": peak_rss_kb, "err_signal": err_signal,
                "err_line": first_error_line(combined) if err_signal else "",
                "stdout_bytes": len(so_b), "stderr_bytes": len(se_b),
                "stdout_head": so_txt[:CLIP], "stderr_head": se_txt[:CLIP],
                "stderr_tail": se_txt[-CLIP:],
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            flag = " ERR*" if err_signal else ""
            print(f"[{level}] {cid} -> exit={rc} {bucket}{flag} "
                  f"{dur_ms}ms rss={peak_rss_kb}", file=sys.stderr)
            n += 1

    print(f"wrote {n} rows + provenance to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
