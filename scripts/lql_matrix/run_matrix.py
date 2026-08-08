#!/usr/bin/env python3
"""LQL strategy-matrix runner — runs the commands and reports what happened.

Runs every LQL command in a corpus against one already-extracted vindex.
The COMPLETE stdout/stderr of every cell goes to <out_dir>/cells/ and is
uploaded as an artifact. That is the result. Reading it is a separate activity.

This driver derives NOTHING from the output. It previously emitted a coarse
bucket, an error-text boolean, a first-error line, and 800-codepoint head/tail
snapshots — and every downstream consumer read those instead of the captures,
which is how a cell whose output said `failed to write model` came to be
reported as a `BEGIN PATCH` parse error: `err_line` takes the FIRST error in a
batch, and a corpus cell is a batch of statements. Across a run, 69% of the raw
bytes never reached the rows at all, to save 1.3 MiB out of 1.9 MiB total.

The JSONL row is now an INDEX into the captures, not a description of them:
which cell ran, what LQL it ran, the process exit code, how long it took, how
much memory it used, how many bytes it wrote. Every one of those is a fact
about the process, not an opinion about the output.

I/O practices (deliberate):
  * Stream *content* never transits the shell argv, and is never truncated.
  * One process orchestrates each cell via an argv list (no shell, no
    injection); WRAP is shlex-split so containment prefixes compose cleanly.
  * A failing cell never aborts the run.

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

RSS_RE = re.compile(r"Maximum resident set size.*?:\s*(\d+)")


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
    except Exception as e:
        # Recorded, not blanked. An empty version string reads as "old binary"
        # or "field unused"; this says the probe itself failed and why.
        ver = f"<--version failed: {type(e).__name__}: {e}>"
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

            peak_rss_kb = ""
            if time_bin and timef.exists():
                m = RSS_RE.search(timef.read_text("utf-8", "replace"))
                if m:
                    peak_rss_kb = int(m.group(1))
                timef.unlink()  # RSS captured; keep only .out/.err for artifacts

            # An INDEX into the captures, not a description of them. `stdout`
            # and `stderr` name the files holding the complete output; nothing
            # here is derived from their contents.
            row = {
                "level": level, "id": cid, "cat": cat, "lql": lql,
                "exit_code": rc, "duration_ms": dur_ms,
                "peak_rss_kb": peak_rss_kb,
                "stdout": str(outf.relative_to(out_path.parent)),
                "stderr": str(errf.relative_to(out_path.parent)),
                "stdout_bytes": outf.stat().st_size,
                "stderr_bytes": errf.stat().st_size,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(f"[{level}] {cid} -> exit={rc} {dur_ms}ms "
                  f"rss={peak_rss_kb} out={row['stdout_bytes']}B "
                  f"err={row['stderr_bytes']}B", file=sys.stderr)
            n += 1

    print(f"wrote {n} rows + provenance to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
