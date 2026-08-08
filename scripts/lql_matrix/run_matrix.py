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
                       [--driver lql|repl-pipe|repl-pty|cli]   (default lql)
Env:     LARQL_BIN (default target/release/larql), MODEL_ID, TMPROOT,
         WRAP (e.g. "larql-probe safe --mem 5600 --"), CELL_TIMEOUT (default 900),
         plus GITHUB_SHA / RUNNER_OS / RUST_BACKTRACE recorded as provenance.
"""
import argparse
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

# Invoked by path from the workflow, so its own directory is not necessarily
# on sys.path. Insert it before importing siblings.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import drivers  # noqa: E402  (needs the sys.path line above)

RSS_RE = re.compile(r"Maximum resident set size.*?:\s*(\d+)")


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Run a corpus against one vindex and capture everything.")
    ap.add_argument("level")
    ap.add_argument("vindex")
    ap.add_argument("corpus")
    ap.add_argument("out")
    ap.add_argument("--driver", default="lql", choices=drivers.DRIVERS)
    ns = ap.parse_args()
    level, vindex, corpus, out, driver = (
        ns.level, ns.vindex, ns.corpus, ns.out, ns.driver)
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
        # stdin=DEVNULL here too: capture_output redirects stdout/stderr but
        # NOT stdin, so a binary that reads stdin would block this probe on
        # the harness's own stdin for the full 30s timeout.
        ver = subprocess.run([larql, "--version"], capture_output=True,
                             stdin=subprocess.DEVNULL,
                             text=True, timeout=30).stdout.strip()
    except Exception as e:
        # Recorded, not blanked. An empty version string reads as "old binary"
        # or "field unused"; this says the probe itself failed and why.
        ver = f"<--version failed: {type(e).__name__}: {e}>"
    meta = {
        "type": "meta", "level": level, "driver": driver, "larql_version": ver,
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
            cid, cat = c["id"], c.get("cat", "")

            # Substitute EVERY statement, not the joined string: a placeholder
            # in a later statement is just as load-bearing as one in the first,
            # and the drivers each assemble the statements differently.
            cell = dict(c)
            if "lql" in cell:
                if isinstance(cell["lql"], str):
                    raise TypeError(
                        f"cell {cid!r}: `lql` is a str, expected a list of "
                        "statements — a str is iterable, so accepting one "
                        "would splice the cell into single characters.")
                cell["lql"] = [subst(stmt) for stmt in cell["lql"]]
            if "argv" in cell:
                cell["argv"] = [subst(a) for a in cell["argv"]]

            # The driver owns how a cell becomes an invocation. This loop owns
            # timing, capture and the row, and knows nothing about LQL.
            argv_cmd, stdin_bytes = drivers.build(driver, cell, larql)

            # The driver is in the filename: the same cell under two drivers
            # would otherwise overwrite one capture with the other.
            outf = cells_dir / f"{level}.{driver}.{cid}.out"
            errf = cells_dir / f"{level}.{driver}.{cid}.err"
            timef = cells_dir / f"{level}.{driver}.{cid}.time"

            argv = ["timeout", "--kill-after=10", cell_timeout, *wrap]
            if time_bin:
                argv += [time_bin, "-v", "-o", str(timef)]
            argv += argv_cmd

            t0 = time.monotonic()
            with outf.open("wb") as so, errf.open("wb") as se:
                # When a driver writes no stdin, close it explicitly rather
                # than letting the child inherit the harness's. Inherited
                # stdin makes a cell's behaviour depend on how the runner was
                # invoked — a command that reads stdin blocks until the cell
                # timeout under one caller and returns instantly under
                # another. `larql repl` is measurably one of those: Task 0
                # recorded it exiting 0 at once on closed stdin.
                if stdin_bytes is None:
                    rc = subprocess.run(argv, stdout=so, stderr=se,
                                        stdin=subprocess.DEVNULL).returncode
                else:
                    rc = subprocess.run(argv, stdout=so, stderr=se,
                                        input=stdin_bytes).returncode
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
                "level": level, "driver": driver, "id": cid, "cat": cat,
                "sent": cell.get("lql") or cell.get("argv"),
                "exit_code": rc, "duration_ms": dur_ms,
                "peak_rss_kb": peak_rss_kb,
                "stdout": str(outf.relative_to(out_path.parent)),
                "stderr": str(errf.relative_to(out_path.parent)),
                "stdout_bytes": outf.stat().st_size,
                "stderr_bytes": errf.stat().st_size,
            }
            with out_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")

            print(f"[{level}/{driver}] {cid} -> exit={rc} {dur_ms}ms "
                  f"rss={peak_rss_kb} out={row['stdout_bytes']}B "
                  f"err={row['stderr_bytes']}B", file=sys.stderr)
            n += 1

    print(f"wrote {n} rows + provenance to {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
