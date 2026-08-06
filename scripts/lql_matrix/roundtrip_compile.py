#!/usr/bin/env python3
"""Round-trip compile orchestrator: for one already-produced vindex, run the
existing LQL/CLI compile paths (passthrough + KNN/COMPOSE edits, with/without
MEMIT) and diff every resulting directory against the untouched HF input and
against each other, using the EXISTING structural+value differ
(`roundtrip_matrix.diff_model_dirs`) as-is. Emits raw measured outcomes only —
no interpretation, never crashes the calling job.

Grammar notes (grounded in direct code inspection, do not second-guess):
  * `USE "<path>";` opens a vindex (NOT `USE VINDEX`).
  * `SAVE PATCH;` takes NO filename — it persists to whatever path a prior
    `BEGIN PATCH "<file>";` named. An anonymous (auto-started) patch session
    errors on SAVE PATCH, so every edit job explicitly opens a named session.
  * `COMPILE CURRENT INTO MODEL "<dir>" FORMAT safetensors;` writes larql's
    internal weight files; edits only reach the weights (MEMIT) when
    LARQL_MEMIT_ENABLE=1 AND the INSERT + SAVE PATCH + COMPILE run in the
    SAME `larql lql` batch — so every edit job's statement is one semicolon-
    separated batch passed to a single `larql lql "..."` invocation.
  * `larql compile --base <hf-or-dir> --vindex <patch.vlp> --output <dir>`
    replays a `.vlp` patch file (NOT a vindex dir) onto the base checkpoint.
    An empty (zero-op) `.vlp` makes it print "no patches found" and return
    BEFORE creating the output dir — legitimate, not a failure; recorded via
    `out_dir_exists=False`.
"""
import argparse
import glob
import json
import os
import shutil
import subprocess
import sys
import time

import roundtrip_matrix as M
import run_matrix as RM

INSERT_STMT = {
    "knn": ('INSERT INTO EDGES (entity, relation, target) '
            'VALUES ("roundtrip-e", "roundtrip-r", "roundtrip-t") MODE KNN;'),
    "compose": ('INSERT INTO EDGES (entity, relation, target) '
                'VALUES ("roundtrip-e", "roundtrip-r", "roundtrip-t") MODE COMPOSE;'),
}


def _job_dir(out_root, job_id):
    return os.path.join(out_root, job_id)


def _mk(out_root, vindex_dir, job_id, driver, mode, insert_form, memit):
    d = _job_dir(out_root, job_id)
    return {
        "id": job_id, "driver": driver, "mode": mode,
        "insert_form": insert_form, "memit": memit,
        "vindex_src": vindex_dir,
        "vindex_copy": os.path.join(d, "vindex"),
        "out_dir": os.path.join(d, "out"),
        "lql_stmt": None, "vlp_path": None, "needs_vlp_from": None,
    }


def plan_jobs(name, vindex_dir, out_root):
    """Enumerate the 8 round-trip jobs for one produced vindex. Pure — builds
    paths and LQL statement text only, runs nothing."""
    jobs = []

    # lql.passthrough — USE + COMPILE, no edits (the LQL no-op baseline).
    j = _mk(out_root, vindex_dir, "lql.passthrough", "lql", "passthrough", None, None)
    j["lql_stmt"] = (f'USE "{j["vindex_copy"]}"; '
                      f'COMPILE CURRENT INTO MODEL "{j["out_dir"]}" FORMAT safetensors;')
    jobs.append(j)

    # lql.<form>.memit{off,on} — BEGIN PATCH (named) + INSERT + SAVE PATCH +
    # COMPILE, all in ONE larql-lql batch (MEMIT only bakes into weights
    # within the same process — see module docstring).
    for form in ("knn", "compose"):
        for memit in (False, True):
            job_id = f"lql.{form}.memit{'on' if memit else 'off'}"
            j = _mk(out_root, vindex_dir, job_id, "lql", "edit", form, memit)
            j["vlp_path"] = os.path.join(_job_dir(out_root, job_id), "patch.vlp")
            j["lql_stmt"] = (
                f'USE "{j["vindex_copy"]}"; '
                f'BEGIN PATCH "{j["vlp_path"]}"; '
                f'{INSERT_STMT[form]} '
                f'SAVE PATCH; '
                f'COMPILE CURRENT INTO MODEL "{j["out_dir"]}" FORMAT safetensors;'
            )
            jobs.append(j)

    # cli.passthrough — `larql compile` against a deliberately empty (0-op)
    # .vlp: exercises the documented "no patches found, no output dir" path.
    j = _mk(out_root, vindex_dir, "cli.passthrough", "cli", "passthrough", None, None)
    j["vlp_path"] = os.path.join(_job_dir(out_root, "cli.passthrough"), "empty.vlp")
    jobs.append(j)

    # cli.<form> — `larql compile` against the .vlp SAVE-PATCHed by the
    # matching lql memitoff job (needs_vlp_from resolved in run_jobs, after
    # that job has actually run and written the file).
    for form in ("knn", "compose"):
        job_id = f"cli.{form}"
        j = _mk(out_root, vindex_dir, job_id, "cli", "edit", form, None)
        j["needs_vlp_from"] = f"lql.{form}.memitoff"
        jobs.append(j)

    return jobs


def comparison_plan(jobs):
    """Enumerate every comparison over a job list. Pure — no I/O. Returns ALL
    comparisons (input-vs-job, edit-vs-passthrough same driver, driver-vs-
    driver on matching (mode, insert_form) incl. memit variants, and the
    lql memit-off-vs-on effect per insert form); nothing is dropped."""
    by_id = {j["id"]: j for j in jobs}
    comparisons = []

    def add(comparison, slug_a, slug_b, driver, mode, insert_form, memit):
        comparisons.append({
            "comparison": comparison, "driver": driver, "mode": mode,
            "insert_form": insert_form, "memit": memit,
            "slug_a": slug_a, "slug_b": slug_b,
        })

    # 1. input_vs_<job> — every job vs the untouched HF snapshot.
    for j in jobs:
        add(f"input_vs_{j['id']}", "_input", j["id"],
            j["driver"], j["mode"], j["insert_form"], j["memit"])

    # 2. <Bjob>_vs_<Ajob> — same-driver edit job vs that driver's passthrough.
    passthrough_id = {"lql": "lql.passthrough", "cli": "cli.passthrough"}
    for j in jobs:
        if j["mode"] != "edit":
            continue
        a_id = passthrough_id[j["driver"]]
        add(f"{j['id']}_vs_{a_id}", j["id"], a_id,
            j["driver"], j["mode"], j["insert_form"], j["memit"])

    # 3. driver_vs_driver — every matching (mode, insert_form) pair across
    #    cli/lql, including every lql memit variant (not just one).
    cli_jobs = [j for j in jobs if j["driver"] == "cli"]
    lql_jobs = [j for j in jobs if j["driver"] == "lql"]
    for cj in cli_jobs:
        for lj in lql_jobs:
            if cj["mode"] != lj["mode"] or cj["insert_form"] != lj["insert_form"]:
                continue
            add(f"driver_vs_driver:{cj['id']}_vs_{lj['id']}", cj["id"], lj["id"],
                "cli_vs_lql", lj["mode"], lj["insert_form"], lj["memit"])

    # 4. memit_effect — lql memitoff vs memiton, per insert form.
    for form in ("knn", "compose"):
        off_id, on_id = f"lql.{form}.memitoff", f"lql.{form}.memiton"
        if off_id in by_id and on_id in by_id:
            add(f"memit_effect:{form}", off_id, on_id,
                "lql", "edit", form, "off_vs_on")

    return comparisons


def run_jobs(jobs, larql_bin, base_hf, cell_timeout=900):
    """Execute every planned job: copy the produced vindex into an isolated
    per-job working copy, run the lql batch or cli compile, and record raw
    mechanical outcomes. Impure — the only function here that shells out."""
    by_id = {j["id"]: j for j in jobs}
    copied = set()
    rows = []

    for j in jobs:
        dst = j["vindex_copy"]
        if dst not in copied:
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(j["vindex_src"], dst)
            copied.add(dst)

        env = dict(os.environ)
        if j["memit"] is True:
            env["LARQL_MEMIT_ENABLE"] = "1"
        else:
            env.pop("LARQL_MEMIT_ENABLE", None)

        if j["driver"] == "lql":
            argv = ["timeout", "--kill-after=10", str(cell_timeout),
                    larql_bin, "lql", j["lql_stmt"]]
        else:
            if j["needs_vlp_from"]:
                src = by_id[j["needs_vlp_from"]]
                hits = sorted(glob.glob(
                    os.path.join(os.path.dirname(src["vlp_path"]), "*.vlp")))
                j["vlp_path"] = hits[0] if hits else src["vlp_path"]
            elif j["driver"] == "cli" and j["mode"] == "passthrough":
                os.makedirs(os.path.dirname(j["vlp_path"]), exist_ok=True)
                with open(j["vlp_path"], "w", encoding="utf-8") as f:
                    json.dump({"version": 1, "base_model": base_hf,
                               "base_checksum": None, "created_at": "",
                               "description": None, "author": None,
                               "tags": [], "operations": []}, f)
            argv = ["timeout", "--kill-after=10", str(cell_timeout),
                    larql_bin, "compile", "--base", base_hf,
                    "--vindex", j["vlp_path"], "--output", j["out_dir"]]

        t0 = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True, env=env)
            rc, out_txt, err_txt = proc.returncode, proc.stdout, proc.stderr
        except OSError as e:
            rc, out_txt, err_txt = -1, "", str(e)
        dur_ms = int((time.monotonic() - t0) * 1000)

        combined = out_txt + "\n" + err_txt
        err_signal = 1 if RM.ERR_RE.search(combined) else 0
        out_dir = j["out_dir"]
        out_exists = os.path.isdir(out_dir)
        n_files = (len([f for f in os.listdir(out_dir)
                        if os.path.isfile(os.path.join(out_dir, f))])
                   if out_exists else 0)

        rows.append({
            "id": j["id"], "driver": j["driver"], "mode": j["mode"],
            "insert_form": j["insert_form"], "memit": j["memit"],
            "exit_code": rc, "bucket": RM.bucket_for(rc),
            "err_signal": err_signal,
            "err_line": RM.first_error_line(combined) if err_signal else "",
            "duration_ms": dur_ms,
            "out_dir_exists": out_exists, "out_dir_n_files": n_files,
        })

    return rows


def run_one_comparison(dir_a, dir_b, meta):
    """Diff two model directories via the EXISTING differ, degrading to a
    measured (not interpreted) row when a dir is missing or the differ itself
    fails on malformed output — never raises."""
    a_exists, b_exists = os.path.isdir(dir_a), os.path.isdir(dir_b)
    if not a_exists or not b_exists:
        return [{**meta, "file": "_manifest", "bijective": False,
                 "dir_a_exists": a_exists, "dir_b_exists": b_exists}]
    try:
        return M.to_rows(meta, M.diff_model_dirs(dir_a, dir_b))
    except (ValueError, OSError) as e:
        return [{**meta, "file": "_error", "error": str(e)}]


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True)
    ap.add_argument("--hf", required=True)
    ap.add_argument("--vindex", required=True)
    ap.add_argument("--larql-bin", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--produce-out", required=True)
    ap.add_argument("--diff-out", required=True)
    ap.add_argument("--cell-timeout", type=int, default=900)
    ns = ap.parse_args(argv)

    try:
        os.makedirs(ns.out_root, exist_ok=True)
        jobs = plan_jobs(ns.name, ns.vindex, ns.out_root)
        produce_rows = run_jobs(jobs, ns.larql_bin, ns.hf, ns.cell_timeout)
        with open(ns.produce_out, "w", encoding="utf-8") as f:
            for r in produce_rows:
                f.write(json.dumps(r) + "\n")

        from huggingface_hub import snapshot_download
        base_dir = snapshot_download(ns.hf)

        by_id = {j["id"]: j for j in jobs}
        comp_rows = []
        for c in comparison_plan(jobs):
            dir_a = base_dir if c["slug_a"] == "_input" else by_id[c["slug_a"]]["out_dir"]
            dir_b = base_dir if c["slug_b"] == "_input" else by_id[c["slug_b"]]["out_dir"]
            comp_rows.extend(run_one_comparison(dir_a, dir_b, c))

        with open(ns.diff_out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"leg": ns.name, "hf": ns.hf,
                                 "produce": produce_rows,
                                 "comparisons": comp_rows}) + "\n")
    except Exception as e:  # noqa: BLE001 — never crash the calling CI job
        with open(ns.diff_out, "w", encoding="utf-8") as f:
            f.write(json.dumps({"leg": ns.name, "error": str(e)}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
