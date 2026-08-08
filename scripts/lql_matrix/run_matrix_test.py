import json
import os
import stat
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))

FAKE_LARQL = """#!/usr/bin/env bash
# Fake larql: echoes its argv and any stdin, so the harness can be tested
# without building the real binary.
echo "ARGV: $*"
if [ ! -t 0 ]; then while IFS= read -r l; do echo "STDIN: $l"; done; fi
echo "to stderr" >&2
exit 3
"""


def _fake_bin(tmp_path):
    p = tmp_path / "larql"
    p.write_text(FAKE_LARQL, encoding="utf-8")
    p.chmod(p.stat().st_mode | stat.S_IEXEC)
    return str(p)


def _corpus(tmp_path):
    p = tmp_path / "corpus.jsonl"
    p.write_text(json.dumps({"id": "c1", "cat": "x",
                             "lql": ['USE "{{VINDEX}}";', "STATS;"]}) + "\n",
                 encoding="utf-8")
    return str(p)


def _run(tmp_path, driver):
    out = tmp_path / "results.jsonl"
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path), CELL_TIMEOUT="30")
    argv = [sys.executable, os.path.join(HERE, "run_matrix.py"),
            "leg1", "/nonexistent.vindex", _corpus(tmp_path), str(out)]
    if driver:
        argv += ["--driver", driver]
    subprocess.run(argv, env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    return [r for r in rows if r.get("type") != "meta"], tmp_path


def test_default_driver_is_lql_and_row_records_it(tmp_path):
    rows, _ = _run(tmp_path, None)
    assert len(rows) == 1
    assert rows[0]["driver"] == "lql"
    assert rows[0]["exit_code"] == 3


def test_capture_files_are_named_by_driver_and_contain_full_output(tmp_path):
    rows, tp = _run(tmp_path, "lql")
    out_path = tp / rows[0]["stdout"]
    assert out_path.exists()
    text = out_path.read_text()
    assert "ARGV: lql" in text
    assert (tp / rows[0]["stderr"]).read_text().strip() == "to stderr"


def test_row_carries_no_derived_opinion(tmp_path):
    rows, _ = _run(tmp_path, "lql")
    forbidden = {"bucket", "err_signal", "err_line", "stdout_head",
                 "stderr_head", "stderr_tail", "status", "ok", "passed"}
    assert forbidden.isdisjoint(rows[0].keys())


def test_repl_pipe_writes_statements_to_stdin(tmp_path):
    rows, tp = _run(tmp_path, "repl-pipe")
    text = (tp / rows[0]["stdout"]).read_text()
    assert "ARGV: repl" in text
    assert 'STDIN: USE "/nonexistent.vindex";' in text
    assert "STDIN: STATS;" in text


def test_substitution_reaches_every_statement_not_just_the_first(tmp_path):
    # {{TMP}} in a later statement must be substituted too. Statements are a
    # list now, so a subst() applied to the joined string only — or to
    # entry [0] — would leave a literal placeholder in the sent command.
    out = tmp_path / "r.jsonl"
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(json.dumps({"id": "c2", "cat": "x",
                                  "lql": ['USE "{{VINDEX}}";',
                                          'BEGIN PATCH "{{TMP}}/p.vlp";']}) + "\n",
                      encoding="utf-8")
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path), CELL_TIMEOUT="30",
               TMPROOT=str(tmp_path / "tmproot"))
    subprocess.run([sys.executable, os.path.join(HERE, "run_matrix.py"),
                    "leg1", "/v.vindex", str(corpus), str(out),
                    "--driver", "repl-pipe"],
                   env=env, check=True, capture_output=True)
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    row = [r for r in rows if r.get("type") != "meta"][0]
    text = (tmp_path / row["stdout"]).read_text()
    assert "{{TMP}}" not in text, "a placeholder survived into the sent command"
    assert "{{VINDEX}}" not in text
    assert "/p.vlp" in text


def test_unknown_driver_is_rejected(tmp_path):
    out = tmp_path / "r.jsonl"
    env = dict(os.environ, LARQL_BIN=_fake_bin(tmp_path))
    r = subprocess.run([sys.executable, os.path.join(HERE, "run_matrix.py"),
                        "leg1", "/v", _corpus(tmp_path), str(out), "--driver", "nope"],
                       env=env, capture_output=True, text=True)
    assert r.returncode != 0


def test_a_driver_that_writes_no_stdin_gets_stdin_closed(tmp_path):
    # Regression. subprocess.run(input=None) INHERITS the parent's stdin, so a
    # cell that reads stdin blocks on whatever the harness was invoked with —
    # cell behaviour would depend on the caller, not on larql. Found by
    # running the real corpus through the shim: it hung. `larql repl` is
    # measurably sensitive to this (Task 0: exit 0 at once on closed stdin).
    reader = tmp_path / "larql"
    reader.write_text(
        "#!/usr/bin/env bash\n"
        # --version must not read stdin, or it blocks the provenance probe and
        # the test stops discriminating between cell behaviours.
        'if [ "$1" = "--version" ]; then echo "fake 0.0"; exit 0; fi\n'
        "if [ -t 0 ]; then echo TTY; else cat; echo EOF_REACHED; fi\n",
        encoding="utf-8")
    reader.chmod(reader.stat().st_mode | stat.S_IEXEC)

    out = tmp_path / "r.jsonl"
    env = dict(os.environ, LARQL_BIN=str(reader), CELL_TIMEOUT="15")

    # stdin must stay OPEN for this to discriminate. subprocess.run(stdin=PIPE,
    # input=None) closes the pipe immediately, so the child sees EOF and the
    # test passes with or without the fix — the first version of this test did
    # exactly that and proved nothing. Hold the write end open ourselves.
    proc = subprocess.Popen([sys.executable, os.path.join(HERE, "run_matrix.py"),
                             "leg1", "/v", _corpus(tmp_path), str(out),
                             "--driver", "lql"],
                            env=env, stdin=subprocess.PIPE,
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        # Without the fix the cell inherits this still-open pipe, blocks in
        # `cat`, and only dies at CELL_TIMEOUT=15s.
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise AssertionError(
            "the cell blocked on inherited stdin: a driver that writes no "
            "stdin must get subprocess.DEVNULL, not the harness's own stdin")
    finally:
        if proc.stdin:
            proc.stdin.close()

    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    row = [x for x in rows if x.get("type") != "meta"][0]
    assert (tmp_path / row["stdout"]).read_text().strip() == "EOF_REACHED"
