"""Acceptance and rejection criteria for the pty read loop.

Task 4 lifts this loop into run_matrix.run_under_pty, so its failure modes are
worth pinning before anything depends on them. These were originally run by
hand and not committed, which a review correctly flagged: a knob added to make
the timeout branch testable (PROBE_PTY_TIMEOUT) with nothing testing it.

The timeout branch is the one that hangs a runner, so it gets the most
attention here.
"""
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PROBE = os.path.join(HERE, "probe_pty.py")


def _run(tmp_path, argv, payload=b"", timeout_s=None, wait=60):
    out = tmp_path / "capture.out"
    env = dict(os.environ)
    if timeout_s is not None:
        env["PROBE_PTY_TIMEOUT"] = str(timeout_s)
    p = subprocess.run([sys.executable, PROBE, str(out), *argv],
                       input=payload, capture_output=True, env=env,
                       timeout=wait)
    return p.stderr.decode(), (out.read_bytes() if out.exists() else b"")


def test_child_sees_a_tty_and_reads_piped_input(tmp_path):
    err, cap = _run(tmp_path, [
        "bash", "-c",
        'if [ -t 0 ]; then echo TTY_YES; else echo TTY_NO; fi; '
        'read -r l; echo "GOT:$l"'], payload=b"hello\n")
    assert b"TTY_YES" in cap
    assert b"GOT:hello" in cap
    assert "exit=0" in err


def test_carriage_returns_are_kept_verbatim(tmp_path):
    # A terminal turns \n into \r\n. Stripping that would be post-processing,
    # and what the terminal actually emitted is part of what happened.
    _, cap = _run(tmp_path, ["bash", "-c", "echo hi"])
    assert b"\r\n" in cap


def test_missing_binary_reports_127_not_a_clean_exit(tmp_path):
    err, _ = _run(tmp_path, ["no-such-binary-xyz-12345"])
    assert "exit=127" in err


def test_nonzero_exit_is_reported(tmp_path):
    err, cap = _run(tmp_path, ["bash", "-c", "echo OUT; exit 42"])
    assert "exit=42" in err
    assert b"OUT" in cap


def test_output_larger_than_one_read_is_not_truncated(tmp_path):
    n = 300000
    _, cap = _run(tmp_path, ["bash", "-c",
                             f'head -c {n} /dev/zero | tr "\\0" "x"'], wait=120)
    assert cap.count(b"x") == n


def test_timeout_kills_the_process_group_and_keeps_partial_output(tmp_path):
    # nohup'd child ignores SIGHUP, which is what distinguishes killpg from
    # kill: killing the session leader makes the kernel SIGHUP the foreground
    # group, so an ordinary child dies either way.
    t0 = time.monotonic()
    err, cap = _run(tmp_path, ["bash", "-c",
                               "echo BEFORE_HANG; nohup sleep 240 >/dev/null 2>&1 & wait"],
                    timeout_s=2, wait=60)
    assert "TIMEOUT" in err
    assert b"BEFORE_HANG" in cap, "partial output must survive a timeout"
    assert time.monotonic() - t0 < 30, "deadline was not honoured"
    leaked = subprocess.run(["pgrep", "-x", "sleep"], capture_output=True)
    try:
        assert leaked.returncode != 0, (
            "a SIGHUP-ignoring descendant outlived the timeout: the loop must "
            "killpg, not kill")
    finally:
        subprocess.run(["pkill", "-x", "sleep"], capture_output=True)


def test_exit_status_survives_a_descendant_holding_the_terminal(tmp_path):
    # The WNOHANG poll is the only path that reaps, and it used to DISCARD the
    # status it had just collected, so the final waitpid raised
    # ChildProcessError and printed `exit=unknown` for a code that had in fact
    # been observed. That is fixed.
    #
    # NO TEST HERE DISCRIMINATES, and this one does not claim to. A review
    # predicted the poll would win "whenever the child exits without the master
    # reaching EIO — a descendant still holding the slave open". Measured on
    # Linux with an instrumented copy, that state is unreachable: a plain
    # background child, one with the pty fds explicitly re-bound, and a
    # setsid'd one all reach EIO, because the session leader's exit
    # disassociates the controlling terminal and the master read returns EIO
    # regardless of who else holds the slave. The poll is a safety net for a
    # race, not a routine path.
    #
    # So what this asserts is the OUTCOME — a real exit code, never
    # `exit=unknown` — which holds whichever branch wins. Writing a test that
    # passed with the fix reverted, and calling it a regression test, is the
    # mistake that already had to be corrected once in run_matrix_test.py.
    err, cap = _run(tmp_path, ["bash", "-c",
                               "sleep 30 & echo STARTED; exit 7"], timeout_s=25)
    try:
        assert b"STARTED" in cap
        assert "exit=unknown" not in err, "an observed exit status was discarded"
        assert "exit=7" in err, f"expected exit=7, got {err!r}"
    finally:
        subprocess.run(["pkill", "-x", "sleep"], capture_output=True)
