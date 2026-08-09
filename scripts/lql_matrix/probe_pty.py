#!/usr/bin/env python3
"""Smallest thing that answers: can we give a child a real tty and read it back?

Task 4's run_under_pty is the fiddly part of this plan — pty.fork, a select
loop, and EIO-on-child-exit are each easy to get subtly wrong in a way that
hangs a runner. This proves the loop on a runner before it is embedded in the
harness. Usage: probe_pty.py <out-file> <cmd> [args...]  (stdin is forwarded)

Nothing here judges. It writes whatever the terminal produced to <out-file>,
byte for byte, control sequences and \\r included, and prints the child's exit
status to stderr. Terminal escapes are part of what happened; stripping them
would be post-processing.
"""
import errno
import os
import pty
import select
import signal
import sys
import time


def write_all(fd, payload, _write=os.write):
    """Write every byte of `payload` to `fd`, looping over short writes.

    os.write may return SHORT on a blocking fd, so a single call can silently
    truncate — and truncated input reads as larql having ignored statements
    rather than as the harness having dropped them. Harmless at the ~30 bytes
    this probe sends; not harmless for Task 4's 9-statement cells, and this is
    the code Task 4 lifts.

    `_write` is injectable because the end-to-end case cannot be tested: a
    payload large enough to force a short write to a pty master DEADLOCKS
    against a child that echoes — the master's output buffer fills, the child
    blocks writing and so stops reading, and the parent's write never returns.
    Measured: 40 KB to `cat` hangs. So the loop is verified here against a fd
    whose writes are short by construction, which is the only place the
    behaviour is reachable at all.
    """
    view = memoryview(payload)
    while view:
        n = _write(fd, view)
        if n <= 0:  # no progress possible; refuse to spin forever
            raise OSError(f"write to fd {fd} returned {n}")
        view = view[n:]


def main():
    out, argv = sys.argv[1], sys.argv[2:]
    payload = sys.stdin.buffer.read()
    pid, fd = pty.fork()
    if pid == 0:
        try:
            os.execvp(argv[0], argv)
        except Exception:
            # 127 is the shell's "command not found"; the parent records it
            # rather than the child dying silently and reading as a clean exit.
            os._exit(127)
    if payload:
        write_all(fd, payload)
    # Overridable so the timeout branch is testable in under a second. It is
    # the branch that hangs a runner if it is wrong, so it is the one that most
    # needs exercising, and a hardcoded 60 makes exercising it cost 60s a go.
    deadline = time.monotonic() + float(os.environ.get("PROBE_PTY_TIMEOUT", "60"))
    reaped = None  # exit status, if the WNOHANG poll below collects it first
    with open(out, "wb") as f:
        while True:
            if time.monotonic() > deadline:
                # killpg, not kill. pty.fork() calls setsid(), so the child is
                # a session AND process-group leader with pgid == pid and its
                # descendants inherit that group.
                #
                # Plain kill(pid) is USUALLY enough and not because it is
                # sufficient: killing the session leader makes the kernel send
                # SIGHUP to the tty's foreground group, which collects the
                # descendants for free. Measured, both forms leave zero strays
                # for `bash -c 'sleep N; echo x'`. But SIGHUP is ignorable, and
                # for a child that ignores it the two diverge — measured, a
                # nohup'd sleep survives kill(pid) and does not survive
                # killpg(pid). `serve`, `chat` and `repl` are all expected to
                # hit this timeout, so the leak would be per-timeout and the
                # stronger form costs nothing.
                try:
                    os.killpg(pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                print("TIMEOUT", file=sys.stderr)
                break
            r, _, _ = select.select([fd], [], [], 1.0)
            if not r:
                # A terminal has no EOF, so a quiet fd does not mean the child
                # is done — poll for the child instead of blocking forever.
                # KEEP the status: this branch is the only one that reaps, and
                # discarding it here made the final waitpid raise
                # ChildProcessError and report `exit=unknown` for a code that
                # had in fact been observed. It fires whenever the child exits
                # without the master fd reaching EIO — a descendant still
                # holding the slave open, which `serve` and `chat` can do.
                wpid, wstatus = os.waitpid(pid, os.WNOHANG)
                if wpid == pid:
                    reaped = wstatus
                    break
                continue
            try:
                chunk = os.read(fd, 65536)
            except OSError as e:
                # The master fd raises EIO when the last slave closes, which is
                # how a pty signals what a pipe would signal with EOF.
                if e.errno == errno.EIO:
                    break
                raise
            if not chunk:
                break
            f.write(chunk)
            f.flush()
    os.close(fd)
    if reaped is not None:
        print(f"exit={os.waitstatus_to_exitcode(reaped)}", file=sys.stderr)
    else:
        try:
            _, status = os.waitpid(pid, 0)
            print(f"exit={os.waitstatus_to_exitcode(status)}", file=sys.stderr)
        except ChildProcessError:
            # Nothing reaped it and it is already gone. Say so rather than
            # printing a number that was never observed.
            print("exit=unknown", file=sys.stderr)


if __name__ == "__main__":
    main()
