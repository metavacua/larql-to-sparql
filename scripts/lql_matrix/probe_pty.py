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
        os.write(fd, payload)
    # Overridable so the timeout branch is testable in under a second. It is
    # the branch that hangs a runner if it is wrong, so it is the one that most
    # needs exercising, and a hardcoded 60 makes exercising it cost 60s a go.
    deadline = time.monotonic() + float(os.environ.get("PROBE_PTY_TIMEOUT", "60"))
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
                if os.waitpid(pid, os.WNOHANG)[0] == pid:
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
    try:
        _, status = os.waitpid(pid, 0)
        print(f"exit={os.waitstatus_to_exitcode(status)}", file=sys.stderr)
    except ChildProcessError:
        # Already reaped by the WNOHANG poll above. Say so rather than
        # printing a number that was never observed.
        print("exit=unknown", file=sys.stderr)


if __name__ == "__main__":
    main()
