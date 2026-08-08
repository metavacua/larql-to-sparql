#!/usr/bin/env python3
"""Build the argv and stdin for one cell under one driver.

Four drivers over the same corpora, because a driver is a distinct user
surface and none covers another:

  lql        larql lql "<statements>"   one-shot batch
  repl-pipe  larql repl, statements on stdin, non-tty
  repl-pty   larql repl under a pseudo-terminal
  cli        the subcommand invoked directly

A cell's `lql` is a LIST of statements. There is no splitter here and there
must not be one: LQL's grammar lives in crates/larql-lql, a second
implementation of it in Python would drift from the first, and statement
boundaries are authored data anyway. `larql_lql::parse` rejects a
multi-statement line outright (`unexpected trailing token`), which is why the
REPL drivers send one statement per line rather than the whole cell — sending
the cell whole would turn every multi-statement cell into one parse error.

The `lql` driver joins with a space, which reproduces the pre-migration
single-string cell byte for byte; that equivalence is asserted against larql's
own splitter in crates/larql-lql/tests/matrix_corpus_wellformed.rs, so all
three LQL drivers demonstrably run the same statement sequence.

repl-pty appends `exit` because a terminal has no EOF: without it the session
would run to the cell timeout every time.

This module knows nothing about corpora, sequencing, or capture. It maps a
cell to an invocation and stops.
"""

DRIVERS = ("lql", "repl-pipe", "repl-pty", "cli")


def _statements(cell):
    stmts = cell["lql"]
    if isinstance(stmts, str):
        raise TypeError(
            f"cell {cell.get('id')!r}: `lql` is a str, expected a list of "
            "statements. A str is iterable, so accepting one here would "
            "splice the cell into single characters instead of failing.")
    return stmts


def build(driver, cell, larql):
    """Return (argv, stdin_bytes). stdin_bytes is None when nothing is written.

    Raises ValueError on an unknown driver, KeyError when the cell lacks the
    field the driver needs, and TypeError on an unmigrated string cell. It
    never fabricates an invocation: a fabricated one would be captured and
    read as a real result.
    """
    if driver == "lql":
        return [larql, "lql", " ".join(_statements(cell))], None
    if driver in ("repl-pipe", "repl-pty"):
        lines = list(_statements(cell))
        if driver == "repl-pty":
            lines.append("exit")
        return [larql, "repl"], ("\n".join(lines) + "\n").encode("utf-8")
    if driver == "cli":
        return [larql, *cell["argv"]], None
    raise ValueError(f"unknown driver {driver!r}; expected one of {DRIVERS}")
