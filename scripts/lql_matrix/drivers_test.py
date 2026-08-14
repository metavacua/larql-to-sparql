import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(__file__))
import drivers as D

CELL_LQL = {"id": "c", "cat": "x", "lql": ['USE "v";', "STATS;"]}
CELL_CLI = {"id": "c", "cat": "x", "argv": ["verify", "{{VINDEX}}"]}


def test_lql_driver_joins_with_spaces_into_one_shot_batch():
    argv, stdin = D.build("lql", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "lql", 'USE "v"; STATS;']
    assert stdin is None


def test_repl_pipe_sends_one_statement_per_line():
    argv, stdin = D.build("repl-pipe", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "repl"]
    assert stdin == b'USE "v";\nSTATS;\n'


def test_repl_pty_appends_exit_because_a_terminal_has_no_eof():
    argv, stdin = D.build("repl-pty", CELL_LQL, "/bin/larql")
    assert argv == ["/bin/larql", "repl"]
    assert stdin == b'USE "v";\nSTATS;\nexit\n'


def test_cli_driver_uses_argv_verbatim():
    argv, stdin = D.build("cli", CELL_CLI, "/bin/larql")
    assert argv == ["/bin/larql", "verify", "{{VINDEX}}"]
    assert stdin is None


def test_unknown_driver_raises():
    with pytest.raises(ValueError):
        D.build("nope", CELL_LQL, "/bin/larql")


# A driver must never fabricate a missing field: a fabricated invocation would
# be captured and read as a real result. Four acceptances above, four
# rejections here — the whole driver x cell-shape matrix.
@pytest.mark.parametrize("driver", ["lql", "repl-pipe", "repl-pty"])
def test_cli_cell_under_an_lql_driver_raises(driver):
    with pytest.raises(KeyError):
        D.build(driver, CELL_CLI, "/bin/larql")


def test_lql_cell_under_cli_driver_raises():
    with pytest.raises(KeyError):
        D.build("cli", CELL_LQL, "/bin/larql")


def test_a_string_lql_cell_is_rejected_not_silently_iterated():
    # A str is iterable, so an unmigrated cell would splice into single
    # characters and produce a screenful of nonsense statements rather than
    # fail. This is the failure mode that looks like a product defect.
    with pytest.raises(TypeError):
        D.build("repl-pipe", {"id": "c", "cat": "x", "lql": 'USE "v"; STATS;'},
                "/bin/larql")


def test_the_shipped_corpus_builds_under_every_lql_driver():
    # Not a shape assertion — this runs the real corpus through the real
    # builder, so a cell that cannot be turned into an invocation fails here
    # rather than in CI.
    import json
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "commands.jsonl"), encoding="utf-8") as f:
        cells = [json.loads(line) for line in f if line.strip()]
    # >= not ==: this test's subject is the BUILDER, and pinning the exact
    # corpus size makes every future cell addition fail it for no reason.
    assert len(cells) >= 60
    for cell in cells:
        for driver in ("lql", "repl-pipe", "repl-pty"):
            argv, stdin = D.build(driver, cell, "/bin/larql")
            assert argv[0] == "/bin/larql"
            if driver == "lql":
                assert stdin is None
            else:
                assert stdin.endswith(b"\n")
