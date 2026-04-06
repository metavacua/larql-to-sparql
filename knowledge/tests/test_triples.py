"""Tests for triple loading and assembly."""

import json
import tempfile
from pathlib import Path

from larql_knowledge.triples import (
    load_triple_file,
    load_all_triples,
    assemble,
    stats,
    merge_triples,
)


def _make_triple_file(dir_path: Path, relation: str, pairs: list) -> Path:
    path = dir_path / f"{relation}.json"
    with open(path, "w") as f:
        json.dump({"relation": relation, "pairs": pairs}, f)
    return path


def test_load_triple_file():
    with tempfile.TemporaryDirectory() as td:
        path = _make_triple_file(Path(td), "capital", [["France", "Paris"]])
        data = load_triple_file(path)
        assert data["relation"] == "capital"
        assert len(data["pairs"]) == 1


def test_load_all_triples():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        _make_triple_file(td, "capital", [["France", "Paris"]])
        _make_triple_file(td, "language", [["France", "French"]])
        combined = load_all_triples(td)
        assert "capital" in combined
        assert "language" in combined
        assert len(combined) == 2


def test_assemble():
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        triples_dir = td / "triples"
        triples_dir.mkdir()
        _make_triple_file(triples_dir, "capital", [["France", "Paris"], ["Germany", "Berlin"]])
        _make_triple_file(triples_dir, "language", [["France", "French"]])

        output = td / "combined.json"
        combined = assemble(triples_dir, output)

        assert output.exists()
        assert len(combined) == 2

        with open(output) as f:
            loaded = json.load(f)
        assert "capital" in loaded
        assert len(loaded["capital"]["pairs"]) == 2


def test_stats():
    triples = {
        "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]},
        "language": {"pairs": [["France", "French"]]},
    }
    s = stats(triples)
    assert s["num_relations"] == 2
    assert s["total_pairs"] == 3
    assert s["relations"]["capital"] == 2


def test_merge_triples():
    target = {
        "capital": {"pairs": [["France", "Paris"]]},
    }
    source = {
        "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]},
        "language": {"pairs": [["France", "French"]]},
    }
    added = merge_triples(target, source)
    assert added == 2  # Berlin + French
    assert len(target["capital"]["pairs"]) == 2
    assert "language" in target


def test_merge_no_duplicates():
    target = {"capital": {"pairs": [["France", "Paris"]]}}
    source = {"capital": {"pairs": [["France", "Paris"]]}}
    added = merge_triples(target, source)
    assert added == 0
    assert len(target["capital"]["pairs"]) == 1
