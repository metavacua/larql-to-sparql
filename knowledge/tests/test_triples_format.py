"""Validate all data/triples/*.json files have required fields and correct format."""

import json
from pathlib import Path

import pytest


DATA_DIR = Path(__file__).parent.parent / "data" / "triples"


def _triple_files() -> list[Path]:
    """Collect all triple JSON files."""
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.json"))


@pytest.fixture(params=_triple_files(), ids=lambda p: p.name)
def triple_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_triple_files_exist():
    """At least some triple files should exist."""
    assert len(_triple_files()) > 0, "No triple files found in data/triples/"


def test_triple_file_is_valid_json(triple_file: Path):
    """Each triple file must be valid JSON."""
    with open(triple_file) as f:
        data = json.load(f)
    assert isinstance(data, dict)


def test_triple_file_has_relation(triple_file: Path):
    """Each triple file must have a 'relation' field."""
    with open(triple_file) as f:
        data = json.load(f)
    assert "relation" in data, f"Missing 'relation' in {triple_file.name}"
    assert isinstance(data["relation"], str)
    assert len(data["relation"]) > 0


def test_triple_file_has_pairs(triple_file: Path):
    """Each triple file must have a non-empty 'pairs' list."""
    with open(triple_file) as f:
        data = json.load(f)
    assert "pairs" in data, f"Missing 'pairs' in {triple_file.name}"
    assert isinstance(data["pairs"], list)
    assert len(data["pairs"]) > 0, f"Empty pairs in {triple_file.name}"


def test_triple_pairs_are_string_tuples(triple_file: Path):
    """Each pair must be a [str, str] list."""
    with open(triple_file) as f:
        data = json.load(f)
    for i, pair in enumerate(data.get("pairs", [])):
        assert isinstance(pair, list), f"pair[{i}] not a list in {triple_file.name}"
        assert len(pair) == 2, f"pair[{i}] length != 2 in {triple_file.name}"
        assert isinstance(pair[0], str), f"pair[{i}][0] not str in {triple_file.name}"
        assert isinstance(pair[1], str), f"pair[{i}][1] not str in {triple_file.name}"
        assert len(pair[0].strip()) > 0, f"pair[{i}][0] empty in {triple_file.name}"
        assert len(pair[1].strip()) > 0, f"pair[{i}][1] empty in {triple_file.name}"
