"""Validate the morphological_relations.json data file."""

import json
from pathlib import Path


DATA_FILE = Path(__file__).parent.parent / "data" / "morphological_relations.json"

EXPECTED_TYPES = ["plural", "gerund", "past_tense", "comparative", "superlative"]


def _load():
    with open(DATA_FILE) as f:
        return json.load(f)


def test_morphological_output_exists():
    assert DATA_FILE.exists(), f"{DATA_FILE} not found"


def test_morphological_has_relations():
    data = _load()
    assert "relations" in data
    assert isinstance(data["relations"], dict)


def test_morphological_has_expected_types():
    rels = _load()["relations"]
    for t in EXPECTED_TYPES:
        assert t in rels, f"Missing expected relation type: {t}"


def test_morphological_pairs_format():
    rels = _load()["relations"]
    for name, rel in rels.items():
        assert "pairs" in rel, f"Missing 'pairs' in relation {name}"
        for i, pair in enumerate(rel["pairs"]):
            assert isinstance(pair, list), f"pair[{i}] not list in {name}"
            assert len(pair) == 2, f"pair[{i}] length != 2 in {name}"
            assert isinstance(pair[0], str), f"pair[{i}][0] not str in {name}"
            assert isinstance(pair[1], str), f"pair[{i}][1] not str in {name}"


def test_morphological_pairs_nonempty():
    rels = _load()["relations"]
    for name, rel in rels.items():
        assert len(rel["pairs"]) >= 10, (
            f"Relation {name} has only {len(rel['pairs'])} pairs (expected >= 10)"
        )


def test_morphological_description():
    rels = _load()["relations"]
    for name, rel in rels.items():
        assert "description" in rel, f"Missing 'description' in relation {name}"
        assert isinstance(rel["description"], str)
        assert len(rel["description"]) > 0
