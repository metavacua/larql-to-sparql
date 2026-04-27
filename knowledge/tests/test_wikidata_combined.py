"""Validate the combined wikidata_triples.json data file."""

import json
from pathlib import Path


DATA_FILE = Path(__file__).parent.parent / "data" / "wikidata_triples.json"

CORE_RELATIONS = ["capital", "language", "continent", "occupation", "birthplace"]


def _load():
    with open(DATA_FILE) as f:
        return json.load(f)


def test_combined_exists():
    assert DATA_FILE.exists(), f"{DATA_FILE} not found"


def test_combined_is_dict():
    data = _load()
    assert isinstance(data, dict)


def test_combined_min_relations():
    data = _load()
    assert len(data) >= 100, f"Only {len(data)} relations (expected >= 100)"


def test_combined_min_pairs():
    data = _load()
    total = 0
    for rel in data.values():
        pairs = rel.get("pairs", rel) if isinstance(rel, dict) else rel
        total += len(pairs)
    assert total > 15000, f"Only {total} total pairs (expected > 15000)"


def test_combined_pairs_format():
    data = _load()
    for name, rel in data.items():
        assert isinstance(rel, dict), f"{name}: value is not a dict"
        assert "pairs" in rel, f"{name}: missing 'pairs' key"
        for i, pair in enumerate(rel["pairs"]):
            assert isinstance(pair, list), f"{name} pair[{i}] not a list"
            assert len(pair) == 2, f"{name} pair[{i}] length != 2"
            assert isinstance(pair[0], str), f"{name} pair[{i}][0] not str"
            assert isinstance(pair[1], str), f"{name} pair[{i}][1] not str"


def test_combined_core_relations_present():
    data = _load()
    for rel in CORE_RELATIONS:
        assert rel in data, f"Missing core relation: {rel}"


def test_combined_no_empty_relations():
    data = _load()
    for name, rel in data.items():
        pairs = rel.get("pairs", []) if isinstance(rel, dict) else rel
        assert len(pairs) > 0, f"Relation {name} has 0 pairs"


def test_combined_no_duplicate_pairs():
    data = _load()
    total_dupes = 0
    for name, rel in data.items():
        pairs = rel.get("pairs", []) if isinstance(rel, dict) else rel
        seen = set()
        for pair in pairs:
            key = (pair[0], pair[1])
            if key in seen:
                total_dupes += 1
            seen.add(key)
    # Allow a tiny number of duplicates from data merging
    assert total_dupes <= 5, (
        f"Too many duplicate pairs across relations: {total_dupes}"
    )
