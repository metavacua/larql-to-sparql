"""Validate the probe_templates.json data file."""

import json
from pathlib import Path


DATA_FILE = Path(__file__).parent.parent / "data" / "probe_templates.json"

CORE_RELATIONS = [
    "capital", "language", "continent", "occupation",
    "birthplace", "author", "director", "founder", "team",
]


def _load():
    with open(DATA_FILE) as f:
        return json.load(f)


def test_templates_exist():
    assert DATA_FILE.exists(), f"{DATA_FILE} not found"
    data = _load()
    assert isinstance(data, dict)


def test_templates_min_count():
    data = _load()
    assert len(data) >= 100, f"Only {len(data)} relations (expected >= 100)"


def test_templates_are_lists():
    data = _load()
    for key, val in data.items():
        assert isinstance(val, list), f"{key}: value is not a list"
        for item in val:
            assert isinstance(item, str), f"{key}: contains non-string item"


def test_templates_have_placeholder():
    data = _load()
    for key, templates in data.items():
        for tmpl in templates:
            assert "{X}" in tmpl, f"{key}: template missing {{X}}: {tmpl!r}"


def test_templates_nonempty():
    data = _load()
    for key, templates in data.items():
        assert len(templates) >= 1, f"{key}: has 0 templates"


def test_templates_cover_core_relations():
    data = _load()
    for rel in CORE_RELATIONS:
        assert rel in data, f"Missing core relation: {rel}"


def test_templates_no_comment_keys():
    data = _load()
    for key in data:
        assert not key.startswith("//"), f"Comment key found: {key}"


def test_template_strings_reasonable_length():
    data = _load()
    for key, templates in data.items():
        for tmpl in templates:
            assert 5 <= len(tmpl) <= 200, (
                f"{key}: template length {len(tmpl)} out of range: {tmpl!r}"
            )
