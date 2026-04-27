"""Validate probe label output format."""

import json
import tempfile
from pathlib import Path

from larql_knowledge.probe.labels import (
    make_label,
    load_feature_labels_rich,
    save_feature_labels_rich,
)


def test_rich_label_has_required_fields():
    """A rich label must have all spec-required fields."""
    lbl = make_label(27, 9515, "capital", confidence=0.97,
                     examples=[["France", "Paris"]])
    required = {"layer", "feature", "relation", "source", "confidence", "examples"}
    assert required.issubset(set(lbl.keys()))


def test_rich_label_field_types():
    """Field types must match the spec."""
    lbl = make_label(27, 9515, "capital", confidence=0.97,
                     examples=[["France", "Paris"]])
    assert isinstance(lbl["layer"], int)
    assert isinstance(lbl["feature"], int)
    assert isinstance(lbl["relation"], str)
    assert isinstance(lbl["source"], str)
    assert isinstance(lbl["confidence"], float)
    assert isinstance(lbl["examples"], list)


def test_rich_label_confidence_range():
    """Confidence must be between 0 and 1."""
    lbl = make_label(27, 9515, "capital", confidence=0.97)
    assert 0.0 <= lbl["confidence"] <= 1.0


def test_rich_label_source_default():
    """Default source should be 'probe'."""
    lbl = make_label(27, 9515, "capital")
    assert lbl["source"] == "probe"


def test_rich_label_roundtrip_json():
    """Labels must survive JSON serialization."""
    labels = [
        make_label(27, 9515, "capital", confidence=0.97,
                   examples=[["France", "Paris"], ["Germany", "Berlin"]]),
        make_label(24, 4532, "language", source="manual", confidence=0.85),
    ]

    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "probe_labels.json"
        save_feature_labels_rich(labels, path)

        with open(path) as f:
            raw = json.load(f)

        assert isinstance(raw, list)
        assert len(raw) == 2

        # Reload via the API
        loaded = load_feature_labels_rich(path)
        assert len(loaded) == 2
        assert loaded[0]["layer"] == 27
        assert loaded[0]["examples"] == [["France", "Paris"], ["Germany", "Berlin"]]
        assert loaded[1]["source"] == "manual"


def test_rich_label_empty_examples():
    """Labels with no examples should have an empty list."""
    lbl = make_label(10, 100, "nationality")
    assert lbl["examples"] == []


def test_rich_label_spec_format():
    """Verify the exact spec format from the requirements."""
    lbl = make_label(
        27, 9515, "capital",
        source="probe",
        confidence=0.97,
        examples=[["France", "Paris"]],
    )
    expected_keys = {"layer", "feature", "relation", "source",
                     "confidence", "examples"}
    assert set(lbl.keys()) == expected_keys
    assert lbl == {
        "layer": 27,
        "feature": 9515,
        "relation": "capital",
        "source": "probe",
        "confidence": 0.97,
        "examples": [["France", "Paris"]],
    }
