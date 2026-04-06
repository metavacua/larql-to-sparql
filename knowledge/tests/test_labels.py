"""Tests for feature label management."""

import json
import tempfile
from pathlib import Path

from larql_knowledge.probe.labels import (
    load_feature_labels,
    load_feature_labels_rich,
    save_feature_labels,
    save_feature_labels_rich,
    merge_labels,
    merge_labels_rich,
    labels_stats,
    make_label,
)


# ------------------------------------------------------------------
# Legacy flat dict tests (backward compatibility)
# ------------------------------------------------------------------

def test_save_and_load():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.json"
        labels = {"L27_F9515": "capital", "L24_F4532": "language"}
        save_feature_labels(labels, path)
        loaded = load_feature_labels(path)
        assert loaded == labels


def test_load_nonexistent():
    labels = load_feature_labels(Path("/nonexistent"))
    assert labels == {}


def test_merge_adds_new():
    existing = {"L27_F9515": "capital"}
    new = {"L24_F4532": "language", "L25_F4207": "continent"}
    added = merge_labels(existing, new)
    assert added == 2
    assert len(existing) == 3


def test_merge_preserves_existing():
    existing = {"L27_F9515": "capital"}
    new = {"L27_F9515": "WRONG"}  # should not overwrite
    added = merge_labels(existing, new)
    assert added == 0
    assert existing["L27_F9515"] == "capital"


def test_labels_stats():
    labels = {
        "L27_F9515": "capital",
        "L24_F4532": "language",
        "L25_F3603": "language",
        "L25_F4207": "continent",
    }
    s = labels_stats(labels)
    assert s["total_features"] == 4
    assert s["num_relations"] == 3
    assert s["relations"]["language"] == 2


# ------------------------------------------------------------------
# Rich format tests
# ------------------------------------------------------------------

def test_make_label():
    lbl = make_label(27, 9515, "capital", confidence=0.97,
                     examples=[["France", "Paris"]])
    assert lbl["layer"] == 27
    assert lbl["feature"] == 9515
    assert lbl["relation"] == "capital"
    assert lbl["source"] == "probe"
    assert lbl["confidence"] == 0.97
    assert lbl["examples"] == [["France", "Paris"]]


def test_save_and_load_rich():
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.json"
        labels = [
            make_label(27, 9515, "capital", confidence=0.97),
            make_label(24, 4532, "language", confidence=0.85),
        ]
        save_feature_labels_rich(labels, path)
        loaded = load_feature_labels_rich(path)
        assert len(loaded) == 2
        assert loaded[0]["relation"] == "capital"
        assert loaded[1]["confidence"] == 0.85


def test_load_rich_nonexistent():
    labels = load_feature_labels_rich(Path("/nonexistent"))
    assert labels == []


def test_load_flat_as_rich():
    """Loading a legacy flat file with load_feature_labels_rich converts it."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.json"
        flat = {"L27_F9515": "capital", "L24_F4532": "language"}
        save_feature_labels(flat, path)

        rich = load_feature_labels_rich(path)
        assert len(rich) == 2
        rels = {r["relation"] for r in rich}
        assert rels == {"capital", "language"}
        assert all("layer" in r for r in rich)


def test_load_rich_as_flat():
    """Loading a rich file with load_feature_labels returns legacy flat dict."""
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "labels.json"
        labels = [
            make_label(27, 9515, "capital"),
            make_label(24, 4532, "language"),
        ]
        save_feature_labels_rich(labels, path)

        flat = load_feature_labels(path)
        assert flat == {"L27_F9515": "capital", "L24_F4532": "language"}


def test_merge_labels_rich():
    existing = [make_label(27, 9515, "capital", confidence=0.9)]
    new = [
        make_label(24, 4532, "language", confidence=0.85),
        make_label(25, 4207, "continent", confidence=0.7),
    ]
    added = merge_labels_rich(existing, new)
    assert added == 2
    assert len(existing) == 3


def test_merge_rich_higher_confidence_replaces():
    existing = [make_label(27, 9515, "capital", confidence=0.7)]
    new = [make_label(27, 9515, "capital", confidence=0.95)]
    added = merge_labels_rich(existing, new)
    assert added == 0
    assert existing[0]["confidence"] == 0.95


def test_merge_rich_lower_confidence_kept():
    existing = [make_label(27, 9515, "capital", confidence=0.95)]
    new = [make_label(27, 9515, "capital", confidence=0.5)]
    added = merge_labels_rich(existing, new)
    assert added == 0
    assert existing[0]["confidence"] == 0.95


def test_labels_stats_rich():
    labels = [
        make_label(27, 9515, "capital"),
        make_label(24, 4532, "language"),
        make_label(25, 3603, "language"),
        make_label(25, 4207, "continent"),
    ]
    s = labels_stats(labels)
    assert s["total_features"] == 4
    assert s["num_relations"] == 3
    assert s["relations"]["language"] == 2
