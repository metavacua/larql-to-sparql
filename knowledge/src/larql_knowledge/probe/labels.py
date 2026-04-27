"""Feature label management -- load, save, merge probe-confirmed labels.

Supports two formats:
  - Legacy flat dict: {"L27_F9515": "capital", ...}
  - Rich list format: [{"layer": 27, "feature": 9515, "relation": "capital",
                         "source": "probe", "confidence": 0.97, "examples": [...]}]

The rich format is the canonical spec format.  Legacy flat dicts are
auto-converted on load so downstream code always sees a uniform list.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Rich label record helpers
# ---------------------------------------------------------------------------

def make_label(
    layer: int,
    feature: int,
    relation: str,
    *,
    source: str = "probe",
    confidence: float = 1.0,
    examples: list[list[str]] | None = None,
) -> dict[str, Any]:
    """Create a single rich label record."""
    return {
        "layer": layer,
        "feature": feature,
        "relation": relation,
        "source": source,
        "confidence": confidence,
        "examples": examples or [],
    }


def _parse_key(key: str) -> tuple[int, int]:
    """Parse 'L27_F9515' into (27, 9515).  Returns (-1, -1) on failure."""
    try:
        parts = key.split("_")
        layer = int(parts[0][1:])
        feature = int(parts[1][1:])
        return layer, feature
    except (IndexError, ValueError):
        return -1, -1


def _is_rich_format(data: Any) -> bool:
    """Return True if *data* is already in the rich list-of-dicts format."""
    if not isinstance(data, list):
        return False
    if len(data) == 0:
        return True
    first = data[0]
    return isinstance(first, dict) and "layer" in first and "relation" in first


def _flat_to_rich(flat: dict[str, str]) -> list[dict[str, Any]]:
    """Convert legacy flat dict to rich list format."""
    result: list[dict[str, Any]] = []
    for key, relation in flat.items():
        layer, feature = _parse_key(key)
        result.append(make_label(layer, feature, relation))
    return result


def _rich_to_flat(rich: list[dict[str, Any]]) -> dict[str, str]:
    """Convert rich list format back to legacy flat dict."""
    flat: dict[str, str] = {}
    for record in rich:
        layer = record.get("layer", -1)
        feature = record.get("feature", -1)
        key = f"L{layer}_F{feature}"
        flat[key] = record["relation"]
    return flat


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def load_feature_labels(path: Path) -> dict:
    """Load feature labels from JSON.  Returns {key: relation_name}.

    Accepts both the legacy flat dict and the rich list format on disk.
    Always returns the legacy flat dict for backward compatibility with
    existing call-sites.
    """
    if not path.exists():
        return {}
    with open(path) as f:
        data = json.load(f)

    if _is_rich_format(data):
        return _rich_to_flat(data)
    return data


def load_feature_labels_rich(path: Path) -> list[dict[str, Any]]:
    """Load feature labels from JSON in the rich list format.

    Accepts both the legacy flat dict and the rich list format on disk.
    Always returns the rich list.
    """
    if not path.exists():
        return []
    with open(path) as f:
        data = json.load(f)

    if _is_rich_format(data):
        return data
    return _flat_to_rich(data)


def save_feature_labels(labels: dict | list, path: Path) -> None:
    """Save feature labels to JSON.

    Accepts either the legacy flat dict or the rich list format.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(labels, f, indent=2, ensure_ascii=False)


def save_feature_labels_rich(labels: list[dict[str, Any]], path: Path) -> None:
    """Save feature labels in rich list format."""
    save_feature_labels(labels, path)


# ---------------------------------------------------------------------------
# Merge
# ---------------------------------------------------------------------------

def merge_labels(existing: dict, new_labels: dict) -> int:
    """Merge new labels into existing.  Returns count of new labels added.

    Existing labels are never overwritten -- probe-confirmed labels
    from earlier runs are preserved.
    """
    added = 0
    for key, rel in new_labels.items():
        if key not in existing:
            existing[key] = rel
            added += 1
    return added


def merge_labels_rich(
    existing: list[dict[str, Any]],
    new_labels: list[dict[str, Any]],
) -> int:
    """Merge rich labels.  Returns count of new labels added.

    Uses (layer, feature) as the dedup key.  Higher-confidence records
    replace lower-confidence ones.
    """
    index: dict[tuple[int, int], int] = {}
    for i, rec in enumerate(existing):
        index[(rec["layer"], rec["feature"])] = i

    added = 0
    for rec in new_labels:
        key = (rec["layer"], rec["feature"])
        if key not in index:
            existing.append(rec)
            index[key] = len(existing) - 1
            added += 1
        else:
            idx = index[key]
            if rec.get("confidence", 0) > existing[idx].get("confidence", 0):
                existing[idx] = rec
    return added


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

def labels_stats(labels: dict | list) -> dict:
    """Compute statistics for feature labels (flat dict or rich list)."""
    from collections import Counter

    if isinstance(labels, list):
        rel_counts = Counter(r["relation"] for r in labels)
        return {
            "total_features": len(labels),
            "num_relations": len(rel_counts),
            "relations": dict(rel_counts.most_common()),
        }

    rel_counts = Counter(labels.values())
    return {
        "total_features": len(labels),
        "num_relations": len(rel_counts),
        "relations": dict(rel_counts.most_common()),
    }
