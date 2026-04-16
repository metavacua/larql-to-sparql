"""Model probing modules.

Run entities through model inference, capture gate activations,
match against reference triples to produce per-feature labels.
"""

from .labels import (
    load_feature_labels,
    load_feature_labels_rich,
    save_feature_labels,
    save_feature_labels_rich,
    merge_labels,
    merge_labels_rich,
    make_label,
)

__all__ = [
    "load_feature_labels",
    "load_feature_labels_rich",
    "save_feature_labels",
    "save_feature_labels_rich",
    "merge_labels",
    "merge_labels_rich",
    "make_label",
]
