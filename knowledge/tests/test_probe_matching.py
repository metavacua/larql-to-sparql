"""Tests for probe matching logic — normalize_object, build_match_index."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

# Import the functions directly from the script module
import importlib.util
spec = importlib.util.spec_from_file_location(
    "probe_mlx",
    Path(__file__).parent.parent / "scripts" / "probe_mlx.py",
)
# We can't import the full module (needs mlx), so test the pure functions
# by extracting them. Use a simpler approach: just test the logic inline.


# -- Reproduce normalize_object and build_match_index locally for testing --

_STOP_WORDS = frozenset({
    "the", "of", "and", "in", "to", "for", "is", "on", "at", "by",
    "an", "or", "de", "la", "le", "el", "du", "des", "von", "van",
    "al", "bin", "ibn", "di", "da", "do", "das", "den", "der", "het",
})


def normalize_object(obj: str) -> list[str]:
    obj_lower = obj.lower().strip()
    forms = {obj_lower}
    words = obj_lower.split()
    if len(words) > 1:
        for word in words:
            if word.endswith("'s"):
                clean = word[:-2]
            elif word.endswith("\u2019s"):
                clean = word[:-2]
            else:
                clean = word
            if clean not in _STOP_WORDS and len(clean) >= 4:
                forms.add(clean)
    return list(forms)


def build_match_index(triples: dict) -> dict[tuple[str, str], str]:
    index = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) < 2:
                continue
            subj = pair[0].lower().strip()
            for form in normalize_object(pair[1]):
                key = (subj, form)
                if key not in index:
                    index[key] = rel_name
    return index


# ---------------------------------------------------------------------------
# normalize_object tests
# ---------------------------------------------------------------------------

def test_normalize_single_word():
    assert normalize_object("Paris") == ["paris"]


def test_normalize_two_word_name():
    forms = normalize_object("Emmanuel Macron")
    assert "emmanuel macron" in forms
    assert "macron" in forms
    assert "emmanuel" in forms


def test_normalize_filters_short_words():
    forms = normalize_object("New York")
    assert "new york" in forms
    # "new" is only 3 chars, should be filtered
    assert "new" not in forms
    assert "york" in forms


def test_normalize_filters_stop_words():
    forms = normalize_object("Leonardo da Vinci")
    assert "leonardo da vinci" in forms
    assert "leonardo" in forms
    assert "vinci" in forms
    assert "da" not in forms


def test_normalize_strips_possessive():
    forms = normalize_object("Schindler's List")
    assert "schindler's list" in forms
    assert "schindler" in forms


def test_normalize_preserves_single_word():
    forms = normalize_object("Berlin")
    assert forms == ["berlin"]


def test_normalize_handles_three_word():
    forms = normalize_object("Charles de Gaulle")
    assert "charles de gaulle" in forms
    assert "charles" in forms
    assert "gaulle" in forms
    assert "de" not in forms


# ---------------------------------------------------------------------------
# build_match_index tests
# ---------------------------------------------------------------------------

def test_match_index_exact():
    triples = {
        "capital": {"pairs": [["France", "Paris"]]}
    }
    idx = build_match_index(triples)
    assert idx[("france", "paris")] == "capital"


def test_match_index_partial_name():
    triples = {
        "head_of_state": {"pairs": [["France", "Emmanuel Macron"]]}
    }
    idx = build_match_index(triples)
    assert idx[("france", "emmanuel macron")] == "head_of_state"
    assert idx[("france", "macron")] == "head_of_state"
    assert idx[("france", "emmanuel")] == "head_of_state"


def test_match_index_no_cross_subject():
    triples = {
        "capital": {"pairs": [["France", "Paris"], ["Germany", "Berlin"]]}
    }
    idx = build_match_index(triples)
    assert ("france", "berlin") not in idx
    assert ("germany", "paris") not in idx


def test_match_index_multiple_relations():
    triples = {
        "capital": {"pairs": [["France", "Paris"]]},
        "language": {"pairs": [["France", "French"]]},
    }
    idx = build_match_index(triples)
    assert idx[("france", "paris")] == "capital"
    assert idx[("france", "french")] == "language"


def test_match_index_empty_pairs():
    triples = {"capital": {"pairs": []}}
    idx = build_match_index(triples)
    assert len(idx) == 0


def test_match_index_multi_word_subject():
    triples = {
        "director": {"pairs": [["Schindlers List", "Steven Spielberg"]]}
    }
    idx = build_match_index(triples)
    assert idx[("schindlers list", "spielberg")] == "director"
    assert idx[("schindlers list", "steven")] == "director"


def test_match_index_first_relation_wins():
    """When two relations map the same (subject, form), first wins."""
    triples = {
        "birthplace": {"pairs": [["Mozart", "Salzburg"]]},
        "deathplace": {"pairs": [["Mozart", "Salzburg"]]},
    }
    idx = build_match_index(triples)
    # Both map ("mozart", "salzburg") — first one wins
    assert idx[("mozart", "salzburg")] == "birthplace"
