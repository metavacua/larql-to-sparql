"""Tests for English grammar pair extraction."""

from larql_knowledge.ingest.grammar import (
    extract_grammar_pairs_from_text,
    generate_grammar_pairs,
)


def test_extract_determiner_noun():
    text = "The cat sat on the table."
    pairs = extract_grammar_pairs_from_text(text)
    assert "determiner_noun" in pairs
    assert ["the", "cat"] in pairs["determiner_noun"]
    assert ["the", "table"] in pairs["determiner_noun"]


def test_extract_preposition_noun():
    text = "The cat sat on the table."
    pairs = extract_grammar_pairs_from_text(text)
    assert "preposition_noun" in pairs
    assert ["on", "table"] not in pairs["preposition_noun"]  # "the" is between


def test_extract_copula_adjective():
    text = "The house is big and the sky is bright."
    pairs = extract_grammar_pairs_from_text(text)
    assert "copula_adjective" in pairs
    assert ["is", "big"] in pairs["copula_adjective"]
    assert ["is", "bright"] in pairs["copula_adjective"]


def test_extract_auxiliary_verb():
    text = "She can run and will swim."
    pairs = extract_grammar_pairs_from_text(text)
    assert "auxiliary_verb" in pairs
    assert ["can", "run"] in pairs["auxiliary_verb"]
    assert ["will", "swim"] in pairs["auxiliary_verb"]


def test_no_duplicates():
    text = "The cat and the cat and the cat."
    pairs = extract_grammar_pairs_from_text(text)
    det_nouns = pairs["determiner_noun"]
    cat_pairs = [p for p in det_nouns if p == ["the", "cat"]]
    assert len(cat_pairs) == 1


def test_generate_grammar_pairs_categories():
    pairs = generate_grammar_pairs()
    assert "determiner_noun" in pairs
    assert "preposition_noun" in pairs
    assert "copula_adjective" in pairs
    assert "auxiliary_verb" in pairs


def test_generate_grammar_pairs_nonempty():
    pairs = generate_grammar_pairs()
    for category, pair_list in pairs.items():
        assert len(pair_list) > 0, f"{category} is empty"


def test_generate_grammar_pairs_format():
    pairs = generate_grammar_pairs()
    for category, pair_list in pairs.items():
        for pair in pair_list:
            assert isinstance(pair, list)
            assert len(pair) == 2
            assert isinstance(pair[0], str)
            assert isinstance(pair[1], str)
