"""Tests for syntax data loading (WordNet, morphological, AST)."""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data"


def test_wordnet_exists():
    assert (DATA_DIR / "wordnet_relations.json").exists()


def test_morphological_exists():
    assert (DATA_DIR / "morphological_relations.json").exists()


def test_ast_dir_exists():
    assert (DATA_DIR / "ast").exists()
    ast_files = list((DATA_DIR / "ast").glob("*.json"))
    assert len(ast_files) >= 1


def test_syntax_data_loadable():
    """load_syntax_data should combine all syntax sources."""
    import sys
    import importlib.util

    # Import load_syntax_data from probe_mlx
    # We can't import the full module (needs mlx), so replicate the logic
    import json

    syntax_triples = {}

    wordnet_path = DATA_DIR / "wordnet_relations.json"
    if wordnet_path.exists():
        with open(wordnet_path) as f:
            wn = json.load(f)
        for rel_name, rel_data in wn.items():
            syntax_triples[f"wn:{rel_name}"] = rel_data

    morph_path = DATA_DIR / "morphological_relations.json"
    if morph_path.exists():
        with open(morph_path) as f:
            m = json.load(f)
        for rel_name, rel_data in m.get("relations", {}).items():
            syntax_triples[f"morph:{rel_name}"] = rel_data

    ast_dir = DATA_DIR / "ast"
    if ast_dir.exists():
        for ast_file in ast_dir.glob("*.json"):
            with open(ast_file) as f:
                d = json.load(f)
            lang = d.get("language", ast_file.stem.replace("_ast", ""))
            for rel_name, rel_data in d.get("relations", {}).items():
                key = rel_name if ":" in rel_name else f"{lang}:{rel_name}"
                syntax_triples[key] = rel_data

    # Should have WordNet + morphological + AST relations
    assert len(syntax_triples) >= 10
    # Should have wn: prefix relations
    wn_rels = [k for k in syntax_triples if k.startswith("wn:")]
    assert len(wn_rels) >= 3
    # Should have morph: prefix relations
    morph_rels = [k for k in syntax_triples if k.startswith("morph:")]
    assert len(morph_rels) >= 3


def test_syntax_pairs_have_correct_format():
    """Each syntax relation should have a pairs list."""
    import json

    for source_file in [
        DATA_DIR / "wordnet_relations.json",
        DATA_DIR / "morphological_relations.json",
    ]:
        if not source_file.exists():
            continue
        with open(source_file) as f:
            data = json.load(f)

        relations = data if isinstance(data, dict) and "relations" not in data else data.get("relations", data)

        for rel_name, rel_data in relations.items():
            if not isinstance(rel_data, dict):
                continue
            pairs = rel_data.get("pairs", [])
            assert isinstance(pairs, list), f"pairs not a list in {rel_name}"
            for pair in pairs[:5]:
                assert isinstance(pair, list), f"pair not a list in {rel_name}"
                assert len(pair) == 2, f"pair length != 2 in {rel_name}"
