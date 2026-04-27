"""Tests for AST pair extraction."""

from larql_knowledge.ingest.ast_extract import extract_pairs_from_source


def test_extract_function_def():
    source = "def hello():\n    pass\n"
    pairs = extract_pairs_from_source(source)
    assert "def" in pairs
    assert ["def", "hello"] in pairs["def"]


def test_extract_class_def():
    source = "class MyClass:\n    pass\n"
    pairs = extract_pairs_from_source(source)
    assert "class" in pairs
    assert ["class", "MyClass"] in pairs["class"]


def test_extract_import():
    source = "import os\nimport sys\n"
    pairs = extract_pairs_from_source(source)
    assert "import" in pairs
    assert ["import", "os"] in pairs["import"]
    assert ["import", "sys"] in pairs["import"]


def test_extract_from_import():
    source = "from pathlib import Path\n"
    pairs = extract_pairs_from_source(source)
    assert "from_import" in pairs
    assert ["from_import", "pathlib"] in pairs["from_import"]


def test_extract_raise():
    source = "raise ValueError('bad')\n"
    pairs = extract_pairs_from_source(source)
    assert "raise" in pairs
    assert ["raise", "ValueError"] in pairs["raise"]


def test_extract_except():
    source = "try:\n    pass\nexcept KeyError:\n    pass\n"
    pairs = extract_pairs_from_source(source)
    assert "except" in pairs
    assert ["except", "KeyError"] in pairs["except"]


def test_invalid_syntax_returns_empty():
    pairs = extract_pairs_from_source("def (broken syntax !!!")
    assert pairs == {}


def test_empty_source():
    pairs = extract_pairs_from_source("")
    assert pairs == {}


def test_multiple_functions():
    source = "def foo():\n    pass\ndef bar():\n    pass\n"
    pairs = extract_pairs_from_source(source)
    names = [p[1] for p in pairs["def"]]
    assert "foo" in names
    assert "bar" in names
