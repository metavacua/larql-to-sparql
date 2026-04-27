"""AST pair extraction for Python source files.

Parses Python source using the built-in ``ast`` module and extracts
keyword -> following_token pairs at AST boundaries.  These capture
syntactic co-occurrence patterns that transformers learn:

    def -> function_name
    class -> class_name
    import -> module_name
    return -> expression_type
    if/while/for -> condition_token
    raise -> exception_name
    with -> context_manager
    except -> exception_type
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Visitor
# ---------------------------------------------------------------------------

class _PairCollector(ast.NodeVisitor):
    """Walk an AST and collect keyword->token pairs."""

    def __init__(self) -> None:
        self.pairs: dict[str, list[list[str]]] = {}

    def _add(self, keyword: str, token: str) -> None:
        if not token or not token.strip():
            return
        self.pairs.setdefault(keyword, [])
        pair = [keyword, token]
        if pair not in self.pairs[keyword]:
            self.pairs[keyword].append(pair)

    # -- visitors --

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        self._add("def", node.name)
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        self._add("async def", node.name)
        self.generic_visit(node)

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self._add("class", node.name)
        for base in node.bases:
            name = _name_of(base)
            if name:
                self._add("class_inherits", name)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self._add("import", alias.name.split(".")[0])
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        if node.module:
            self._add("from_import", node.module.split(".")[0])
        self.generic_visit(node)

    def visit_Return(self, node: ast.Return) -> None:
        if node.value is not None:
            rtype = type(node.value).__name__
            self._add("return", rtype)
        self.generic_visit(node)

    def visit_Raise(self, node: ast.Raise) -> None:
        if node.exc is not None:
            name = _name_of(node.exc)
            if name:
                self._add("raise", name)
        self.generic_visit(node)

    def visit_If(self, node: ast.If) -> None:
        tok = _first_token(node.test)
        if tok:
            self._add("if", tok)
        self.generic_visit(node)

    def visit_While(self, node: ast.While) -> None:
        tok = _first_token(node.test)
        if tok:
            self._add("while", tok)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        tok = _name_of(node.target)
        if tok:
            self._add("for", tok)
        self.generic_visit(node)

    def visit_With(self, node: ast.With) -> None:
        for item in node.items:
            name = _name_of(item.context_expr)
            if name:
                self._add("with", name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.type is not None:
            name = _name_of(node.type)
            if name:
                self._add("except", name)
        self.generic_visit(node)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            name = _name_of(target)
            if name:
                vtype = type(node.value).__name__
                self._add("assign", vtype)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        name = _name_of(node.func)
        if name:
            self._add("call", name)
        self.generic_visit(node)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _name_of(node: ast.AST | None) -> str | None:
    """Try to extract a simple name string from an AST node."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Subscript):
        return _name_of(node.value)
    if isinstance(node, ast.Starred):
        return _name_of(node.value)
    if isinstance(node, ast.Tuple) and node.elts:
        return _name_of(node.elts[0])
    return None


def _first_token(node: ast.AST | None) -> str | None:
    """Extract the 'first token' of a condition expression."""
    if node is None:
        return None
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Call):
        return _name_of(node.func)
    if isinstance(node, ast.Compare):
        return _first_token(node.left)
    if isinstance(node, ast.BoolOp):
        return _first_token(node.values[0]) if node.values else None
    if isinstance(node, ast.UnaryOp):
        return _first_token(node.operand)
    return type(node).__name__


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pairs_from_source(source: str) -> dict[str, list[list[str]]]:
    """Parse Python source and return {keyword: [[keyword, token], ...]}."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return {}
    collector = _PairCollector()
    collector.visit(tree)
    return collector.pairs


def extract_pairs_from_file(path: Path) -> dict[str, list[list[str]]]:
    """Parse a Python file and return keyword->token pairs."""
    source = path.read_text(encoding="utf-8", errors="replace")
    return extract_pairs_from_source(source)


def extract_pairs_from_directory(
    directory: Path,
    *,
    max_files: int = 500,
) -> dict[str, Any]:
    """Extract AST pairs from all .py files under *directory*.

    Returns a dict suitable for writing to JSON::

        {
            "source": "python_ast",
            "num_files": 42,
            "relations": {
                "def": {"pairs": [["def", "main"], ...]},
                ...
            }
        }
    """
    combined: dict[str, list[list[str]]] = {}
    num_files = 0

    for py_file in sorted(directory.rglob("*.py"))[:max_files]:
        try:
            pairs = extract_pairs_from_file(py_file)
        except (OSError, UnicodeDecodeError):
            continue
        num_files += 1
        for keyword, pair_list in pairs.items():
            existing = combined.setdefault(keyword, [])
            existing_set = {tuple(p) for p in existing}
            for pair in pair_list:
                if tuple(pair) not in existing_set:
                    existing.append(pair)
                    existing_set.add(tuple(pair))

    relations = {
        keyword: {"pairs": pair_list}
        for keyword, pair_list in sorted(combined.items())
    }

    return {
        "source": "python_ast",
        "num_files": num_files,
        "relations": relations,
    }


def save_ast_pairs(data: dict[str, Any], output_path: Path) -> None:
    """Save extracted AST pairs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
