"""AST pair extraction for multiple programming languages via tree-sitter.

Uses tree-sitter Python bindings when the grammar is installed, falling
back to regex-based extraction otherwise.  Extracts keyword -> token
pairs from source code at AST boundaries -- the same syntactic
co-occurrence patterns that transformers learn.

Supported languages: rust, javascript, typescript, java, go, c, cpp,
ruby, php, kotlin, swift, scala, haskell, bash, sql, lua, elixir,
html, css.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Language metadata
# ---------------------------------------------------------------------------

SUPPORTED_LANGUAGES: list[str] = [
    "rust",
    "javascript",
    "typescript",
    "java",
    "go",
    "c",
    "cpp",
    "ruby",
    "php",
    "kotlin",
    "swift",
    "scala",
    "haskell",
    "bash",
    "sql",
    "lua",
    "elixir",
    "html",
    "css",
]

LANGUAGE_EXTENSIONS: dict[str, list[str]] = {
    "rust": [".rs"],
    "javascript": [".js", ".mjs", ".cjs", ".jsx"],
    "typescript": [".ts", ".tsx"],
    "java": [".java"],
    "go": [".go"],
    "c": [".c", ".h"],
    "cpp": [".cpp", ".cxx", ".cc", ".hpp", ".hxx", ".hh"],
    "ruby": [".rb"],
    "php": [".php"],
    "kotlin": [".kt", ".kts"],
    "swift": [".swift"],
    "scala": [".scala", ".sc"],
    "haskell": [".hs", ".lhs"],
    "bash": [".sh", ".bash"],
    "sql": [".sql"],
    "lua": [".lua"],
    "elixir": [".ex", ".exs"],
    "html": [".html", ".htm"],
    "css": [".css"],
}

_EXTENSION_TO_LANGUAGE: dict[str, str] = {}
for _lang, _exts in LANGUAGE_EXTENSIONS.items():
    for _ext in _exts:
        _EXTENSION_TO_LANGUAGE[_ext] = _lang

# ---------------------------------------------------------------------------
# Per-language keyword definitions
# Each entry maps a keyword to a (relation_tag, description) tuple.
# The relation_tag is used as JSON key; the description documents the pair.
# ---------------------------------------------------------------------------

_LANGUAGE_KEYWORDS: dict[str, dict[str, tuple[str, str]]] = {
    "rust": {
        "fn":     ("function_def",   "Function definition: fn keyword followed by function name"),
        "let":    ("variable_bind",  "Variable binding: let keyword followed by variable name"),
        "use":    ("use_import",     "Use import: use keyword followed by module path"),
        "impl":   ("impl_block",     "Impl block: impl keyword followed by type name"),
        "struct": ("struct_def",     "Struct definition: struct keyword followed by struct name"),
        "enum":   ("enum_def",       "Enum definition: enum keyword followed by enum name"),
        "match":  ("match_expr",     "Match expression: match keyword followed by scrutinee"),
        "trait":  ("trait_def",      "Trait definition: trait keyword followed by trait name"),
        "pub":    ("pub_item",       "Public item: pub keyword followed by fn/struct/etc"),
        "mod":    ("module_def",     "Module definition: mod keyword followed by module name"),
    },
    "javascript": {
        "function": ("function_def",  "Function definition: function keyword followed by name"),
        "const":    ("const_decl",    "Const declaration: const keyword followed by name"),
        "let":      ("let_decl",      "Let declaration: let keyword followed by name"),
        "require":  ("require_call",  "Require call: require keyword followed by module"),
        "class":    ("class_def",     "Class definition: class keyword followed by name"),
        "import":   ("import_decl",   "Import declaration: import keyword followed by module"),
        "export":   ("export_decl",   "Export declaration: export keyword followed by item"),
        "async":    ("async_fn",      "Async function: async keyword followed by function"),
        "new":      ("new_expr",      "New expression: new keyword followed by constructor name"),
    },
    "typescript": {
        "interface": ("interface_def", "Interface definition: interface keyword followed by name"),
        "type":      ("type_alias",    "Type alias: type keyword followed by name"),
        "enum":      ("enum_def",      "Enum definition: enum keyword followed by name"),
        "extends":   ("extends_cls",   "Extends clause: extends keyword followed by parent type"),
        "implements": ("implements_if", "Implements clause: implements keyword followed by interface"),
    },
    "java": {
        "class":     ("class_def",    "Class definition: class keyword followed by name"),
        "import":    ("import_decl",  "Import declaration: import keyword followed by package"),
        "void":      ("void_method",  "Void method: void keyword followed by method name"),
        "public":    ("public_decl",  "Public declaration: public keyword followed by class/method"),
        "interface": ("interface_def","Interface definition: interface keyword followed by name"),
        "extends":   ("extends_cls",  "Extends clause: extends keyword followed by parent class"),
    },
    "go": {
        "func":      ("function_def", "Function definition: func keyword followed by name"),
        "import":    ("import_decl",  "Import declaration: import keyword followed by package"),
        "var":       ("var_decl",     "Variable declaration: var keyword followed by name"),
        "type":      ("type_def",     "Type definition: type keyword followed by name"),
        "struct":    ("struct_def",   "Struct definition: struct keyword inside type block"),
        "interface": ("interface_def","Interface definition: interface keyword inside type block"),
    },
    "c": {
        "int":      ("int_decl",     "Int declaration: int keyword followed by identifier"),
        "struct":   ("struct_def",   "Struct definition: struct keyword followed by name"),
        "void":     ("void_fn",      "Void function: void keyword followed by function name"),
        "typedef":  ("typedef_decl", "Typedef: typedef keyword followed by type name"),
        "#define":  ("macro_def",    "Macro definition: #define directive followed by macro name"),
        "#include": ("include_dir",  "Include directive: #include followed by header name"),
    },
    "cpp": {
        "class":     ("class_def",     "Class definition: class keyword followed by name"),
        "template":  ("template_decl", "Template declaration: template keyword followed by type"),
        "namespace": ("namespace_def", "Namespace definition: namespace keyword followed by name"),
        "virtual":   ("virtual_meth", "Virtual method: virtual keyword followed by method"),
        "auto":      ("auto_var",      "Auto variable: auto keyword followed by variable name"),
    },
    "ruby": {
        "def":     ("method_def",   "Method definition: def keyword followed by method name"),
        "class":   ("class_def",    "Class definition: class keyword followed by name"),
        "module":  ("module_def",   "Module definition: module keyword followed by name"),
        "require": ("require_call", "Require call: require keyword followed by library string"),
        "attr":    ("attr_decl",    "Attribute declaration: attr_* keyword followed by symbol"),
    },
    "php": {
        "function":  ("function_def",  "Function definition: function keyword followed by name"),
        "class":     ("class_def",     "Class definition: class keyword followed by name"),
        "namespace": ("namespace_def", "Namespace definition: namespace keyword followed by name"),
        "use":       ("use_decl",      "Use declaration: use keyword followed by class name"),
    },
    "kotlin": {
        "fun":   ("function_def", "Function definition: fun keyword followed by name"),
        "val":   ("val_decl",     "Val declaration: val keyword followed by name"),
        "var":   ("var_decl",     "Var declaration: var keyword followed by name"),
        "class": ("class_def",    "Class definition: class keyword followed by name"),
        "data":  ("data_class",   "Data class: data keyword followed by class"),
    },
    "swift": {
        "func":     ("function_def", "Function definition: func keyword followed by name"),
        "class":    ("class_def",    "Class definition: class keyword followed by name"),
        "struct":   ("struct_def",   "Struct definition: struct keyword followed by name"),
        "enum":     ("enum_def",     "Enum definition: enum keyword followed by name"),
        "let":      ("let_decl",     "Let declaration: let keyword followed by name"),
        "var":      ("var_decl",     "Var declaration: var keyword followed by name"),
        "protocol": ("protocol_def", "Protocol definition: protocol keyword followed by name"),
    },
    "scala": {
        "def":    ("method_def",  "Method definition: def keyword followed by name"),
        "val":    ("val_decl",    "Val declaration: val keyword followed by name"),
        "class":  ("class_def",   "Class definition: class keyword followed by name"),
        "object": ("object_def",  "Object definition: object keyword followed by name"),
        "trait":  ("trait_def",   "Trait definition: trait keyword followed by name"),
    },
    "haskell": {
        "data":     ("data_type",     "Data type: data keyword followed by type name"),
        "class":    ("typeclass_def", "Typeclass definition: class keyword followed by name"),
        "instance": ("instance_def",  "Instance declaration: instance keyword followed by typeclass"),
        "where":    ("where_clause",  "Where clause: where keyword followed by definition"),
        "import":   ("import_decl",   "Import declaration: import keyword followed by module"),
    },
    "bash": {
        "function": ("function_def", "Function definition: function keyword followed by name"),
        "if":       ("if_cond",      "If condition: if keyword followed by condition"),
        "for":      ("for_loop",     "For loop: for keyword followed by variable"),
        "export":   ("export_var",   "Export variable: export keyword followed by variable name"),
    },
    "sql": {
        "SELECT": ("select_col",   "Select column: SELECT keyword followed by column name"),
        "FROM":   ("from_table",   "From table: FROM keyword followed by table name"),
        "WHERE":  ("where_cond",   "Where condition: WHERE keyword followed by condition"),
        "JOIN":   ("join_table",   "Join table: JOIN keyword followed by table name"),
        "INSERT": ("insert_table", "Insert table: INSERT keyword followed by table name"),
        "CREATE": ("create_table", "Create table: CREATE keyword followed by table name"),
    },
    "lua": {
        "function": ("function_def", "Function definition: function keyword followed by name"),
        "local":    ("local_decl",   "Local declaration: local keyword followed by name"),
        "require":  ("require_call", "Require call: require keyword followed by module"),
    },
    "elixir": {
        "def":       ("function_def", "Function definition: def keyword followed by name"),
        "defmodule": ("module_def",   "Module definition: defmodule keyword followed by name"),
        "use":       ("use_macro",    "Use macro: use keyword followed by module name"),
        "import":    ("import_decl",  "Import declaration: import keyword followed by module"),
    },
    "html": {
        "div":    ("div_class",   "Div element: div tag with class attribute"),
        "a":      ("anchor_href", "Anchor element: a tag with href attribute"),
        "img":    ("img_src",     "Image element: img tag with src attribute"),
        "input":  ("input_type",  "Input element: input tag with type attribute"),
        "form":   ("form_action", "Form element: form tag with action attribute"),
        "script": ("script_src",  "Script element: script tag with src attribute"),
    },
    "css": {
        "color":   ("color_prop",   "Color property: color keyword followed by value"),
        "font":    ("font_prop",    "Font property: font keyword followed by value"),
        "display": ("display_prop", "Display property: display keyword followed by value"),
        "margin":  ("margin_prop",  "Margin property: margin keyword followed by value"),
        "@media":  ("media_query",  "Media query: @media keyword followed by query"),
    },
}

# ---------------------------------------------------------------------------
# Regex fallback patterns
# Each pattern has one capture group for the token that follows the keyword.
# ---------------------------------------------------------------------------

_REGEX_PATTERNS: dict[str, dict[str, re.Pattern[str]]] = {
    "rust": {
        "fn":     re.compile(r"\bfn\s+(\w+)"),
        "let":    re.compile(r"\blet\s+(?:mut\s+)?(\w+)"),
        "use":    re.compile(r"\buse\s+([\w:]+)"),
        "impl":   re.compile(r"\bimpl(?:\s*<[^>]*>)?\s+(\w+)"),
        "struct": re.compile(r"\bstruct\s+(\w+)"),
        "enum":   re.compile(r"\benum\s+(\w+)"),
        "match":  re.compile(r"\bmatch\s+(\w+)"),
        "trait":  re.compile(r"\btrait\s+(\w+)"),
        "pub":    re.compile(r"\bpub\s+(?:\(crate\)\s+)?(\w+)"),
        "mod":    re.compile(r"\bmod\s+(\w+)"),
    },
    "javascript": {
        "function": re.compile(r"\bfunction\s+(\w+)"),
        "const":    re.compile(r"\bconst\s+(\w+)"),
        "let":      re.compile(r"\blet\s+(\w+)"),
        "require":  re.compile(r"""require\(\s*['"]([^'"]+)['"]\s*\)"""),
        "class":    re.compile(r"\bclass\s+(\w+)"),
        "import":   re.compile(r"""\bimport\s+.*?\bfrom\s+['"]([^'"]+)['"]"""),
        "export":   re.compile(r"\bexport\s+(?:default\s+)?(\w+)"),
        "async":    re.compile(r"\basync\s+function\s+(\w+)"),
        "new":      re.compile(r"\bnew\s+(\w+)"),
    },
    "typescript": {
        "interface":  re.compile(r"\binterface\s+(\w+)"),
        "type":       re.compile(r"\btype\s+(\w+)"),
        "enum":       re.compile(r"\benum\s+(\w+)"),
        "extends":    re.compile(r"\bextends\s+(\w+)"),
        "implements": re.compile(r"\bimplements\s+(\w+)"),
    },
    "java": {
        "class":     re.compile(r"\bclass\s+(\w+)"),
        "import":    re.compile(r"\bimport\s+([\w.]+)"),
        "void":      re.compile(r"\bvoid\s+(\w+)"),
        "public":    re.compile(r"\bpublic\s+(?:static\s+)?(?:final\s+)?(\w+)"),
        "interface": re.compile(r"\binterface\s+(\w+)"),
        "extends":   re.compile(r"\bextends\s+(\w+)"),
    },
    "go": {
        "func":      re.compile(r"\bfunc\s+(?:\([^)]*\)\s+)?(\w+)"),
        "import":    re.compile(r"""["']([^"']+)["']"""),
        "var":       re.compile(r"\bvar\s+(\w+)"),
        "type":      re.compile(r"\btype\s+(\w+)"),
        "struct":    re.compile(r"\btype\s+(\w+)\s+struct\b"),
        "interface": re.compile(r"\btype\s+(\w+)\s+interface\b"),
    },
    "c": {
        "int":      re.compile(r"\bint\s+(\w+)"),
        "struct":   re.compile(r"\bstruct\s+(\w+)"),
        "void":     re.compile(r"\bvoid\s+(\w+)"),
        "typedef":  re.compile(r"\btypedef\s+\w+\s+(\w+)"),
        "#define":  re.compile(r"#define\s+(\w+)"),
        "#include": re.compile(r"""#include\s*[<"]([^>"]+)[>"]"""),
    },
    "cpp": {
        "class":     re.compile(r"\bclass\s+(\w+)"),
        "template":  re.compile(r"\btemplate\s*<([^>]+)>"),
        "namespace": re.compile(r"\bnamespace\s+(\w+)"),
        "virtual":   re.compile(r"\bvirtual\s+\w+\s+(\w+)"),
        "auto":      re.compile(r"\bauto\s+(\w+)"),
    },
    "ruby": {
        "def":     re.compile(r"\bdef\s+(\w+[!?]?)"),
        "class":   re.compile(r"\bclass\s+(\w+)"),
        "module":  re.compile(r"\bmodule\s+(\w+)"),
        "require": re.compile(r"""require\s+['"]([^'"]+)['"]"""),
        "attr":    re.compile(r"\battr_\w+\s+:(\w+)"),
    },
    "php": {
        "function":  re.compile(r"\bfunction\s+(\w+)"),
        "class":     re.compile(r"\bclass\s+(\w+)"),
        "namespace": re.compile(r"\bnamespace\s+([\w\\]+)"),
        "use":       re.compile(r"\buse\s+([\w\\]+)"),
    },
    "kotlin": {
        "fun":   re.compile(r"\bfun\s+(?:<[^>]*>\s+)?(\w+)"),
        "val":   re.compile(r"\bval\s+(\w+)"),
        "var":   re.compile(r"\bvar\s+(\w+)"),
        "class": re.compile(r"\bclass\s+(\w+)"),
        "data":  re.compile(r"\bdata\s+class\s+(\w+)"),
    },
    "swift": {
        "func":     re.compile(r"\bfunc\s+(\w+)"),
        "class":    re.compile(r"\bclass\s+(\w+)"),
        "struct":   re.compile(r"\bstruct\s+(\w+)"),
        "enum":     re.compile(r"\benum\s+(\w+)"),
        "let":      re.compile(r"\blet\s+(\w+)"),
        "var":      re.compile(r"\bvar\s+(\w+)"),
        "protocol": re.compile(r"\bprotocol\s+(\w+)"),
    },
    "scala": {
        "def":    re.compile(r"\bdef\s+(\w+)"),
        "val":    re.compile(r"\bval\s+(\w+)"),
        "class":  re.compile(r"\bclass\s+(\w+)"),
        "object": re.compile(r"\bobject\s+(\w+)"),
        "trait":  re.compile(r"\btrait\s+(\w+)"),
    },
    "haskell": {
        "data":     re.compile(r"\bdata\s+(\w+)"),
        "class":    re.compile(r"\bclass\s+(?:.*=>)?\s*(\w+)"),
        "instance": re.compile(r"\binstance\s+(?:.*=>)?\s*(\w+)"),
        "where":    re.compile(r"\bwhere\s+(\w+)"),
        "import":   re.compile(r"\bimport\s+(?:qualified\s+)?(\w[\w.]*)"),
    },
    "bash": {
        "function": re.compile(r"\bfunction\s+(\w+)|(\w+)\s*\(\s*\)"),
        "if":       re.compile(r"\bif\s+\[?\[?\s*(-\w+\s+)?(\S+)"),
        "for":      re.compile(r"\bfor\s+(\w+)"),
        "export":   re.compile(r"\bexport\s+(\w+)"),
    },
    "sql": {
        "SELECT": re.compile(r"\bSELECT\s+(?:DISTINCT\s+)?(\w+)", re.IGNORECASE),
        "FROM":   re.compile(r"\bFROM\s+(\w+)", re.IGNORECASE),
        "WHERE":  re.compile(r"\bWHERE\s+(\w+)", re.IGNORECASE),
        "JOIN":   re.compile(r"\bJOIN\s+(\w+)", re.IGNORECASE),
        "INSERT": re.compile(r"\bINSERT\s+INTO\s+(\w+)", re.IGNORECASE),
        "CREATE": re.compile(r"\bCREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?(\w+)", re.IGNORECASE),
    },
    "lua": {
        "function": re.compile(r"\bfunction\s+(\w[\w.]*)"),
        "local":    re.compile(r"\blocal\s+(?:function\s+)?(\w+)"),
        "require":  re.compile(r"""require\s*\(?['"]([^'"]+)['"]\)?"""),
    },
    "elixir": {
        "def":       re.compile(r"\bdefp?\s+(\w+)"),
        "defmodule": re.compile(r"\bdefmodule\s+([\w.]+)"),
        "use":       re.compile(r"\buse\s+([\w.]+)"),
        "import":    re.compile(r"\bimport\s+([\w.]+)"),
    },
    "html": {
        "div":    re.compile(r"""<div\b[^>]*\bclass\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
        "a":      re.compile(r"""<a\b[^>]*\bhref\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
        "img":    re.compile(r"""<img\b[^>]*\bsrc\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
        "input":  re.compile(r"""<input\b[^>]*\btype\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
        "form":   re.compile(r"""<form\b[^>]*\baction\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
        "script": re.compile(r"""<script\b[^>]*\bsrc\s*=\s*['"]([^'"]+)['"]""", re.IGNORECASE),
    },
    "css": {
        "color":   re.compile(r"\bcolor\s*:\s*([^;}\s]+)"),
        "font":    re.compile(r"\bfont(?:-[\w]+)?\s*:\s*([^;}\s]+)"),
        "display": re.compile(r"\bdisplay\s*:\s*([^;}\s]+)"),
        "margin":  re.compile(r"\bmargin(?:-[\w]+)?\s*:\s*([^;}\s]+)"),
        "@media":  re.compile(r"@media\s+([^{]+)"),
    },
}

# ---------------------------------------------------------------------------
# Tree-sitter based extraction
# ---------------------------------------------------------------------------

# Tree-sitter grammar package names (pip install tree-sitter-<lang>)
_TS_GRAMMAR_PACKAGES: dict[str, str] = {
    "rust":       "tree_sitter_rust",
    "javascript": "tree_sitter_javascript",
    "typescript": "tree_sitter_typescript",
    "java":       "tree_sitter_java",
    "go":         "tree_sitter_go",
    "c":          "tree_sitter_c",
    "cpp":        "tree_sitter_cpp",
    "ruby":       "tree_sitter_ruby",
    "php":        "tree_sitter_php",
    "kotlin":     "tree_sitter_kotlin",
    "swift":      "tree_sitter_swift",
    "scala":      "tree_sitter_scala",
    "haskell":    "tree_sitter_haskell",
    "bash":       "tree_sitter_bash",
    "sql":        "tree_sitter_sql",
    "lua":        "tree_sitter_lua",
    "elixir":     "tree_sitter_elixir",
    "html":       "tree_sitter_html",
    "css":        "tree_sitter_css",
}

# Tree-sitter AST node types to keyword mapping.
# Maps (language, node_type) -> keyword for the extraction.
_TS_NODE_MAP: dict[str, dict[str, str]] = {
    "rust": {
        "function_item": "fn",
        "let_declaration": "let",
        "use_declaration": "use",
        "impl_item": "impl",
        "struct_item": "struct",
        "enum_item": "enum",
        "match_expression": "match",
        "trait_item": "trait",
        "mod_item": "mod",
    },
    "javascript": {
        "function_declaration": "function",
        "lexical_declaration": "const",
        "variable_declaration": "let",
        "call_expression": "require",
        "class_declaration": "class",
        "import_statement": "import",
        "export_statement": "export",
    },
    "java": {
        "class_declaration": "class",
        "import_declaration": "import",
        "method_declaration": "void",
        "interface_declaration": "interface",
    },
    "go": {
        "function_declaration": "func",
        "method_declaration": "func",
        "import_declaration": "import",
        "var_declaration": "var",
        "type_declaration": "type",
    },
}


def _try_get_ts_parser(language: str):
    """Try to load a tree-sitter parser for the given language.

    Returns (Parser, Language) or (None, None) if unavailable.
    """
    try:
        import tree_sitter  # noqa: F811
    except ImportError:
        return None, None

    pkg_name = _TS_GRAMMAR_PACKAGES.get(language)
    if pkg_name is None:
        return None, None

    try:
        mod = __import__(pkg_name)
        lang_fn = getattr(mod, "language", None)
        if lang_fn is None:
            return None, None
        lang_obj = tree_sitter.Language(lang_fn())
        parser = tree_sitter.Parser(lang_obj)
        return parser, lang_obj
    except Exception:
        return None, None


def _extract_name_from_ts_node(node) -> str | None:
    """Extract a name identifier from a tree-sitter node."""
    # Look for a direct child that is an identifier or name node
    for child in node.children:
        if child.type in ("identifier", "name", "type_identifier", "field_identifier"):
            return child.text.decode("utf-8", errors="replace")
    # For some nodes, the first named child is the name
    for child in node.named_children:
        if child.type in ("identifier", "name", "type_identifier"):
            return child.text.decode("utf-8", errors="replace")
    return None


def _extract_via_treesitter(
    source: str, language: str, parser, lang_obj
) -> dict[str, list[list[str]]] | None:
    """Extract pairs using a tree-sitter parser.

    Returns None if tree-sitter extraction is not supported for this
    language (falls back to regex).
    """
    node_map = _TS_NODE_MAP.get(language)
    if node_map is None:
        # No node mapping defined -- fall back to regex
        return None

    tree = parser.parse(source.encode("utf-8"))
    pairs: dict[str, list[list[str]]] = {}
    seen: dict[str, set[str]] = {}

    def _walk(node):
        keyword = node_map.get(node.type)
        if keyword is not None:
            name = _extract_name_from_ts_node(node)
            if name and name.strip():
                s = seen.setdefault(keyword, set())
                if name not in s:
                    pairs.setdefault(keyword, []).append([keyword, name])
                    s.add(name)
        for child in node.children:
            _walk(child)

    _walk(tree.root_node)
    return pairs if pairs else None


# ---------------------------------------------------------------------------
# Regex fallback extraction
# ---------------------------------------------------------------------------

def _extract_via_regex(source: str, language: str) -> dict[str, list[list[str]]]:
    """Extract pairs using regex patterns for *language*."""
    patterns = _REGEX_PATTERNS.get(language, {})
    pairs: dict[str, list[list[str]]] = {}
    seen: dict[str, set[str]] = {}

    for keyword, pattern in patterns.items():
        for match in pattern.finditer(source):
            # Use first non-None group
            token = None
            for g in match.groups():
                if g is not None:
                    token = g.strip()
                    break
            if not token:
                continue
            # Normalize: take last path segment for module paths
            if "/" in token:
                token = token.rsplit("/", 1)[-1]
            # Skip overly long tokens (probably not real identifiers)
            if len(token) > 120:
                continue
            s = seen.setdefault(keyword, set())
            if token not in s:
                pairs.setdefault(keyword, []).append([keyword, token])
                s.add(token)

    return pairs


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_pairs_from_source(
    source: str, language: str
) -> dict[str, list[list[str]]]:
    """Extract keyword->token pairs from source code.

    Parameters
    ----------
    source : str
        Source code text.
    language : str
        Language identifier (e.g. ``"rust"``, ``"javascript"``).

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping from keyword to list of ``[keyword, token]`` pairs.
    """
    language = language.lower()
    if language not in _REGEX_PATTERNS:
        return {}

    # Try tree-sitter first
    parser, lang_obj = _try_get_ts_parser(language)
    if parser is not None:
        result = _extract_via_treesitter(source, language, parser, lang_obj)
        if result is not None:
            return result

    # Fall back to regex
    return _extract_via_regex(source, language)


def detect_language(path: Path) -> str | None:
    """Detect language from file extension. Returns None if unknown."""
    return _EXTENSION_TO_LANGUAGE.get(path.suffix.lower())


def extract_pairs_from_file(
    path: Path, language: str | None = None
) -> dict[str, list[list[str]]]:
    """Extract keyword->token pairs from a source file.

    Parameters
    ----------
    path : Path
        Path to the source file.
    language : str or None
        Language identifier.  Auto-detected from extension if ``None``.

    Returns
    -------
    dict[str, list[list[str]]]
        Mapping from keyword to list of ``[keyword, token]`` pairs.
    """
    if language is None:
        language = detect_language(path)
    if language is None:
        return {}
    source = path.read_text(encoding="utf-8", errors="replace")
    return extract_pairs_from_source(source, language)


def extract_pairs_from_directory(
    directory: Path,
    language: str,
    *,
    max_files: int = 500,
) -> dict[str, Any]:
    """Extract AST pairs from all matching files under *directory*.

    Returns a dict suitable for writing to JSON::

        {
            "language": "rust",
            "source": "tree-sitter + regex",
            "num_files": 150,
            "relations": {
                "rs:function_def": {
                    "description": "...",
                    "keyword": "fn",
                    "pairs": [["fn", "main"], ...]
                }
            }
        }
    """
    language = language.lower()
    extensions = LANGUAGE_EXTENSIONS.get(language, [])
    if not extensions:
        return {"language": language, "source": "tree-sitter + regex",
                "num_files": 0, "relations": {}}

    combined: dict[str, list[list[str]]] = {}
    combined_seen: dict[str, set[tuple[str, str]]] = {}
    num_files = 0

    files: list[Path] = []
    for ext in extensions:
        files.extend(directory.rglob(f"*{ext}"))
    files = sorted(files)[:max_files]

    for src_file in files:
        try:
            pairs = extract_pairs_from_file(src_file, language)
        except (OSError, UnicodeDecodeError):
            continue
        num_files += 1
        for keyword, pair_list in pairs.items():
            s = combined_seen.setdefault(keyword, set())
            existing = combined.setdefault(keyword, [])
            for pair in pair_list:
                key = (pair[0], pair[1])
                if key not in s:
                    existing.append(pair)
                    s.add(key)

    # Build prefix from language
    kw_meta = _LANGUAGE_KEYWORDS.get(language, {})
    prefix = extensions[0].lstrip(".") if extensions else language[:2]

    relations: dict[str, dict[str, Any]] = {}
    for keyword, pair_list in sorted(combined.items()):
        meta = kw_meta.get(keyword)
        if meta is not None:
            tag, description = meta
            rel_key = f"{prefix}:{tag}"
        else:
            tag = keyword.replace(" ", "_").replace("#", "").lower()
            rel_key = f"{prefix}:{tag}"
            description = f"{keyword} keyword followed by token"

        relations[rel_key] = {
            "description": description,
            "keyword": keyword,
            "pairs": pair_list,
        }

    return {
        "language": language,
        "source": "tree-sitter + regex",
        "num_files": num_files,
        "relations": relations,
    }


def save_pairs(data: dict[str, Any], output_path: Path) -> None:
    """Save extracted pairs to JSON."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
