#!/usr/bin/env python3
"""Extract keyword->token AST pairs from code corpora for all supported languages.

Uses tree-sitter parsers when available, falling back to regex-based
extraction.  Writes one JSON file per language to the output directory.

Usage:
    # From a corpus directory with per-language subdirectories:
    python3 scripts/extract_all_ast_pairs.py --corpus-dir ~/code-corpus --output-dir data/ast

    # Specific languages only:
    python3 scripts/extract_all_ast_pairs.py --corpus-dir ~/code-corpus --languages rust,go,java

    # Auto-detect and parse system stdlib/builtin libraries:
    python3 scripts/extract_all_ast_pairs.py --stdlib --output-dir data/ast

    # Print supported languages:
    python3 scripts/extract_all_ast_pairs.py --language-info
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Any

# Add src to path for direct script execution
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from larql_knowledge.ingest.treesitter_extract import (
    LANGUAGE_EXTENSIONS,
    SUPPORTED_LANGUAGES,
    extract_pairs_from_directory,
    save_pairs,
)


# ---------------------------------------------------------------------------
# Stdlib / built-in library discovery
# ---------------------------------------------------------------------------

def _find_stdlib_dir(language: str) -> Path | None:
    """Try to locate the standard library source for *language*.

    Returns a Path if found, or None.
    """
    home = Path.home()

    if language == "rust":
        # rustup puts stdlib source in the toolchain sysroot
        rustup_home = Path(os.environ.get("RUSTUP_HOME", home / ".rustup"))
        toolchains = rustup_home / "toolchains"
        if toolchains.is_dir():
            for tc in sorted(toolchains.iterdir(), reverse=True):
                lib_src = tc / "lib" / "rustlib" / "src" / "rust" / "library"
                if lib_src.is_dir():
                    return lib_src
        # Also check cargo registry for popular crates
        cargo_registry = home / ".cargo" / "registry" / "src"
        if cargo_registry.is_dir():
            return cargo_registry

    elif language == "go":
        goroot = os.environ.get("GOROOT")
        if goroot:
            src = Path(goroot) / "src"
            if src.is_dir():
                return src
        # Common locations
        for candidate in [
            Path("/usr/local/go/src"),
            Path("/usr/lib/go/src"),
            home / "go" / "src",
            home / "sdk" / "go" / "src",
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "java":
        java_home = os.environ.get("JAVA_HOME")
        if java_home:
            src_zip = Path(java_home) / "lib" / "src.zip"
            src_dir = Path(java_home) / "src"
            if src_dir.is_dir():
                return src_dir
            if src_zip.exists():
                return src_zip.parent  # caller needs to handle zip
        # OpenJDK source
        for candidate in [
            Path("/usr/lib/jvm"),
            home / ".sdkman" / "candidates" / "java",
        ]:
            if candidate.is_dir():
                for jdk in sorted(candidate.iterdir(), reverse=True):
                    src = jdk / "src"
                    if src.is_dir():
                        return src

    elif language == "javascript" or language == "typescript":
        # Node.js built-in modules
        node_path = shutil.which("node")
        if node_path:
            node_dir = Path(node_path).resolve().parent.parent
            lib = node_dir / "lib"
            if lib.is_dir():
                return lib
        # Also check node_modules for popular packages
        for candidate in [
            Path("/usr/local/lib/node_modules"),
            home / "node_modules",
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "ruby":
        ruby_path = shutil.which("ruby")
        if ruby_path:
            ruby_dir = Path(ruby_path).resolve().parent.parent
            lib = ruby_dir / "lib" / "ruby"
            if lib.is_dir():
                return lib
        for candidate in [
            Path("/usr/lib/ruby"),
            Path("/usr/local/lib/ruby"),
            home / ".rbenv" / "versions",
            home / ".rvm" / "src",
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "c" or language == "cpp":
        for candidate in [
            Path("/usr/include"),
            Path("/usr/local/include"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "php":
        for candidate in [
            Path("/usr/share/php"),
            Path("/usr/local/share/php"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "elixir":
        for candidate in [
            Path("/usr/lib/elixir/lib"),
            Path("/usr/local/lib/elixir/lib"),
            home / ".asdf" / "installs" / "elixir",
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "lua":
        for candidate in [
            Path("/usr/share/lua"),
            Path("/usr/local/share/lua"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "haskell":
        for candidate in [
            home / ".ghcup" / "ghc",
            home / ".stack" / "programs",
            Path("/usr/lib/ghc"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "swift":
        # Xcode / Swift toolchain
        for candidate in [
            Path("/Library/Developer/CommandLineTools/usr/share/swift"),
            Path("/Applications/Xcode.app/Contents/Developer/Toolchains/"
                 "XcodeDefault.xctoolchain/usr/lib/swift"),
            home / ".swiftenv" / "versions",
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "kotlin":
        for candidate in [
            home / ".sdkman" / "candidates" / "kotlin",
            Path("/usr/share/kotlin"),
            Path("/usr/local/share/kotlin"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "scala":
        for candidate in [
            home / ".sdkman" / "candidates" / "scala",
            Path("/usr/share/scala"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "bash":
        for candidate in [
            Path("/etc"),
            Path("/usr/share/bash-completion"),
        ]:
            if candidate.is_dir():
                return candidate

    elif language == "sql":
        # No standard stdlib for SQL
        return None

    elif language == "html" or language == "css":
        # No standard stdlib for markup/style
        return None

    return None


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------

def _filter_pairs(
    data: dict[str, Any],
    *,
    min_count: int = 5,
    max_pairs: int | None = None,
) -> dict[str, Any]:
    """Filter relations: drop pairs below *min_count*, cap at *max_pairs*."""
    filtered_relations: dict[str, Any] = {}
    for rel_key, rel_data in data.get("relations", {}).items():
        pairs = rel_data.get("pairs", [])
        if len(pairs) < min_count:
            continue
        if max_pairs is not None:
            pairs = pairs[:max_pairs]
        filtered_relations[rel_key] = {**rel_data, "pairs": pairs}

    return {**data, "relations": filtered_relations}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def print_language_info() -> None:
    """Print supported languages and their file extensions."""
    print(f"Supported languages ({len(SUPPORTED_LANGUAGES)}):\n")
    for lang in SUPPORTED_LANGUAGES:
        exts = LANGUAGE_EXTENSIONS.get(lang, [])
        ext_str = ", ".join(exts)
        stdlib_dir = _find_stdlib_dir(lang)
        stdlib_str = f"  stdlib: {stdlib_dir}" if stdlib_dir else "  stdlib: not found"
        print(f"  {lang:<14s}  extensions: {ext_str:<30s}{stdlib_str}")


def process_language(
    language: str,
    source_dir: Path,
    output_dir: Path,
    *,
    max_files: int = 500,
    min_count: int = 5,
    max_pairs: int | None = None,
) -> dict[str, Any] | None:
    """Extract and save AST pairs for a single language."""
    print(f"\n{'='*60}")
    print(f"Processing: {language}")
    print(f"  Source: {source_dir}")

    if not source_dir.is_dir():
        print(f"  SKIP: directory does not exist")
        return None

    data = extract_pairs_from_directory(
        source_dir, language, max_files=max_files
    )

    if data["num_files"] == 0:
        print(f"  SKIP: no matching files found")
        return None

    print(f"  Parsed {data['num_files']} files")
    total_before = sum(
        len(r["pairs"]) for r in data["relations"].values()
    )
    print(f"  Raw: {len(data['relations'])} relation types, {total_before} pairs")

    # Apply filtering
    data = _filter_pairs(data, min_count=min_count, max_pairs=max_pairs)

    total_after = sum(
        len(r["pairs"]) for r in data["relations"].values()
    )
    print(f"  After filter (>={min_count}): {len(data['relations'])} types, {total_after} pairs")

    if not data["relations"]:
        print(f"  SKIP: no relations passed filter")
        return None

    # Print top relations
    for rel_key, rel_data in sorted(
        data["relations"].items(), key=lambda x: -len(x[1]["pairs"])
    )[:10]:
        kw = rel_data.get("keyword", "?")
        print(f"    {rel_key:<30s} ({kw:<10s}) {len(rel_data['pairs']):>5d} pairs")

    # Save
    output_path = output_dir / f"{language}_ast.json"
    save_pairs(data, output_path)
    print(f"  Saved to {output_path}")

    return data


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract AST keyword->token pairs from code corpora."
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        help="Directory containing code corpus (with per-language subdirectories or mixed).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "data" / "ast",
        help="Directory to write output JSON files (default: data/ast/).",
    )
    parser.add_argument(
        "--languages",
        type=str,
        default=None,
        help="Comma-separated list of languages to process (default: all).",
    )
    parser.add_argument(
        "--max-files",
        type=int,
        default=500,
        help="Maximum files to parse per language (default: 500).",
    )
    parser.add_argument(
        "--max-pairs",
        type=int,
        default=None,
        help="Maximum pairs to keep per relation type (default: unlimited).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Minimum pairs for a relation to be kept (default: 5).",
    )
    parser.add_argument(
        "--stdlib",
        action="store_true",
        help="Auto-detect and parse stdlib/builtin libraries for each language.",
    )
    parser.add_argument(
        "--language-info",
        action="store_true",
        help="Print supported languages, extensions, and stdlib paths, then exit.",
    )

    args = parser.parse_args()

    if args.language_info:
        print_language_info()
        return

    if not args.stdlib and args.corpus_dir is None:
        parser.error("Must specify --corpus-dir or --stdlib")

    languages = SUPPORTED_LANGUAGES
    if args.languages:
        languages = [l.strip().lower() for l in args.languages.split(",")]
        unknown = [l for l in languages if l not in SUPPORTED_LANGUAGES]
        if unknown:
            print(f"Warning: unknown languages: {', '.join(unknown)}")
            languages = [l for l in languages if l in SUPPORTED_LANGUAGES]

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    summary: dict[str, dict[str, int]] = {}

    for language in languages:
        source_dir: Path | None = None

        if args.stdlib:
            source_dir = _find_stdlib_dir(language)
            if source_dir is None:
                print(f"\n  {language}: no stdlib found, skipping")
                continue
        elif args.corpus_dir is not None:
            # Try per-language subdirectory first, then scan entire corpus
            lang_subdir = args.corpus_dir / language
            if lang_subdir.is_dir():
                source_dir = lang_subdir
            else:
                source_dir = args.corpus_dir

        if source_dir is None:
            continue

        data = process_language(
            language,
            source_dir,
            output_dir,
            max_files=args.max_files,
            min_count=args.min_count,
            max_pairs=args.max_pairs,
        )

        if data is not None:
            total_pairs = sum(
                len(r["pairs"]) for r in data["relations"].values()
            )
            summary[language] = {
                "files": data["num_files"],
                "relations": len(data["relations"]),
                "pairs": total_pairs,
            }

    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    if not summary:
        print("  No data extracted.")
    else:
        print(f"  {'Language':<14s} {'Files':>6s} {'Relations':>10s} {'Pairs':>8s}")
        print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*8}")
        total_files = 0
        total_rels = 0
        total_pairs = 0
        for lang in sorted(summary):
            s = summary[lang]
            print(f"  {lang:<14s} {s['files']:>6d} {s['relations']:>10d} {s['pairs']:>8d}")
            total_files += s["files"]
            total_rels += s["relations"]
            total_pairs += s["pairs"]
        print(f"  {'-'*14} {'-'*6} {'-'*10} {'-'*8}")
        print(f"  {'TOTAL':<14s} {total_files:>6d} {total_rels:>10d} {total_pairs:>8d}")

    print(f"\nOutput directory: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
