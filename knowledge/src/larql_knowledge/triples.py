"""Triple file loading and assembly.

Handles loading individual triple JSON files and assembling them
into a combined wikidata_triples.json for the LARQL engine.
"""

import json
from pathlib import Path


def load_triple_file(path: Path) -> dict:
    """Load a single triple JSON file."""
    with open(path) as f:
        return json.load(f)


def load_all_triples(triples_dir: Path) -> dict:
    """Load all triple files from a directory into a combined dict."""
    combined = {}
    for f in sorted(triples_dir.glob("*.json")):
        data = load_triple_file(f)
        relation = data.get("relation", f.stem)
        combined[relation] = {
            "pid": data.get("pid", ""),
            "pairs": data.get("pairs", []),
        }
    return combined


def save_combined(triples: dict, output_path: Path):
    """Save combined triples to a single JSON file."""
    with open(output_path, "w") as f:
        json.dump(triples, f, indent=2, ensure_ascii=False)


def assemble(triples_dir: Path, output_path: Path) -> dict:
    """Load all triple files and assemble into combined output."""
    combined = load_all_triples(triples_dir)
    save_combined(combined, output_path)
    return combined


def stats(triples: dict) -> dict:
    """Compute statistics for a combined triples dict."""
    total_pairs = sum(len(v["pairs"]) for v in triples.values())
    return {
        "num_relations": len(triples),
        "total_pairs": total_pairs,
        "relations": {
            name: len(data["pairs"])
            for name, data in sorted(triples.items(), key=lambda x: -len(x[1]["pairs"]))
        },
    }


def merge_triples(target: dict, source: dict) -> int:
    """Merge source triples into target. Returns count of new pairs added."""
    added = 0
    for rel_name, rel_data in source.items():
        if rel_name not in target:
            target[rel_name] = rel_data
            added += len(rel_data.get("pairs", []))
        else:
            existing = set(tuple(p) for p in target[rel_name]["pairs"])
            for pair in rel_data.get("pairs", []):
                key = tuple(pair)
                if key not in existing:
                    target[rel_name]["pairs"].append(pair)
                    existing.add(key)
                    added += 1
    return added
