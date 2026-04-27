#!/usr/bin/env python3
"""Compare probe results across multiple models.

Loads feature_labels.json from each probe directory and generates a
comparison report showing overlap, per-model statistics, layer distribution,
and top confident features.

Usage:
    python3 scripts/compare_probes.py probes/gemma-3-4b-it/ probes/llama-3-8b/

    # Save JSON report
    python3 scripts/compare_probes.py probes/gemma-3-4b-it/ probes/llama-3-8b/ \\
        --output comparison_report.json
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path


def load_probe_dir(probe_dir: str) -> dict:
    """Load feature_labels.json from a probe directory.

    Returns:
        dict with keys: name, labels, path
    """
    probe_path = Path(probe_dir)
    labels_file = probe_path / "feature_labels.json"

    if not labels_file.exists():
        print(f"Warning: {labels_file} not found, skipping", file=sys.stderr)
        return None

    with open(labels_file, "r") as f:
        labels = json.load(f)

    return {
        "name": probe_path.name,
        "labels": labels,
        "path": str(probe_path),
    }


def parse_feature_key(key: str) -> tuple[int | None, int | None]:
    """Parse a feature key like 'L22_F8674' into (layer, feature_idx)."""
    m = re.match(r"L(\d+)_F(\d+)", key)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def analyze_model(probe: dict) -> dict:
    """Compute per-model statistics."""
    labels = probe["labels"]

    # Relation counts
    relation_counts = Counter(labels.values())

    # Layer distribution
    layer_counts = Counter()
    layer_relations = defaultdict(Counter)
    for key, relation in labels.items():
        layer, _ = parse_feature_key(key)
        if layer is not None:
            layer_counts[layer] += 1
            layer_relations[layer][relation] += 1

    # Sorted layer distribution
    layer_dist = {}
    for layer in sorted(layer_counts.keys()):
        layer_dist[layer] = {
            "count": layer_counts[layer],
            "relations": dict(layer_relations[layer].most_common(5)),
        }

    return {
        "name": probe["name"],
        "total_features": len(labels),
        "unique_relations": len(relation_counts),
        "relation_counts": dict(relation_counts.most_common()),
        "layer_distribution": layer_dist,
        "max_layer": max(layer_counts.keys()) if layer_counts else 0,
    }


def compute_overlap(analyses: list[dict]) -> dict:
    """Compute relation overlap across all models."""
    model_relations = {}
    for a in analyses:
        model_relations[a["name"]] = set(a["relation_counts"].keys())

    all_relations = set()
    for rels in model_relations.values():
        all_relations |= rels

    # Relations in ALL models
    if model_relations:
        shared = set.intersection(*model_relations.values())
    else:
        shared = set()

    # Relations unique to each model
    unique_per_model = {}
    for name, rels in model_relations.items():
        others = set()
        for other_name, other_rels in model_relations.items():
            if other_name != name:
                others |= other_rels
        unique_per_model[name] = sorted(rels - others)

    # Pairwise overlap matrix
    pairwise = {}
    names = sorted(model_relations.keys())
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            if i < j:
                overlap = model_relations[a] & model_relations[b]
                pairwise[f"{a} & {b}"] = {
                    "count": len(overlap),
                    "relations": sorted(overlap),
                }

    # Per-relation: which models have it?
    relation_presence = {}
    for rel in sorted(all_relations):
        present_in = [
            name for name, rels in model_relations.items() if rel in rels
        ]
        relation_presence[rel] = present_in

    return {
        "total_unique_relations": len(all_relations),
        "shared_across_all": sorted(shared),
        "shared_count": len(shared),
        "unique_per_model": unique_per_model,
        "pairwise_overlap": pairwise,
        "relation_presence": relation_presence,
    }


def format_report(analyses: list[dict], overlap: dict) -> str:
    """Format a human-readable comparison report."""
    lines = []
    w = 72

    lines.append("=" * w)
    lines.append("PROBE COMPARISON REPORT")
    lines.append("=" * w)
    lines.append("")

    # --- Summary table ---
    lines.append("MODEL SUMMARY")
    lines.append("-" * w)
    header = f"{'Model':<30} {'Features':>10} {'Relations':>10}"
    lines.append(header)
    lines.append("-" * w)
    for a in analyses:
        lines.append(
            f"{a['name']:<30} {a['total_features']:>10} {a['unique_relations']:>10}"
        )
    lines.append("")

    # --- Overlap ---
    lines.append("RELATION OVERLAP")
    lines.append("-" * w)
    lines.append(f"Total unique relations across all models: {overlap['total_unique_relations']}")
    lines.append(f"Relations found in ALL models: {overlap['shared_count']}")
    if overlap["shared_across_all"]:
        for rel in overlap["shared_across_all"]:
            # Show count per model
            counts = []
            for a in analyses:
                c = a["relation_counts"].get(rel, 0)
                counts.append(f"{a['name']}:{c}")
            lines.append(f"  {rel:<25} [{', '.join(counts)}]")
    lines.append("")

    # Unique per model
    for name, unique in overlap["unique_per_model"].items():
        if unique:
            lines.append(f"Unique to {name}: {', '.join(unique)}")
    lines.append("")

    # Pairwise
    if overlap["pairwise_overlap"]:
        lines.append("PAIRWISE OVERLAP")
        lines.append("-" * w)
        for pair, info in overlap["pairwise_overlap"].items():
            lines.append(f"  {pair}: {info['count']} shared relations")
        lines.append("")

    # --- Per-model details ---
    for a in analyses:
        lines.append(f"MODEL: {a['name']}")
        lines.append("-" * w)

        # Top relations
        lines.append("  Top relations:")
        for rel, count in list(a["relation_counts"].items())[:15]:
            bar = "#" * min(count, 40)
            lines.append(f"    {rel:<25} {count:>4}  {bar}")
        if len(a["relation_counts"]) > 15:
            lines.append(f"    ... and {len(a['relation_counts']) - 15} more")
        lines.append("")

        # Layer distribution
        lines.append("  Layer distribution:")
        max_count = max(
            (d["count"] for d in a["layer_distribution"].values()), default=1
        )
        for layer, info in sorted(a["layer_distribution"].items()):
            bar_len = int(info["count"] / max(max_count, 1) * 30)
            bar = "#" * bar_len
            top_rels = ", ".join(
                f"{r}({c})" for r, c in list(info["relations"].items())[:3]
            )
            lines.append(
                f"    L{layer:<3} {info['count']:>4}  {bar:<32} {top_rels}"
            )
        lines.append("")

    lines.append("=" * w)
    return "\n".join(lines)


def build_json_report(analyses: list[dict], overlap: dict) -> dict:
    """Build a structured JSON report."""
    return {
        "models": {a["name"]: a for a in analyses},
        "overlap": overlap,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare probe results across multiple models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 scripts/compare_probes.py probes/gemma-3-4b-it/ probes/llama-3-8b/
  python3 scripts/compare_probes.py probes/*/ --output report.json
        """,
    )
    parser.add_argument(
        "probe_dirs",
        nargs="+",
        help="Probe directories containing feature_labels.json",
    )
    parser.add_argument(
        "--output",
        help="Save JSON report to this file",
    )
    args = parser.parse_args()

    # Load all probes
    probes = []
    for d in args.probe_dirs:
        probe = load_probe_dir(d)
        if probe is not None:
            probes.append(probe)

    if not probes:
        print("Error: no valid probe directories found.", file=sys.stderr)
        sys.exit(1)

    print(f"Loaded {len(probes)} probe(s): {[p['name'] for p in probes]}\n")

    # Analyze each model
    analyses = [analyze_model(p) for p in probes]

    # Compute overlap
    overlap = compute_overlap(analyses)

    # Print formatted report
    report = format_report(analyses, overlap)
    print(report)

    # Optionally save JSON
    if args.output:
        json_report = build_json_report(analyses, overlap)
        with open(args.output, "w") as f:
            json.dump(json_report, f, indent=2)
        print(f"\nJSON report saved to: {args.output}")


if __name__ == "__main__":
    main()
