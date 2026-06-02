#!/usr/bin/env python3
"""
LQL distillation script — Architecture B (KNN store).

Reads a JSON corpus of (prompt, first_token) pairs and installs LQL
statement-type knowledge into a vindex using the Architecture B
retrieval-override KNN store (vindex.knn_add).

The KNN store works differently from the compose/FFN-slot approach:
  - Stores (layer, residual_key, target_token) entries
  - At inference, checks cosine(residual, stored_key) > 0.75
  - If threshold met, overrides the model's top-1 prediction
  - No dependence on free-slot FFN activations
  - Scales to N entries with no cross-fact interference

This is the Architecture B path validated at 25K edges, 87 edges/s,
100% same-prompt retrieval (see crates/larql-lql/src/executor/mutation/insert/knn.rs).

Usage:
    python lql_distill.py --vindex /path/to/vindex --corpus lql_corpus.json
    python lql_distill.py --vindex /path/to/vindex --corpus lql_corpus.json \\
        --layers 20-27 --split train --dry-run
"""
import argparse
import json
import sys

import numpy as np


def parse_layer_range(s):
    parts = s.split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(f"layer range must be START-END, got {s!r}")
    return int(parts[0]), int(parts[1])


def distill(vindex_path, corpus, alpha, layer_start, layer_end, dry_run):
    try:
        import larql
    except ImportError:
        print("ERROR: larql not found. Build it first:", file=sys.stderr)
        print("  cd crates/larql-python && uv run maturin develop --release", file=sys.stderr)
        sys.exit(1)

    vindex = larql.load_vindex(vindex_path)

    results = []
    for i, entry in enumerate(corpus):
        prompt = entry["prompt"]
        first_token = entry["first_token"]
        statement_type = entry.get("statement_type", "?")

        print(f"[{i+1}/{len(corpus)}] {statement_type:20s} {prompt[:50]}", end=" ... ", flush=True)

        # Capture residuals via forward pass with trace
        _, residuals = vindex.infer_trace(prompt)
        residuals_by_layer = dict(residuals)

        if not dry_run:
            # Architecture B: store residual keys in KNN store.
            # The inference path checks cosine(residual, key) > 0.75
            # and overrides the model's top-1 prediction with first_token.
            # Install at every layer in the range — whichever fires
            # first at inference wins.
            inserted = 0
            for layer in range(layer_start, layer_end):
                if layer not in residuals_by_layer:
                    continue
                r = np.array(residuals_by_layer[layer], dtype=np.float32)
                r_norm = float(np.linalg.norm(r))
                if r_norm < 1e-8:
                    continue
                # L2-normalize the key (KNN store normalizes internally too,
                # but being explicit ensures consistent behavior)
                key = r / r_norm
                vindex.knn_add(
                    layer, key, first_token,
                    entity=statement_type, relation="lql-statement",
                )
                inserted += 1
            print(f"({inserted} keys)", end=" ", flush=True)

        # Verify: does knn_query at each layer find our key?
        # Also check if infer() returns our token (same-prompt retrieval).
        top_preds = vindex.infer(prompt)
        top1_token = top_preds[0][0] if top_preds else ""
        success = first_token.upper() in top1_token.upper()

        status = "OK  " if success else "FAIL"
        if dry_run:
            status = "DRY "
        top5_tokens = [p[0] for p in top_preds[:5]]
        print(f"{status}  top1={top1_token!r}")

        results.append({
            "prompt": prompt,
            "first_token": first_token,
            "statement_type": statement_type,
            "category": entry.get("category", "?"),
            "success": success,
            "top1": top1_token,
            "top5": top5_tokens,
        })

    n = len(results)
    ok = sum(r["success"] for r in results)
    print(f"\nTop-1 accuracy: {ok/n:.1%} ({ok}/{n})")

    # Per-category breakdown
    categories = sorted({r["category"] for r in results})
    for cat in categories:
        cat_results = [r for r in results if r["category"] == cat]
        cat_ok = sum(r["success"] for r in cat_results)
        print(f"  {cat:16s} {cat_ok/len(cat_results):.0%}  ({cat_ok}/{len(cat_results)})")

    return results


def main():
    ap = argparse.ArgumentParser(description="Distill LQL statement-type knowledge into a vindex")
    ap.add_argument("--vindex", required=True, help="Path to the vindex directory")
    ap.add_argument("--corpus", required=True, help="Path to lql_corpus.json")
    ap.add_argument("--alpha", type=float, default=0.25,
                    help="Down vector scale per layer (default: 0.25, validated sweet spot)")
    ap.add_argument("--layers", type=parse_layer_range, default=(20, 28),
                    metavar="START-END",
                    help="Layer range to insert into (default: 20-27 = 8 knowledge layers)")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="train",
                    help="Which corpus split to distill (default: train)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Run infer_trace and measure baseline accuracy without inserting")
    args = ap.parse_args()

    with open(args.corpus) as f:
        corpus = json.load(f)

    if args.split != "all":
        corpus = [e for e in corpus if e.get("split", "train") == args.split]

    if not corpus:
        print(f"No entries found for split={args.split!r}", file=sys.stderr)
        sys.exit(1)

    print(f"Corpus: {len(corpus)} entries (split={args.split})")
    print(f"Vindex: {args.vindex}")
    print(f"Alpha:  {args.alpha}  Layers: {args.layers[0]}-{args.layers[1]-1}  Dry-run: {args.dry_run}")
    print()

    results = distill(
        vindex_path=args.vindex,
        corpus=corpus,
        alpha=args.alpha,
        layer_start=args.layers[0],
        layer_end=args.layers[1],
        dry_run=args.dry_run,
    )

    # Failures
    failures = [r for r in results if not r["success"]]
    if failures:
        print(f"\nFailures ({len(failures)}):")
        for r in failures:
            print(f"  [{r['statement_type']}] {r['prompt'][:60]}")
            print(f"    expected first_token={r['first_token']!r}, top5={r['top5'][:3]}")


if __name__ == "__main__":
    main()
