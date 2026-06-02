#!/usr/bin/env python3
"""
LQL distillation script.

Reads a JSON corpus of (prompt, first_token) pairs and installs LQL
statement-type knowledge into a vindex using the training-free insert
method validated in docs/training-free-insert.md.

Usage:
    python lql_distill.py --vindex /path/to/vindex --corpus lql_corpus.json
    python lql_distill.py --vindex /path/to/vindex --corpus lql_corpus.json \\
        --alpha 0.25 --layers 20-27 --split train --dry-run

Formula (from training-free-insert.md, validated at 94.6% confidence):
    gate_vec = residual[layer] * (avg_gate_norm / norm(residual))
    down_vec = embed(target_token) * embed_scale * alpha
    insert at layers 20-27, alpha=0.25 per layer (8 layers total)
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
    embed_scale = vindex.embed_scale()  # sqrt(hidden_size), e.g. 50.6 for Gemma-3 4B

    results = []
    for i, entry in enumerate(corpus):
        prompt = entry["prompt"]
        first_token = entry["first_token"]
        statement_type = entry.get("statement_type", "?")

        print(f"[{i+1}/{len(corpus)}] {statement_type:20s} {prompt[:55]}", end=" ... ", flush=True)

        # Capture residuals via forward pass with trace
        _, residuals = vindex.infer_trace(prompt)
        residuals_by_layer = dict(residuals)

        # Scale gate vecs to match existing gate vector norms
        all_norms = [np.linalg.norm(v) for _, v in residuals]
        avg_norm = float(np.mean(all_norms)) if all_norms else 1.0

        # Target token embedding direction
        token_embed = vindex.embed(first_token)

        if not dry_run:
            for layer in range(layer_start, layer_end):
                if layer not in residuals_by_layer:
                    continue
                r = residuals_by_layer[layer]
                r_norm = float(np.linalg.norm(r))
                if r_norm < 1e-8:
                    continue

                gate_vec = r * (avg_norm / r_norm)
                down_vec = token_embed * embed_scale * alpha

                feat = vindex.find_free_feature(layer)
                vindex.set_gate_vector(layer, feat, gate_vec)
                vindex.set_down_vector(layer, feat, down_vec)
                vindex.set_feature_meta(layer, feat, first_token, 0.9)

        # Verify: does the inserted token appear in top-5?
        top_preds = vindex.infer(prompt)
        top5_tokens = [p[0] for p in top_preds[:5]]
        # Match on prefix: "DESCRIBE" may tokenize as "▁DESCRIBE" etc.
        success = any(first_token.upper() in t.upper() for t in top5_tokens)

        status = "OK  " if success else "FAIL"
        if dry_run:
            status = "DRY "
        print(f"{status}  top5={top5_tokens[:3]}")

        results.append({
            "prompt": prompt,
            "first_token": first_token,
            "statement_type": statement_type,
            "category": entry.get("category", "?"),
            "success": success,
            "top5": top5_tokens,
        })

    n = len(results)
    ok = sum(r["success"] for r in results)
    print(f"\nTop-5 accuracy: {ok/n:.1%} ({ok}/{n})")

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
