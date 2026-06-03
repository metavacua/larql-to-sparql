#!/usr/bin/env python3
"""
LQL evaluation script — read-only, no vindex modification.

Measures top-1 and top-5 accuracy of a vindex on the LQL corpus.
Run this before distillation (baseline) and after (post-distill) to
measure the gain. Also use as a regression check: existing knowledge
(France→Paris, Germany→Berlin) must be preserved after INSERT.

Usage:
    # Baseline (before distillation)
    python lql_eval.py --vindex /path/to/vindex --corpus lql_corpus.json

    # Eval split only
    python lql_eval.py --vindex /path/to/vindex --corpus lql_corpus.json --split eval

    # Include sanity checks for existing model knowledge
    python lql_eval.py --vindex /path/to/vindex --corpus lql_corpus.json --sanity
"""
import argparse
import json
import sys


SANITY_CHECKS = [
    {"prompt": "The capital of France is", "expected_contains": "Paris",
     "description": "France → Paris"},
    {"prompt": "The capital of Germany is", "expected_contains": "Berlin",
     "description": "Germany → Berlin"},
    {"prompt": "The capital of Japan is", "expected_contains": "Tokyo",
     "description": "Japan → Tokyo"},
    {"prompt": "The largest planet in our solar system is", "expected_contains": "Jupiter",
     "description": "largest planet → Jupiter"},
]


def eval_corpus(vindex, corpus, split):
    if split != "all":
        corpus = [e for e in corpus if e.get("split", "train") == split]

    if not corpus:
        print(f"No entries for split={split!r}", file=sys.stderr)
        return [], 0.0, 0.0

    results = []
    for entry in corpus:
        top_preds = vindex.infer(entry["prompt"])
        top1 = top_preds[0][0] if top_preds else ""
        top5 = [p[0] for p in top_preds[:5]]
        ft = entry["first_token"].upper()

        # Resolve the actual first token the model should produce.
        # Llama-family tokenizers often represent keywords as single tokens with
        # a leading space. The model's top-5 tokens should contain that.
        tids_space = vindex.tokenize(" " + entry["first_token"])
        tids_bare = vindex.tokenize(entry["first_token"])
        if len(tids_space) == 1:
            target_decoded = vindex.decode([tids_space[0]]).strip()
        elif len(tids_bare) == 1:
            target_decoded = vindex.decode([tids_bare[0]]).strip()
        else:
            # Multi-token: fall back to substring match on full keyword
            target_decoded = ft

        hit1 = target_decoded.upper() in top1.upper() or ft in top1.upper()
        hit5 = any(target_decoded.upper() in t.upper() or ft in t.upper() for t in top5)
        results.append({**entry, "top1": top1, "top5": top5, "hit1": hit1, "hit5": hit5,
                        "target_decoded": target_decoded})

    n = len(results)
    top1_acc = sum(r["hit1"] for r in results) / n
    top5_acc = sum(r["hit5"] for r in results) / n
    return results, top1_acc, top5_acc


def run_sanity(vindex):
    print("\nSanity checks (existing model knowledge must be preserved):")
    all_pass = True
    for check in SANITY_CHECKS:
        top_preds = vindex.infer(check["prompt"])
        top1 = top_preds[0][0] if top_preds else ""
        ok = check["expected_contains"].lower() in top1.lower()
        status = "OK  " if ok else "FAIL"
        if not ok:
            all_pass = False
        top3 = [p[0] for p in top_preds[:3]]
        print(f"  {status}  {check['description']:30s}  top1={top1!r}  top3={top3}")
    return all_pass


def main():
    ap = argparse.ArgumentParser(description="Evaluate LQL statement-type accuracy of a vindex")
    ap.add_argument("--vindex", required=True, help="Path to vindex directory")
    ap.add_argument("--corpus", required=True, help="Path to lql_corpus.json")
    ap.add_argument("--split", choices=["train", "eval", "all"], default="all",
                    help="Which corpus split to evaluate (default: all)")
    ap.add_argument("--sanity", action="store_true",
                    help="Also run sanity checks for existing factual knowledge")
    ap.add_argument("--show-failures", action="store_true",
                    help="Print each failed entry")
    args = ap.parse_args()

    try:
        import larql
    except ImportError:
        print("ERROR: larql not found. Build it first:", file=sys.stderr)
        print("  cd crates/larql-python && uv run maturin develop --release", file=sys.stderr)
        sys.exit(1)

    vindex = larql.load_vindex(args.vindex)

    with open(args.corpus) as f:
        corpus = json.load(f)

    print(f"Vindex: {args.vindex}")
    print(f"Corpus: {args.corpus}  split={args.split}")
    print()

    results, top1_acc, top5_acc = eval_corpus(vindex, corpus, args.split)
    if not results:
        sys.exit(1)

    n = len(results)
    print(f"Overall  top-1: {top1_acc:.1%}  top-5: {top5_acc:.1%}  (n={n})")

    # Per-category breakdown
    categories = sorted({r["category"] for r in results})
    print()
    print(f"{'category':16s}  {'top-1':>6}  {'top-5':>6}  n")
    print("-" * 45)
    for cat in categories:
        cr = [r for r in results if r["category"] == cat]
        c1 = sum(r["hit1"] for r in cr) / len(cr)
        c5 = sum(r["hit5"] for r in cr) / len(cr)
        print(f"{cat:16s}  {c1:>6.0%}  {c5:>6.0%}  {len(cr)}")

    # Per-statement-type breakdown
    print()
    print(f"{'statement_type':22s}  {'top-1':>6}  {'top-5':>6}")
    print("-" * 40)
    types = sorted({r["statement_type"] for r in results})
    for st in types:
        sr = [r for r in results if r["statement_type"] == st]
        s1 = sum(r["hit1"] for r in sr) / len(sr)
        s5 = sum(r["hit5"] for r in sr) / len(sr)
        mark = "" if s5 >= 0.8 else "  ← needs work"
        print(f"{st:22s}  {s1:>6.0%}  {s5:>6.0%}{mark}")

    if args.show_failures:
        failures = [r for r in results if not r["hit5"]]
        if failures:
            print(f"\nTop-5 failures ({len(failures)}):")
            for r in failures:
                print(f"  [{r['statement_type']}] {r['prompt'][:60]}")
                print(f"    expected first_token={r['first_token']!r}")
                print(f"    top5={r['top5'][:3]}")

    if args.sanity:
        ok = run_sanity(vindex)
        if not ok:
            print("\nERROR: sanity checks failed — existing knowledge was corrupted by INSERT")
            sys.exit(2)

    # Exit code: 0 = pass (top5 ≥ 80%), 1 = fail
    sys.exit(0 if top5_acc >= 0.80 else 1)


if __name__ == "__main__":
    main()
