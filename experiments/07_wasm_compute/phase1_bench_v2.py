#!/usr/bin/env python3
"""
Phase 1 v2: Benchmark Runner

Uses post-hoc correction instead of mid-generation injection.
Three modes compared:
  1. Baseline — standard generation, extract answer
  2. Corrected — same generation, scan for arithmetic errors, correct chain
  3. (same generation, just different answer extraction)

This means we only generate ONCE per item, then compare raw vs corrected answers.
Half the compute of v1.

Usage:
  python phase1_bench_v2.py --model google/gemma-3-4b-it --benchmark gsm8k --limit 50
  python phase1_bench_v2.py --resume  # resume from checkpoint
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from phase1_pipeline_v2 import (
    PostHocPipeline, PipelineConfig, CorrectionResult,
    scan_arithmetic, correct_chain, rederive_answer,
    extract_numeric_answer, normalize_answer,
)

OUTPUT_DIR = "results_phase1_v2"


# ---------------------------------------------------------------------------
# Benchmark loaders (same as v1)
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "test", limit: Optional[int] = None) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": answer,
            "solution": row["answer"],
        })
    return items


def load_svamp(limit: Optional[int] = None) -> List[dict]:
    from datasets import load_dataset
    ds = load_dataset("ChilleD/SVAMP", split="test")
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        items.append({
            "id": f"svamp_{i}",
            "question": row["Body"] + " " + row["Question"],
            "answer": str(row["Answer"]).replace(".0", ""),
            "equation": row.get("Equation", ""),
        })
    return items


BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "svamp": load_svamp,
}


# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

def format_prompt(benchmark: str, question: str) -> str:
    if benchmark == "gsm8k":
        return (
            f"Solve this math problem step by step. "
            f"End your answer with #### followed by the numeric answer.\n\n"
            f"{question}"
        )
    elif benchmark == "svamp":
        return (
            f"Solve this word problem. "
            f"End your answer with #### followed by the numeric answer.\n\n"
            f"{question}"
        )
    return question


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def answers_match(predicted: str, expected: str) -> bool:
    if not predicted or not expected:
        return False
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)
    if pred_norm == exp_norm:
        return True
    try:
        pred_f = float(pred_norm)
        exp_f = float(exp_norm)
        return abs(pred_f - exp_f) < 1e-4 * max(1, abs(exp_f))
    except ValueError:
        return pred_norm == exp_norm


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

def run_benchmark(
    pipeline: PostHocPipeline,
    benchmark: str,
    items: List[dict],
    checkpoint_dir: str,
) -> List[dict]:
    """Run benchmark. Single generation per item, compare raw vs corrected."""
    checkpoint_file = os.path.join(checkpoint_dir, f"{benchmark}_checkpoint.jsonl")
    completed_ids = set()
    records = []

    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            for line in f:
                rec = json.loads(line)
                completed_ids.add(rec["id"])
                records.append(rec)
        print(f"  Resumed: {len(completed_ids)} items done", flush=True)

    for i, item in enumerate(items):
        if item["id"] in completed_ids:
            continue

        prompt = format_prompt(benchmark, item["question"])

        try:
            result = pipeline.generate(prompt)
        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}", flush=True)
            continue

        raw_correct = answers_match(result.raw_answer or "", item["answer"])
        corrected_correct = answers_match(result.corrected_answer or "", item["answer"])

        record = {
            "id": item["id"],
            "question": item["question"][:200],
            "expected": item["answer"],
            "raw_predicted": result.raw_answer,
            "corrected_predicted": result.corrected_answer,
            "raw_correct": raw_correct,
            "corrected_correct": corrected_correct,
            "tokens": result.tokens_generated,
            "generation_ms": result.generation_ms,
            "correction_ms": result.correction_ms,
            "n_checks": len(result.checks),
            "n_corrections": len(result.corrections),
            "corrections": [
                {"expr": c.expression, "claimed": c.claimed, "correct": c.correct}
                for c in result.corrections
            ],
            "full_output": result.raw_output,
            "corrected_output": result.corrected_output if result.corrections else None,
        }
        records.append(record)

        # Checkpoint
        with open(checkpoint_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Progress
        done = len(records)
        raw_acc = sum(1 for r in records if r["raw_correct"]) / done
        corr_acc = sum(1 for r in records if r["corrected_correct"]) / done
        items_w_corrections = sum(1 for r in records if r["n_corrections"] > 0)

        if done % 5 == 0 or done <= 3 or done == len(items):
            print(
                f"  [{done}/{len(items)}] "
                f"raw={raw_acc:.1%} corrected={corr_acc:.1%} "
                f"fixes={items_w_corrections} "
                f"avg={sum(r['generation_ms'] for r in records)/done:.0f}ms",
                flush=True
            )

    return records


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def generate_report(benchmark: str, records: List[dict], output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    total = len(records)
    raw_correct = sum(1 for r in records if r["raw_correct"])
    corr_correct = sum(1 for r in records if r["corrected_correct"])
    items_with_arith = sum(1 for r in records if r["n_checks"] > 0)
    items_with_errors = sum(1 for r in records if r["n_corrections"] > 0)
    total_corrections = sum(r["n_corrections"] for r in records)

    # Categorise
    fixed = [r for r in records if not r["raw_correct"] and r["corrected_correct"]]
    broken = [r for r in records if r["raw_correct"] and not r["corrected_correct"]]
    both_right = [r for r in records if r["raw_correct"] and r["corrected_correct"]]
    both_wrong = [r for r in records if not r["raw_correct"] and not r["corrected_correct"]]

    raw_acc = raw_correct / total if total else 0
    corr_acc = corr_correct / total if total else 0
    lift = corr_acc - raw_acc
    fp_rate = len(broken) / total if total else 0

    lines = []
    lines.append(f"# Phase 1 v2 Results: Post-Hoc Correction — {benchmark}\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")

    lines.append("## Summary\n")
    lines.append(f"| Metric | Value |")
    lines.append(f"|--------|-------|")
    lines.append(f"| Items | {total} |")
    lines.append(f"| Raw accuracy | {raw_acc:.1%} ({raw_correct}/{total}) |")
    lines.append(f"| Corrected accuracy | {corr_acc:.1%} ({corr_correct}/{total}) |")
    lines.append(f"| **Accuracy lift** | **{lift:+.1%}** |")
    lines.append(f"| Items with arithmetic | {items_with_arith} |")
    lines.append(f"| Items with errors found | {items_with_errors} |")
    lines.append(f"| Total corrections | {total_corrections} |")
    lines.append(f"| Fixed by solver | {len(fixed)} |")
    lines.append(f"| Broken by solver | {len(broken)} |")
    lines.append(f"| False positive rate | {fp_rate:.1%} |")

    # Hit token limit analysis
    hit_limit = sum(1 for r in records if r["tokens"] >= 510)
    wrong_hit_limit = sum(1 for r in records if not r["raw_correct"] and r["tokens"] >= 510)
    lines.append(f"| Hit token limit | {hit_limit} |")
    lines.append(f"| Wrong + hit limit | {wrong_hit_limit} |")

    lines.append(f"\n## Success Criteria\n")
    lines.append(f"- [{'x' if lift > 0.05 else ' '}] Accuracy lift >5%: {lift:+.1%}")
    lines.append(f"- [{'x' if fp_rate < 0.02 else ' '}] False positive rate <2%: {fp_rate:.1%}")

    if fixed:
        lines.append(f"\n## Fixed by Solver\n")
        for r in fixed:
            lines.append(f"- **{r['id']}**: expected={r['expected']}, "
                         f"raw={r['raw_predicted']}, corrected={r['corrected_predicted']}")
            for c in r["corrections"]:
                lines.append(f"  - `{c['expr']} = {c['claimed']}` → `{c['correct']}`")

    if broken:
        lines.append(f"\n## Broken by Solver\n")
        for r in broken:
            lines.append(f"- **{r['id']}**: expected={r['expected']}, "
                         f"raw={r['raw_predicted']}, corrected={r['corrected_predicted']}")
            for c in r["corrections"]:
                lines.append(f"  - `{c['expr']} = {c['claimed']}` → `{c['correct']}`")

    # Error analysis: why are things still wrong?
    lines.append(f"\n## Error Analysis\n")
    lines.append(f"Wrong answers remaining: {len(both_wrong)}\n")
    for r in both_wrong[:10]:
        lines.append(f"- **{r['id']}**: expected={r['expected']}, predicted={r['raw_predicted']}, "
                     f"tokens={r['tokens']}, arith_checks={r['n_checks']}")

    report_text = "\n".join(lines) + "\n"
    report_path = os.path.join(output_dir, "RESULTS.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport: {report_path}", flush=True)

    # Raw JSON
    json_path = os.path.join(output_dir, "results.json")
    summary = {
        "benchmark": benchmark,
        "total": total,
        "raw_accuracy": raw_acc,
        "corrected_accuracy": corr_acc,
        "lift": lift,
        "fixed": len(fixed),
        "broken": len(broken),
        "fp_rate": fp_rate,
    }
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    return lift > 0.05


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 v2: Benchmark Runner")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--benchmark", default="gsm8k", choices=["gsm8k", "svamp"])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = PipelineConfig(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )
    pipeline = PostHocPipeline(config)

    loader = BENCHMARK_LOADERS[args.benchmark]
    if args.benchmark == "gsm8k":
        items = loader(split="test", limit=args.limit)
    else:
        items = loader(limit=args.limit)
    print(f"Loaded {len(items)} {args.benchmark} items", flush=True)

    if not args.resume:
        cp = os.path.join(args.output_dir, f"{args.benchmark}_checkpoint.jsonl")
        if os.path.exists(cp):
            os.remove(cp)

    records = run_benchmark(pipeline, args.benchmark, items, args.output_dir)

    passed = generate_report(args.benchmark, records, args.output_dir)
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
