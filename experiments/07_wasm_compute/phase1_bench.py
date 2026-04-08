#!/usr/bin/env python3
"""
Phase 1: Benchmark Runner

Run GSM8K, MATH, SVAMP, AQuA-RAT. Compare baseline vs solver-augmented.
Measures:
  - Accuracy lift (% correct with solver vs without)
  - Intervention rate (fraction triggering solver)
  - False positive rate (solver dispatched but harmed the answer)
  - Latency overhead (ms per token added by solver pipeline)

Usage:
  # Full suite (all benchmarks, both modes)
  python phase1_bench.py --model google/gemma-3-4b-it

  # Single benchmark, quick test
  python phase1_bench.py --model google/gemma-3-4b-it --benchmark gsm8k --limit 50

  # Resume from checkpoint
  python phase1_bench.py --model google/gemma-3-4b-it --resume
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

from phase1_pipeline import (
    ComputePipeline, PipelineConfig, GenerationResult,
    extract_numeric_answer, normalize_answer,
)

OUTPUT_DIR = "results_phase1"


# ---------------------------------------------------------------------------
# Benchmark loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split: str = "test", limit: Optional[int] = None) -> List[dict]:
    """Load GSM8K benchmark."""
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split=split)
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        # Extract numeric answer from "#### <number>"
        answer = row["answer"].split("####")[-1].strip().replace(",", "")
        items.append({
            "id": f"gsm8k_{i}",
            "question": row["question"],
            "answer": answer,
            "solution": row["answer"],
        })
    return items


def load_math(split: str = "test", limit: Optional[int] = None) -> List[dict]:
    """Load MathQA benchmark (accessible alternative to MATH)."""
    from datasets import load_dataset
    ds = load_dataset("math_qa", split=split, trust_remote_code=True)
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        # Format options like AQuA
        options_str = row["options"]
        question = f"{row['Problem']}\n\nOptions:\n{options_str}"
        items.append({
            "id": f"math_{i}",
            "question": question,
            "answer": row["correct"],  # letter: a, b, c, d, e
            "rationale": row.get("Rationale", ""),
            "category": row.get("category", ""),
        })
    return items


def load_svamp(limit: Optional[int] = None) -> List[dict]:
    """Load SVAMP benchmark."""
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


def load_aqua(limit: Optional[int] = None) -> List[dict]:
    """Load AQuA-RAT benchmark."""
    from datasets import load_dataset
    ds = load_dataset("deepmind/aqua_rat", "raw", split="test")
    items = []
    for i, row in enumerate(ds):
        if limit and i >= limit:
            break
        # Format options
        options_str = "\n".join(row["options"])
        question = f"{row['question']}\n\nOptions:\n{options_str}"
        items.append({
            "id": f"aqua_{i}",
            "question": question,
            "answer": row["correct"],  # letter: A, B, C, D, E
            "rationale": row.get("rationale", ""),
        })
    return items


def extract_boxed(text: str) -> Optional[str]:
    """Extract \\boxed{...} from MATH solutions."""
    # Find the last \boxed{...}
    pattern = r'\\boxed\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}'
    matches = re.findall(pattern, text)
    if matches:
        return matches[-1]
    return None


BENCHMARK_LOADERS = {
    "gsm8k": load_gsm8k,
    "math": load_math,
    "svamp": load_svamp,
    "aqua": load_aqua,
}


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

def format_prompt(benchmark: str, question: str) -> str:
    """Format question with appropriate prompt template."""
    if benchmark == "gsm8k":
        return (
            f"Solve this math problem step by step. "
            f"End your answer with #### followed by the numeric answer.\n\n"
            f"{question}"
        )
    elif benchmark == "math":
        return (
            f"Solve this problem and select the correct option (a, b, c, d, or e). "
            f"State your answer as a single letter.\n\n{question}"
        )
    elif benchmark == "svamp":
        return (
            f"Solve this word problem. "
            f"End your answer with #### followed by the numeric answer.\n\n"
            f"{question}"
        )
    elif benchmark == "aqua":
        return (
            f"Solve this problem and select the correct option (A, B, C, D, or E). "
            f"State your answer as a single letter.\n\n{question}"
        )
    else:
        return question


# ---------------------------------------------------------------------------
# Answer comparison
# ---------------------------------------------------------------------------

def answers_match(predicted: str, expected: str, benchmark: str) -> bool:
    """Compare predicted vs expected answer."""
    if not predicted or not expected:
        return False

    if benchmark in ("aqua", "math"):
        # Letter matching
        pred_letter = re.search(r'[A-E]', predicted.upper())
        return pred_letter is not None and pred_letter.group() == expected.upper()

    # Numeric comparison
    pred_norm = normalize_answer(predicted)
    exp_norm = normalize_answer(expected)

    if pred_norm == exp_norm:
        return True

    # Try float comparison with tolerance
    try:
        pred_f = float(pred_norm)
        exp_f = float(exp_norm)
        return abs(pred_f - exp_f) < 1e-4 * max(1, abs(exp_f))
    except ValueError:
        return pred_norm == exp_norm


# ---------------------------------------------------------------------------
# Benchmark runner
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    benchmark: str
    mode: str  # "baseline" or "augmented"
    total: int = 0
    correct: int = 0
    interventions: int = 0           # total solver interventions
    items_with_intervention: int = 0  # items where solver fired at least once
    total_solver_us: float = 0       # total solver compute time
    total_generation_ms: float = 0   # total generation time
    errors: List[dict] = field(default_factory=list)

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total > 0 else 0

    @property
    def intervention_rate(self) -> float:
        return self.items_with_intervention / self.total if self.total > 0 else 0

    @property
    def avg_solver_us(self) -> float:
        return self.total_solver_us / self.interventions if self.interventions > 0 else 0

    @property
    def avg_generation_ms(self) -> float:
        return self.total_generation_ms / self.total if self.total > 0 else 0


def run_benchmark(
    pipeline: ComputePipeline,
    benchmark: str,
    items: List[dict],
    mode: str,
    checkpoint_dir: str,
) -> BenchmarkResult:
    """Run a benchmark in baseline or augmented mode."""
    result = BenchmarkResult(benchmark=benchmark, mode=mode)

    # Load checkpoint if exists
    checkpoint_file = os.path.join(checkpoint_dir, f"{benchmark}_{mode}_checkpoint.jsonl")
    completed_ids = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            for line in f:
                item = json.loads(line)
                completed_ids.add(item["id"])
                _update_result(result, item)
        print(f"  Resumed from checkpoint: {len(completed_ids)} items done")

    pipeline.config.solver_enabled = (mode == "augmented")

    for i, item in enumerate(items):
        if item["id"] in completed_ids:
            continue

        prompt = format_prompt(benchmark, item["question"])

        try:
            gen = pipeline.generate(prompt)
        except Exception as e:
            print(f"  ERROR on {item['id']}: {e}")
            continue

        # Extract and compare answer
        if benchmark in ("aqua", "math"):
            predicted = gen.output  # extract letter directly
        else:
            predicted = extract_numeric_answer(gen.output)

        correct = answers_match(
            predicted or "", item["answer"], benchmark
        )

        record = {
            "id": item["id"],
            "question": item["question"][:200],
            "expected": item["answer"],
            "predicted": predicted,
            "correct": correct,
            "tokens": gen.tokens_generated,
            "time_ms": gen.total_time_ms,
            "interventions": [asdict(iv) for iv in gen.interventions],
            "output_snippet": gen.output[:300],
        }

        _update_result(result, record)

        # Checkpoint
        with open(checkpoint_file, "a") as f:
            f.write(json.dumps(record) + "\n")

        # Progress — flush every item for visibility
        if (i + 1) % 5 == 0 or (i + 1) == len(items) or result.total <= 3:
            print(f"  [{mode}] {result.correct}/{result.total} correct "
                  f"({result.accuracy:.1%}), "
                  f"{result.items_with_intervention} interventions, "
                  f"avg {result.avg_generation_ms:.0f}ms/item",
                  flush=True)

    return result


def _update_result(result: BenchmarkResult, record: dict):
    """Update running totals from a record."""
    result.total += 1
    if record["correct"]:
        result.correct += 1
    else:
        result.errors.append(record)
    ivs = record.get("interventions", [])
    result.interventions += len(ivs)
    if ivs:
        result.items_with_intervention += 1
    for iv in ivs:
        result.total_solver_us += iv.get("solver_us", 0)
    result.total_generation_ms += record.get("time_ms", 0)


# ---------------------------------------------------------------------------
# Error analysis
# ---------------------------------------------------------------------------

def analyze_errors(baseline: BenchmarkResult, augmented: BenchmarkResult) -> dict:
    """Compare errors between baseline and augmented."""
    baseline_wrong = {e["id"] for e in baseline.errors}
    augmented_wrong = {e["id"] for e in augmented.errors}

    fixed_by_solver = baseline_wrong - augmented_wrong
    broken_by_solver = augmented_wrong - baseline_wrong
    both_wrong = baseline_wrong & augmented_wrong

    return {
        "fixed_by_solver": len(fixed_by_solver),
        "broken_by_solver": len(broken_by_solver),
        "both_wrong": len(both_wrong),
        "baseline_only_wrong": len(baseline_wrong - both_wrong),
        "augmented_only_wrong": len(augmented_wrong - both_wrong),
        "false_positive_rate": len(broken_by_solver) / augmented.total if augmented.total > 0 else 0,
    }


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def generate_report(
    results: Dict[str, Tuple[BenchmarkResult, BenchmarkResult]],
    output_dir: str,
):
    """Generate summary report."""
    os.makedirs(output_dir, exist_ok=True)
    lines = []
    lines.append("# Phase 1 Results: Token-Level Compute Engine\n")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M')}\n")

    # Summary table
    lines.append("## Summary\n")
    lines.append("| Benchmark | Baseline | Augmented | Lift | Intervention Rate | FP Rate |")
    lines.append("|-----------|----------|-----------|------|-------------------|---------|")

    for bench_name, (baseline, augmented) in results.items():
        analysis = analyze_errors(baseline, augmented)
        lift = augmented.accuracy - baseline.accuracy
        lines.append(
            f"| {bench_name} | {baseline.accuracy:.1%} ({baseline.correct}/{baseline.total}) "
            f"| {augmented.accuracy:.1%} ({augmented.correct}/{augmented.total}) "
            f"| {lift:+.1%} "
            f"| {augmented.intervention_rate:.1%} "
            f"| {analysis['false_positive_rate']:.1%} |"
        )

    # Per-benchmark details
    for bench_name, (baseline, augmented) in results.items():
        analysis = analyze_errors(baseline, augmented)
        lines.append(f"\n## {bench_name}\n")
        lines.append(f"- Baseline accuracy: {baseline.accuracy:.1%} ({baseline.correct}/{baseline.total})")
        lines.append(f"- Augmented accuracy: {augmented.accuracy:.1%} ({augmented.correct}/{augmented.total})")
        lines.append(f"- **Accuracy lift: {augmented.accuracy - baseline.accuracy:+.1%}**")
        lines.append(f"- Problems fixed by solver: {analysis['fixed_by_solver']}")
        lines.append(f"- Problems broken by solver: {analysis['broken_by_solver']}")
        lines.append(f"- Intervention rate: {augmented.intervention_rate:.1%}")
        lines.append(f"- Avg solver latency: {augmented.avg_solver_us:.1f}μs")
        lines.append(f"- Avg generation time: baseline {baseline.avg_generation_ms:.0f}ms, "
                      f"augmented {augmented.avg_generation_ms:.0f}ms")

    # Success criteria check
    lines.append("\n## Success Criteria\n")
    max_lift = max(
        (aug.accuracy - base.accuracy)
        for base, aug in results.values()
    ) if results else 0
    min_fp = min(
        analyze_errors(base, aug)["false_positive_rate"]
        for base, aug in results.values()
    ) if results else 1
    max_latency_overhead = max(
        (aug.avg_generation_ms - base.avg_generation_ms)
        for base, aug in results.values()
    ) if results else 0

    lines.append(f"- [{'x' if max_lift > 0.05 else ' '}] Accuracy lift >5% on at least one benchmark: "
                 f"{max_lift:+.1%}")
    lines.append(f"- [{'x' if min_fp < 0.02 else ' '}] False positive rate <2%: {min_fp:.1%}")
    lines.append(f"- [{'x' if max_latency_overhead < 5 else ' '}] Latency overhead <5ms/token: "
                 f"{max_latency_overhead:.1f}ms")

    phase1_pass = max_lift > 0.05 and min_fp < 0.02
    lines.append(f"\n**Phase 1 {'PASSES' if phase1_pass else 'DOES NOT PASS'} — "
                 f"{'proceed to Phase 2' if phase1_pass else 'investigate before Phase 2'}**")

    report_text = "\n".join(lines) + "\n"

    report_path = os.path.join(output_dir, "RESULTS.md")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")

    # Also save raw JSON
    raw = {}
    for bench_name, (baseline, augmented) in results.items():
        analysis = analyze_errors(baseline, augmented)
        raw[bench_name] = {
            "baseline": {
                "accuracy": baseline.accuracy,
                "correct": baseline.correct,
                "total": baseline.total,
                "avg_ms": baseline.avg_generation_ms,
            },
            "augmented": {
                "accuracy": augmented.accuracy,
                "correct": augmented.correct,
                "total": augmented.total,
                "intervention_rate": augmented.intervention_rate,
                "avg_solver_us": augmented.avg_solver_us,
                "avg_ms": augmented.avg_generation_ms,
            },
            "analysis": analysis,
        }

    json_path = os.path.join(output_dir, "results.json")
    with open(json_path, "w") as f:
        json.dump(raw, f, indent=2)
    print(f"Raw results saved to {json_path}")

    return phase1_pass


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1: Benchmark Runner")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="HF model name")
    parser.add_argument("--benchmark", default="all",
                        choices=["all", "gsm8k", "math", "svamp", "aqua"],
                        help="Which benchmark to run")
    parser.add_argument("--limit", type=int, default=None,
                        help="Limit items per benchmark (for quick testing)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cpu")
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    config = PipelineConfig(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )
    pipeline = ComputePipeline(config)

    benchmarks = (
        list(BENCHMARK_LOADERS.keys())
        if args.benchmark == "all"
        else [args.benchmark]
    )

    results = {}
    for bench_name in benchmarks:
        print(f"\n{'='*60}")
        print(f"Benchmark: {bench_name}")
        print(f"{'='*60}")

        loader = BENCHMARK_LOADERS[bench_name]
        if bench_name in ("gsm8k", "math"):
            items = loader(split="test", limit=args.limit)
        else:
            items = loader(limit=args.limit)

        print(f"Loaded {len(items)} items", flush=True)

        # Clear checkpoints if not resuming
        if not args.resume:
            for mode in ("baseline", "augmented"):
                cp = os.path.join(args.output_dir, f"{bench_name}_{mode}_checkpoint.jsonl")
                if os.path.exists(cp):
                    os.remove(cp)

        # Run baseline
        print(f"\nRunning baseline...", flush=True)
        baseline = run_benchmark(pipeline, bench_name, items, "baseline", args.output_dir)
        print(f"Baseline: {baseline.accuracy:.1%} ({baseline.correct}/{baseline.total})")

        # Run augmented
        print(f"\nRunning augmented...", flush=True)
        augmented = run_benchmark(pipeline, bench_name, items, "augmented", args.output_dir)
        print(f"Augmented: {augmented.accuracy:.1%} ({augmented.correct}/{augmented.total})")

        lift = augmented.accuracy - baseline.accuracy
        print(f"Lift: {lift:+.1%}")

        results[bench_name] = (baseline, augmented)

    # Generate report
    phase1_pass = generate_report(results, args.output_dir)

    return 0 if phase1_pass else 1


if __name__ == "__main__":
    sys.exit(main())
