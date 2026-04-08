#!/usr/bin/env python3
"""
Phase 1 v2: Post-Hoc Correction Pipeline

v1 failed because mid-generation injection corrupts the KV cache and
triggers on partial expressions. v2 takes a different approach:

  1. Let the model generate its FULL reasoning chain
  2. Scan the chain for arithmetic expressions and their claimed results
  3. Verify each computation with the solver
  4. If errors found, correct the chain and re-derive the final answer

This is verify-and-correct, not intercept-and-replace.

Usage:
  python phase1_pipeline_v2.py --model google/gemma-3-4b-it --prompt "What is 6 * 7?"
"""

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from phase1_parser import normalise
from phase1_solvers import safe_eval, solve, SolverResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    model_name: str = "google/gemma-3-4b-it"
    max_new_tokens: int = 256
    temperature: float = 0.0
    device: str = "cpu"
    dtype: str = "bfloat16"


# ---------------------------------------------------------------------------
# Arithmetic chain scanner
# ---------------------------------------------------------------------------

# Pattern: <expr> = <claimed_result>
# e.g., "3 * 60 = 180", "400 + 60 = 460", "16 - 3 - 4 = 9"
_CHAIN_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?'                        # first number
    r'(?:\s*[\+\-\*/]\s*\d+(?:\.\d+)?)+)'    # operator + number, repeated
    r'\s*=\s*'                                # equals sign
    r'(\-?\d+(?:,\d{3})*(?:\.\d+)?)'         # claimed result
)

# Percentage pattern: "X% of Y = Z"
_PCT_PATTERN = re.compile(
    r'(\d+(?:\.\d+)?)\s*%\s*(?:of)\s+(\d+(?:\.\d+)?)'
    r'\s*=\s*'
    r'(\-?\d+(?:,\d{3})*(?:\.\d+)?)',
    re.IGNORECASE,
)


@dataclass
class ArithCheck:
    expression: str       # "3 * 60"
    claimed: str          # "180"
    correct: str          # "180" (from solver)
    is_correct: bool
    position: int         # char offset in text


def scan_arithmetic(text: str) -> List[ArithCheck]:
    """Scan text for arithmetic expressions with claimed results.
    Verify each one."""
    checks = []
    normalised = normalise(text)

    # Check percentages
    for m in _PCT_PATTERN.finditer(normalised):
        pct, of_val, claimed = m.group(1), m.group(2), m.group(3)
        claimed_clean = claimed.replace(",", "")
        correct_val = float(pct) / 100.0 * float(of_val)
        if correct_val == int(correct_val):
            correct_str = str(int(correct_val))
        else:
            correct_str = str(correct_val)
        checks.append(ArithCheck(
            expression=f"{pct}% of {of_val}",
            claimed=claimed_clean,
            correct=correct_str,
            is_correct=_nums_equal(claimed_clean, correct_str),
            position=m.start(),
        ))

    # Check arithmetic expressions
    for m in _CHAIN_PATTERN.finditer(normalised):
        expr_str = m.group(1).strip()
        claimed = m.group(2).replace(",", "").strip()

        result = safe_eval(expr_str)
        if result is None:
            continue

        if result == int(result) and abs(result) < 1e15:
            correct_str = str(int(result))
        else:
            correct_str = str(result)

        checks.append(ArithCheck(
            expression=expr_str,
            claimed=claimed,
            correct=correct_str,
            is_correct=_nums_equal(claimed, correct_str),
            position=m.start(),
        ))

    return checks


def _nums_equal(a: str, b: str) -> bool:
    """Compare two numeric strings with tolerance."""
    try:
        fa, fb = float(a), float(b)
        if fa == fb:
            return True
        return abs(fa - fb) < 1e-4 * max(1, abs(fb))
    except ValueError:
        return a.strip() == b.strip()


# ---------------------------------------------------------------------------
# Chain correction
# ---------------------------------------------------------------------------

def correct_chain(text: str, checks: List[ArithCheck]) -> Tuple[str, List[ArithCheck]]:
    """Correct arithmetic errors in a reasoning chain.

    Returns corrected text and list of corrections made.
    """
    corrections = [c for c in checks if not c.is_correct]
    if not corrections:
        return text, []

    # Sort by position descending so replacements don't shift offsets
    normalised = normalise(text)
    corrected = normalised

    for check in sorted(corrections, key=lambda c: c.position, reverse=True):
        # Replace "expr = wrong" with "expr = correct"
        old = f"{check.expression} = {check.claimed}"
        new = f"{check.expression} = {check.correct}"
        # Only replace at/near the expected position
        idx = corrected.find(old, max(0, check.position - 10))
        if idx >= 0:
            corrected = corrected[:idx] + new + corrected[idx + len(old):]

    return corrected, corrections


# ---------------------------------------------------------------------------
# Final answer re-derivation
# ---------------------------------------------------------------------------

def rederive_answer(text: str, corrections: List[ArithCheck]) -> Optional[str]:
    """After correcting arithmetic errors, re-derive the final answer.

    Strategy:
    1. If the text has "#### <number>", recompute from the corrected chain
    2. If the last arithmetic expression was corrected, the final answer changes
    3. Otherwise, scan for the last number after correction
    """
    # GSM8K format
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        # Check if the #### answer matches any corrected expression's result
        stated_answer = m.group(1).replace(",", "")
        for c in corrections:
            if _nums_equal(stated_answer, c.claimed):
                # The final answer used the wrong intermediate value
                # Replace with corrected value
                return text[:m.start(1)] + c.correct + text[m.end(1):]
        return text  # #### answer wasn't affected

    # If corrections were applied, the last number in the text might have changed
    return text


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

@dataclass
class CorrectionResult:
    prompt: str
    raw_output: str               # original model output
    corrected_output: str         # after arithmetic corrections
    checks: List[ArithCheck]      # all arithmetic found
    corrections: List[ArithCheck] # only the wrong ones
    raw_answer: Optional[str]     # answer from raw output
    corrected_answer: Optional[str]  # answer after corrections
    tokens_generated: int
    generation_ms: float
    correction_ms: float


class PostHocPipeline:
    """Generate → Verify → Correct pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        print(f"Loading {self.config.model_name}...", flush=True)
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
        )
        self.model = self.model.to(self.config.device)
        self.model.eval()
        self._loaded = True
        print(f"Model loaded on {self.config.device}", flush=True)

    def generate(self, prompt: str) -> CorrectionResult:
        """Full pipeline: generate → scan → correct."""
        self.load()

        # Step 1: Generate normally
        t0 = time.perf_counter()
        raw_output, n_tokens = self._generate_text(prompt)
        gen_ms = (time.perf_counter() - t0) * 1000

        # Step 2: Scan for arithmetic and verify
        t1 = time.perf_counter()
        checks = scan_arithmetic(raw_output)

        # Step 3: Correct errors
        corrected_output, corrections = correct_chain(raw_output, checks)

        # Step 4: Re-derive answer if needed
        if corrections:
            corrected_output = rederive_answer(corrected_output, corrections) or corrected_output

        corr_ms = (time.perf_counter() - t1) * 1000

        # Extract answers
        raw_answer = extract_numeric_answer(raw_output)
        corrected_answer = extract_numeric_answer(corrected_output)

        return CorrectionResult(
            prompt=prompt,
            raw_output=raw_output,
            corrected_output=corrected_output,
            checks=checks,
            corrections=corrections,
            raw_answer=raw_answer,
            corrected_answer=corrected_answer,
            tokens_generated=n_tokens,
            generation_ms=gen_ms,
            correction_ms=corr_ms,
        )

    def _generate_text(self, prompt: str) -> Tuple[str, int]:
        """Generate text from prompt. Returns (text, num_tokens)."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                do_sample=False,
            )

        new_tokens = outputs[0][input_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return output_text, len(new_tokens)


# ---------------------------------------------------------------------------
# Answer extraction (shared with phase1_pipeline)
# ---------------------------------------------------------------------------

def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from model output."""
    if not text:
        return None
    text = text.strip()

    # GSM8K: "#### <number>"
    m = re.search(r'####\s*([\-\d,\.]+)', text)
    if m:
        return m.group(1).replace(",", "")

    # "The answer is <number>"
    m = re.search(r'(?:the\s+)?answer\s+is\s+([\-\d,\.]+)', text, re.IGNORECASE)
    if m:
        return m.group(1).replace(",", "")

    # "= <number>" at end
    m = re.search(r'=\s*([\-\d,\.]+)\s*$', text)
    if m:
        return m.group(1).replace(",", "")

    # Last number
    numbers = re.findall(r'[\-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    if answer is None:
        return ""
    answer = answer.strip().replace(",", "")
    try:
        val = float(answer)
        if val == int(val) and abs(val) < 1e15:
            return str(int(val))
        return f"{val:.6f}".rstrip('0').rstrip('.')
    except ValueError:
        return answer


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Phase 1 v2: Post-Hoc Correction")
    parser.add_argument("--model", default="google/gemma-3-4b-it")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    config = PipelineConfig(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        device=args.device,
    )

    pipeline = PostHocPipeline(config)
    result = pipeline.generate(args.prompt)

    print(f"\nRaw output:\n{result.raw_output[:500]}")
    print(f"\nArithmetic checks ({len(result.checks)}):")
    for c in result.checks:
        status = "OK" if c.is_correct else f"WRONG (should be {c.correct})"
        print(f"  {c.expression} = {c.claimed}  [{status}]")

    if result.corrections:
        print(f"\nCorrected output:\n{result.corrected_output[:500]}")
        print(f"\nRaw answer: {result.raw_answer}")
        print(f"Corrected answer: {result.corrected_answer}")
    else:
        print(f"\nNo corrections needed.")
        print(f"Answer: {result.raw_answer}")

    print(f"\nGeneration: {result.generation_ms:.0f}ms, Correction: {result.correction_ms:.2f}ms")


if __name__ == "__main__":
    main()
