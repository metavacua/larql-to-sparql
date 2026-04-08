#!/usr/bin/env python3
"""
Phase 1: Pipeline Integration

End-to-end: Gemma 3-4B (or any HF model) generation with token-level
solver interception. Two modes:

  1. Baseline — standard model generation
  2. Augmented — generation with solver interception

The pipeline monitors the output token stream. When a computable
expression is detected, it:
  1. Pauses generation
  2. Parses the expression
  3. Dispatches to solver
  4. Injects answer tokens
  5. Resumes generation

Usage:
  python phase1_pipeline.py --model google/gemma-3-4b-it --prompt "What is 6 * 7?"
  python phase1_pipeline.py --model google/gemma-3-4b-it --prompt "What is 6 * 7?" --no-solver
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

from phase1_parser import parse, parse_all, StreamParser, normalise
from phase1_solvers import solve, SolverResult


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class PipelineConfig:
    model_name: str = "google/gemma-3-4b-it"
    max_new_tokens: int = 512
    temperature: float = 0.0       # greedy for reproducibility
    top_p: float = 1.0
    device: str = "auto"
    dtype: str = "bfloat16"
    solver_enabled: bool = True
    # How many tokens of context the stream parser keeps
    parser_window: int = 300
    # Maximum solver calls per generation (safety limit)
    max_interventions: int = 20


# ---------------------------------------------------------------------------
# Intervention record
# ---------------------------------------------------------------------------

@dataclass
class Intervention:
    position: int           # token position where intervention happened
    matched_text: str       # text that triggered the solver
    expr_type: str          # type of expression detected
    solver_result: str      # answer from solver
    solver_name: str        # which solver was used
    solver_us: float        # solver latency in microseconds
    tokens_injected: int    # how many tokens were injected


@dataclass
class GenerationResult:
    prompt: str
    output: str
    tokens_generated: int
    total_time_ms: float
    interventions: List[Intervention] = field(default_factory=list)
    solver_enabled: bool = True

    def to_dict(self):
        d = asdict(self)
        return d


# ---------------------------------------------------------------------------
# Core pipeline
# ---------------------------------------------------------------------------

class ComputePipeline:
    """Token-level solver interception pipeline."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self):
        """Load model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading {self.config.model_name}...")
        dtype_map = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }
        dtype = dtype_map.get(self.config.dtype, torch.bfloat16)

        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)

        # MPS has matmul compatibility issues with some models (Gemma 3).
        # Default to CPU for reliability; caller can override with --device mps.
        device = self.config.device
        if device == "auto":
            device = "cpu"

        self.model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            torch_dtype=dtype,
            device_map=device if device not in ("mps", "cpu") else None,
        )
        if device in ("mps", "cpu"):
            self.model = self.model.to(device)
        self.model.eval()
        self._loaded = True
        try:
            dev = self.model.device
        except Exception:
            dev = device
        print(f"Model loaded on {dev}", flush=True)

    def generate_baseline(self, prompt: str) -> GenerationResult:
        """Standard generation without solver interception."""
        self.load()
        t0 = time.perf_counter()

        inputs = self._encode_prompt(prompt)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature if self.config.temperature > 0 else None,
                top_p=self.config.top_p,
                do_sample=self.config.temperature > 0,
            )

        new_tokens = outputs[0][input_len:]
        output_text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            prompt=prompt,
            output=output_text,
            tokens_generated=len(new_tokens),
            total_time_ms=elapsed_ms,
            solver_enabled=False,
        )

    def generate_augmented(self, prompt: str) -> GenerationResult:
        """Generation with solver interception."""
        self.load()
        t0 = time.perf_counter()

        inputs = self._encode_prompt(prompt)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        generated_tokens = []
        interventions = []
        stream_parser = StreamParser(window_size=self.config.parser_window)
        num_interventions = 0

        # Token-by-token generation
        past_key_values = None
        for step in range(self.config.max_new_tokens):
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    use_cache=True,
                )

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1, :]

            # Greedy decoding
            if self.config.temperature <= 0:
                next_token_id = logits.argmax(dim=-1, keepdim=True)
            else:
                probs = torch.softmax(logits / self.config.temperature, dim=-1)
                next_token_id = torch.multinomial(probs, num_samples=1)

            token_id = next_token_id.item()

            # Check for EOS
            if token_id == self.tokenizer.eos_token_id:
                break

            token_text = self.tokenizer.decode([token_id])
            generated_tokens.append(token_id)

            # Feed to stream parser
            expr = stream_parser.feed(token_text)

            if expr is not None and num_interventions < self.config.max_interventions:
                # Solver dispatch
                result = solve(expr)
                if result is not None:
                    # Inject solver answer as tokens
                    answer_text = " " + result.value
                    answer_ids = self.tokenizer.encode(answer_text, add_special_tokens=False)

                    interventions.append(Intervention(
                        position=len(generated_tokens),
                        matched_text=expr.raw,
                        expr_type=expr.expr_type.name,
                        solver_result=result.value,
                        solver_name=result.solver,
                        solver_us=result.elapsed_us,
                        tokens_injected=len(answer_ids),
                    ))
                    num_interventions += 1

                    # Add answer tokens to the sequence
                    for aid in answer_ids:
                        generated_tokens.append(aid)

                    # Update KV cache: feed the injected tokens through the model
                    inject_ids = torch.tensor([answer_ids], device=input_ids.device)
                    inject_mask = torch.ones(
                        1, attention_mask.shape[1] + len(generated_tokens),
                        device=attention_mask.device, dtype=attention_mask.dtype,
                    )
                    with torch.no_grad():
                        inject_out = self.model(
                            input_ids=inject_ids,
                            attention_mask=inject_mask,
                            past_key_values=past_key_values,
                            use_cache=True,
                        )
                    past_key_values = inject_out.past_key_values

                    # Reset parser window (answer already injected)
                    stream_parser.reset()

                    # Update for next step
                    next_token_id = torch.tensor([[answer_ids[-1]]], device=input_ids.device)

            # Prepare for next step
            input_ids = next_token_id.view(1, 1)
            attention_mask = torch.ones(
                1, attention_mask.shape[1] + 1,
                device=attention_mask.device, dtype=attention_mask.dtype,
            )

        output_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        elapsed_ms = (time.perf_counter() - t0) * 1000

        return GenerationResult(
            prompt=prompt,
            output=output_text,
            tokens_generated=len(generated_tokens),
            total_time_ms=elapsed_ms,
            interventions=interventions,
            solver_enabled=True,
        )

    def generate(self, prompt: str) -> GenerationResult:
        """Generate with or without solver based on config."""
        if self.config.solver_enabled:
            return self.generate_augmented(prompt)
        else:
            return self.generate_baseline(prompt)

    def _encode_prompt(self, prompt: str) -> dict:
        """Encode prompt using chat template if available."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": prompt}]
            text = self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt

        inputs = self.tokenizer(text, return_tensors="pt")
        device = next(self.model.parameters()).device
        return {k: v.to(device) for k, v in inputs.items()}


# ---------------------------------------------------------------------------
# Answer extraction for benchmarks
# ---------------------------------------------------------------------------

def extract_numeric_answer(text: str) -> Optional[str]:
    """Extract the final numeric answer from model output.

    Handles common formats:
      - "The answer is 42"
      - "#### 42"
      - "= 42"
      - Last number in the text
    """
    text = text.strip()

    # GSM8K format: "#### <number>"
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

    # Last number in text
    numbers = re.findall(r'[\-]?\d+(?:,\d{3})*(?:\.\d+)?', text)
    if numbers:
        return numbers[-1].replace(",", "")

    return None


def normalize_answer(answer: str) -> str:
    """Normalize a numeric answer for comparison."""
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
    parser = argparse.ArgumentParser(description="Phase 1: Compute Pipeline")
    parser.add_argument("--model", default="google/gemma-3-4b-it", help="HF model name")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--no-solver", action="store_true", help="Disable solver (baseline)")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--device", default="auto", help="Device: auto, mps, cpu")
    args = parser.parse_args()

    config = PipelineConfig(
        model_name=args.model,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        solver_enabled=not args.no_solver,
        device=args.device,
    )

    pipeline = ComputePipeline(config)
    result = pipeline.generate(args.prompt)

    print(f"\nPrompt: {result.prompt}")
    print(f"Output: {result.output}")
    print(f"Tokens: {result.tokens_generated}")
    print(f"Time: {result.total_time_ms:.1f}ms")
    if result.interventions:
        print(f"\nInterventions ({len(result.interventions)}):")
        for iv in result.interventions:
            print(f"  [{iv.position}] {iv.expr_type}: {iv.matched_text!r} → {iv.solver_result}"
                  f" ({iv.solver_name}, {iv.solver_us:.1f}μs, {iv.tokens_injected} tokens)")

    extracted = extract_numeric_answer(result.output)
    if extracted:
        print(f"\nExtracted answer: {extracted}")


if __name__ == "__main__":
    main()
