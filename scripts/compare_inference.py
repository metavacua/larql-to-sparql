#!/usr/bin/env python3
"""Compare larql inference against MLX reference implementation.

Usage: python3 scripts/compare_inference.py <model> "<prompt>" [top_k]
"""

import sys
import subprocess
import json
from pathlib import Path

import mlx.core as mx
from mlx_lm import load as mlx_load


def cast_model_f32(model):
    """Cast all model parameters to float32 for exact comparison."""
    from mlx.utils import tree_map

    def to_f32(p):
        if isinstance(p, mx.array) and p.dtype != mx.float32:
            return p.astype(mx.float32)
        return p

    new_params = tree_map(to_f32, model.parameters())
    model.update(new_params)
    return model


def mlx_top_k(model_path: str, prompt: str, top_k: int = 10):
    """Run MLX inference in F32 and return top-k predictions."""
    model, tokenizer = mlx_load(model_path)
    model = cast_model_f32(model)
    mx.eval(model.parameters())

    tokens = mx.array(tokenizer.encode(prompt))
    logits = model(tokens[None])  # (1, seq_len, vocab_size)
    last_logits = logits[0, -1, :].astype(mx.float32)

    # Top-k
    top_indices = mx.argpartition(-last_logits, kth=top_k)[:top_k]
    top_logit_vals = last_logits[top_indices]

    # Softmax over full vocab for proper probabilities
    max_logit = mx.max(last_logits)
    exp_logits = mx.exp(last_logits - max_logit)
    probs = exp_logits / mx.sum(exp_logits)
    top_probs = probs[top_indices]

    # Sort by probability
    sort_idx = mx.argsort(-top_probs)
    results = []
    for i in sort_idx.tolist():
        tok_id = top_indices[i].item()
        prob = top_probs[i].item()
        logit = top_logit_vals[i].item()
        token_str = tokenizer.decode([tok_id])
        results.append((token_str, prob, logit, tok_id))

    # Also get logit stats
    print(f"  MLX logit range: [{mx.min(last_logits).item():.2f}, {mx.max(last_logits).item():.2f}]")
    print(f"  MLX top logits: {[f'{r[2]:.3f}' for r in results[:5]]}")

    token_ids = tokenizer.encode(prompt)
    return results, token_ids


def larql_top_k(model_path: str, prompt: str, top_k: int = 10):
    """Run larql inference and return top-k predictions."""
    result = subprocess.run(
        ["cargo", "run", "--release", "-p", "larql-cli", "--",
         "predict", model_path, "-p", prompt, "-k", str(top_k)],
        capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent)
    )
    # Parse output — format: "  1. token               0.1234 (12.34%)"
    predictions = []
    import re
    for line in result.stdout.splitlines():
        m = re.match(r'\s+\d+\.\s+(.+?)\s+(\d+\.\d+)\s+\(', line)
        if m:
            token = m.group(1).strip()
            prob = float(m.group(2))
            predictions.append((token, prob))

    # Get token IDs from stderr
    token_ids = []
    for line in result.stderr.splitlines():
        if "tokens:" in line:
            # Extract [1, 2, 3] from the line
            bracket_start = line.find('[')
            bracket_end = line.find(']')
            if bracket_start >= 0 and bracket_end >= 0:
                ids_str = line[bracket_start+1:bracket_end]
                token_ids = [int(x.strip()) for x in ids_str.split(',')]

    return predictions, token_ids


def main():
    if len(sys.argv) < 3:
        print("Usage: python3 scripts/compare_inference.py <model> \"<prompt>\" [top_k]")
        sys.exit(1)

    model_path = sys.argv[1]
    prompt = sys.argv[2]
    top_k = int(sys.argv[3]) if len(sys.argv) > 3 else 10

    print(f"Model: {model_path}")
    print(f"Prompt: {prompt!r}")
    print()

    # MLX reference
    print("Running MLX reference...")
    mlx_results, mlx_tokens = mlx_top_k(model_path, prompt, top_k)
    print(f"  MLX token IDs: {mlx_tokens}")

    # larql
    print("Running larql...")
    larql_results, larql_tokens = larql_top_k(model_path, prompt, top_k)
    print(f"  larql token IDs: {larql_tokens}")

    # Compare token IDs
    if mlx_tokens != larql_tokens:
        print(f"\n⚠ TOKEN MISMATCH: MLX={mlx_tokens} vs larql={larql_tokens}")
    else:
        print(f"\n  Token IDs match: {mlx_tokens}")

    # Print comparison table
    print(f"\n{'Rank':<5} {'MLX Token':<20} {'MLX Prob':>10} {'larql Token':<20} {'larql Prob':>10} {'Match':>6}")
    print("-" * 75)

    for i in range(top_k):
        mlx_tok = mlx_results[i][0] if i < len(mlx_results) else "?"
        mlx_prob = mlx_results[i][1] if i < len(mlx_results) else 0.0
        larql_tok = larql_results[i][0] if i < len(larql_results) else "?"
        larql_prob = larql_results[i][1] if i < len(larql_results) else 0.0

        match = "yes" if mlx_tok.strip() == larql_tok.strip() else "NO"
        print(f"{i+1:<5} {mlx_tok:<20} {mlx_prob:>9.4f} {larql_tok:<20} {larql_prob:>9.4f} {match:>6}")

    # Summary
    mlx_top1 = mlx_results[0][0].strip() if mlx_results else "?"
    larql_top1 = larql_results[0][0].strip() if larql_results else "?"
    if mlx_top1 == larql_top1:
        print(f"\nTop-1 MATCH: {mlx_top1!r}")
    else:
        print(f"\nTop-1 MISMATCH: MLX={mlx_top1!r} vs larql={larql_top1!r}")


if __name__ == "__main__":
    main()
