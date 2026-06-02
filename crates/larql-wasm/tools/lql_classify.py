#!/usr/bin/env python3
"""
LQL statement-type classifier — deterministic inference.

Loads static weights from lql_classifier_weights.json and classifies
natural language prompts into LQL statement types.

This module is intentionally minimal and ¬L∧¬M:
  - Fixed vocabulary (no dynamic growth)
  - Single matrix multiply + argmax (no recursion)
  - No training, no model loading, no heavy dependencies
  - Pure numpy at runtime (or pure Python if numpy absent)

The same logic translates directly to Rust/no_std for wasm32v1-none:
  fn classify(prompt: &str) -> &'static str {
      let features = bag_of_words(prompt, VOCAB);
      let scores = mat_vec_mul(WEIGHTS, &features) + BIAS;
      CLASSES[argmax(&scores)]
  }

Usage:
  # Classify a single prompt
  python lql_classify.py "Describe what the model knows about France"
  # → DESCRIBE

  # Classify from stdin (one prompt per line)
  echo "List all edges" | python lql_classify.py --stdin

  # Show top-3 predictions with confidence
  python lql_classify.py --top 3 "Show statistics about the vindex"
"""
import argparse
import json
import re
import sys
from pathlib import Path


DEFAULT_WEIGHTS = Path(__file__).parent / "lql_classifier_weights.json"


def tokenize(text: str) -> list:
    return re.findall(r"[a-z][a-z_-]*", text.lower())


def load_weights(path: Path) -> dict:
    return json.loads(path.read_text())


def classify(prompt: str, model: dict, top_k: int = 1) -> list:
    """
    Classify a natural language prompt into LQL statement type(s).

    Returns list of (class, confidence) sorted by confidence descending.
    Pure Python/numpy implementation — no external ML dependencies at runtime.
    """
    vocab = model["vocabulary"]
    weights = model["weights"]   # list[n_classes][vocab_size]
    bias = model["bias"]         # list[n_classes]
    classes = model["classes"]

    # Bag-of-words feature vector (binary)
    words = set(tokenize(prompt))
    features = [1.0 if w in words else 0.0 for w in vocab]

    # scores = weights @ features + bias
    scores = []
    for i, (w_row, b) in enumerate(zip(weights, bias)):
        score = sum(w * f for w, f in zip(w_row, features)) + b
        scores.append(score)

    # Softmax for probabilities
    max_score = max(scores)
    exp_scores = [2.718281828 ** (s - max_score) for s in scores]
    total = sum(exp_scores)
    probs = [e / total for e in exp_scores]

    # Sort by probability descending
    ranked = sorted(zip(classes, probs), key=lambda x: -x[1])
    return ranked[:top_k]


def classify_numpy(prompt: str, model: dict, top_k: int = 1) -> list:
    """Faster numpy variant — same semantics as classify()."""
    try:
        import numpy as np
    except ImportError:
        return classify(prompt, model, top_k)

    vocab = model["vocabulary"]
    W = np.array(model["weights"], dtype=np.float32)   # [n_classes, vocab_size]
    b = np.array(model["bias"], dtype=np.float32)       # [n_classes]
    classes = model["classes"]

    words = set(tokenize(prompt))
    features = np.array([1.0 if w in words else 0.0 for w in vocab], dtype=np.float32)

    scores = W @ features + b  # [n_classes]
    exp = np.exp(scores - scores.max())
    probs = exp / exp.sum()

    top_idx = np.argsort(-probs)[:top_k]
    return [(classes[i], float(probs[i])) for i in top_idx]


def main():
    ap = argparse.ArgumentParser(description="Classify NL prompts into LQL statement types")
    ap.add_argument("prompt", nargs="?", help="Natural language prompt to classify")
    ap.add_argument("--weights", default=str(DEFAULT_WEIGHTS),
                    help="Path to classifier weights JSON")
    ap.add_argument("--top", type=int, default=1, metavar="K",
                    help="Return top-K predictions (default: 1)")
    ap.add_argument("--stdin", action="store_true",
                    help="Read prompts from stdin, one per line")
    ap.add_argument("--json", action="store_true",
                    help="Output results as JSON")
    args = ap.parse_args()

    weights_path = Path(args.weights)
    if not weights_path.exists():
        print(f"ERROR: weights not found at {weights_path}", file=sys.stderr)
        print("Run lql_classifier_train.py first.", file=sys.stderr)
        sys.exit(1)

    model = load_weights(weights_path)

    if args.stdin:
        for line in sys.stdin:
            prompt = line.rstrip()
            if not prompt:
                continue
            results = classify_numpy(prompt, model, args.top)
            if args.json:
                print(json.dumps({"prompt": prompt, "predictions": results}))
            else:
                top_class, top_conf = results[0]
                print(f"{top_class:12s}  {top_conf:.3f}  {prompt[:60]}")
    elif args.prompt:
        results = classify_numpy(args.prompt, model, args.top)
        if args.json:
            print(json.dumps({"prompt": args.prompt, "predictions": results}))
        else:
            for cls, conf in results:
                print(f"{cls:12s}  {conf:.3f}")
    else:
        # Interactive mode
        print(f"LQL classifier  ({len(model['classes'])} classes, "
              f"{len(model['vocabulary'])} vocab, "
              f"eval_acc={model.get('eval_accuracy', '?'):.1%})")
        print("Enter prompts (Ctrl-D to exit):")
        try:
            while True:
                prompt = input("> ").strip()
                if not prompt:
                    continue
                results = classify_numpy(prompt, model, max(args.top, 3))
                for cls, conf in results:
                    bar = "█" * int(conf * 20)
                    print(f"  {cls:12s}  {conf:.3f}  {bar}")
        except (EOFError, KeyboardInterrupt):
            pass


if __name__ == "__main__":
    main()
