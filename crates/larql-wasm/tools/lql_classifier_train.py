#!/usr/bin/env python3
"""
LQL statement-type classifier — training script.

Trains a bag-of-words logistic regression classifier on the generated
corpus and exports static weights to lql_classifier_weights.json.

The exported weights are the ONLY runtime dependency — lql_classify.py
and the future Rust/WASM inference module use only:
  vocabulary    : list[str]  (fixed token set)
  weights       : float[n_classes][vocab_size]
  bias          : float[n_classes]
  classes       : list[str]  (first_token values)

Inference is: features = bag_of_words(prompt, vocabulary)
              scores   = weights @ features + bias
              result   = classes[argmax(scores)]

This is provably ¬L∧¬M: fixed-size arrays, no recursion, O(vocab × n_classes)
arithmetic. Suitable for wasm32v1-none deployment.

Usage:
  # Generate corpus first:
  python lql_corpus_gen.py --repo-root ../../../..

  # Train classifier:
  python lql_classifier_train.py \
    --corpus lql_corpus_full.json \
    --output lql_classifier_weights.json
"""
import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np


def tokenize(text: str) -> list[str]:
    """Lower-case word tokenization, keeping LQL keywords intact."""
    return re.findall(r"[a-z][a-z_-]*", text.lower())


def build_vocabulary(corpus: list[dict], min_df: int = 2, max_features: int = 512) -> list[str]:
    """Build vocabulary from training corpus — words that discriminate classes."""
    df: Counter = Counter()
    for entry in corpus:
        if entry.get("split") != "train":
            continue
        words = set(tokenize(entry["prompt"]))
        df.update(words)

    # Keep words that appear in at least min_df training prompts
    vocab = [w for w, count in df.most_common(max_features * 2) if count >= min_df]

    # Always include strong LQL trigger words even if rare
    trigger_words = [
        "describe", "walk", "infer", "select", "explain", "trace",
        "insert", "delete", "update", "merge", "rebalance",
        "extract", "compile", "diff", "use", "show", "stats",
        "compact", "begin", "save", "apply", "remove",
        "features", "layers", "relations", "edges", "entities", "models",
        "patch", "patches", "vindex", "model", "inference",
        "knowledge", "residual", "predict", "activate", "fire",
        "list", "find", "query", "what", "how", "which",
        "bake", "export", "compare", "merge", "calibrate",
        "confidence", "layer", "relation", "entity", "target",
    ]
    for w in trigger_words:
        if w not in vocab:
            vocab.append(w)

    return vocab[:max_features]


def bag_of_words(text: str, vocabulary: list[str]) -> np.ndarray:
    """Binary bag-of-words feature vector."""
    words = set(tokenize(text))
    return np.array([1.0 if w in words else 0.0 for w in vocabulary], dtype=np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def train_logistic_regression(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    lr: float = 0.1,
    epochs: int = 500,
    l2: float = 0.01,
) -> tuple[np.ndarray, np.ndarray]:
    """Simple softmax logistic regression via gradient descent.

    Returns (weights [n_classes, n_features], bias [n_classes]).
    Pure numpy — no sklearn required.
    """
    n_samples, n_features = X.shape
    W = np.zeros((n_classes, n_features), dtype=np.float64)
    b = np.zeros(n_classes, dtype=np.float64)

    # One-hot encode labels
    Y = np.zeros((n_samples, n_classes), dtype=np.float64)
    for i, label in enumerate(y):
        Y[i, label] = 1.0

    for epoch in range(epochs):
        # Forward pass
        logits = X @ W.T + b  # [n_samples, n_classes]
        probs = np.array([softmax(row) for row in logits])  # [n_samples, n_classes]

        # Loss (cross-entropy + L2)
        loss = -np.mean(np.sum(Y * np.log(probs + 1e-12), axis=1))
        loss += 0.5 * l2 * np.sum(W ** 2)

        # Gradient
        dL = (probs - Y) / n_samples  # [n_samples, n_classes]
        dW = dL.T @ X + l2 * W        # [n_classes, n_features]
        db = dL.sum(axis=0)            # [n_classes]

        W -= lr * dW
        b -= lr * db

        if epoch % 100 == 0:
            preds = np.argmax(logits, axis=1)
            acc = (preds == y).mean()
            print(f"  epoch {epoch:4d}  loss={loss:.4f}  train_acc={acc:.3f}")

    return W, b


def evaluate(W, b, X, y, classes):
    logits = X @ W.T + b
    preds = np.argmax(logits, axis=1)
    acc = (preds == y).mean()
    print(f"Accuracy: {acc:.1%} ({(preds==y).sum()}/{len(y)})")

    # Per-class breakdown
    from collections import defaultdict
    per_class: dict = defaultdict(lambda: [0, 0])
    for pred, true in zip(preds, y):
        per_class[classes[true]][1] += 1
        if pred == true:
            per_class[classes[true]][0] += 1
    print("\nPer class:")
    for cls in sorted(per_class):
        ok, total = per_class[cls]
        bar = "█" * ok + "░" * (total - ok)
        print(f"  {cls:12s}  {ok}/{total:2d}  {bar}")
    return acc


def main():
    ap = argparse.ArgumentParser(description="Train LQL statement-type classifier")
    ap.add_argument("--corpus", default="crates/larql-wasm/tools/lql_corpus_full.json",
                    help="Path to generated corpus JSON")
    ap.add_argument("--output", default="crates/larql-wasm/tools/lql_classifier_weights.json",
                    help="Output weights JSON")
    ap.add_argument("--max-features", type=int, default=256,
                    help="Vocabulary size (default: 256)")
    ap.add_argument("--epochs", type=int, default=800)
    ap.add_argument("--lr", type=float, default=0.15)
    ap.add_argument("--l2", type=float, default=0.005)
    args = ap.parse_args()

    corpus = json.loads(Path(args.corpus).read_text())
    print(f"Corpus: {len(corpus)} entries")

    train = [e for e in corpus if e.get("split") == "train"]
    eval_ = [e for e in corpus if e.get("split") == "eval"]
    print(f"  train={len(train)}  eval={len(eval_)}")

    # Build vocabulary from training data
    vocab = build_vocabulary(corpus, min_df=1, max_features=args.max_features)
    print(f"\nVocabulary: {len(vocab)} tokens")

    # Collect unique classes (first_token values)
    classes = sorted({e["first_token"] for e in corpus})
    class_to_idx = {c: i for i, c in enumerate(classes)}
    print(f"Classes ({len(classes)}): {classes}")

    # Build feature matrices
    X_train = np.array([bag_of_words(e["prompt"], vocab) for e in train])
    y_train = np.array([class_to_idx[e["first_token"]] for e in train])
    X_eval = np.array([bag_of_words(e["prompt"], vocab) for e in eval_])
    y_eval = np.array([class_to_idx[e["first_token"]] for e in eval_])

    print(f"\nTraining ({args.epochs} epochs)...")
    W, b = train_logistic_regression(
        X_train, y_train,
        n_classes=len(classes),
        lr=args.lr,
        epochs=args.epochs,
        l2=args.l2,
    )

    print("\n── Train evaluation ──")
    evaluate(W, b, X_train, y_train, classes)
    print("\n── Eval evaluation ──")
    eval_acc = evaluate(W, b, X_eval, y_eval, classes)

    # Export weights
    weights_data = {
        "vocabulary": vocab,
        "weights": W.tolist(),
        "bias": b.tolist(),
        "classes": classes,
        "eval_accuracy": float(eval_acc),
        "n_train": len(train),
        "n_eval": len(eval_),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(weights_data, indent=2))
    print(f"\nWeights saved → {out}")
    print(f"  vocab={len(vocab)}  classes={len(classes)}  eval_acc={eval_acc:.1%}")
    print(f"  Weight matrix: {len(classes)} × {len(vocab)} = {len(classes)*len(vocab)} floats")


if __name__ == "__main__":
    main()
