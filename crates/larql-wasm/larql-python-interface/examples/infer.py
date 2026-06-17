"""
Inference from a vindex — Rust attention + vindex walk FFN.

Weights are mmap'd (zero-copy). First call loads, subsequent calls
reuse cached weights with OS page cache warming.

Usage:
    python examples/infer.py [path/to/model.vindex]
"""

import sys
import time
import larql

vindex = larql.load(sys.argv[1] if len(sys.argv) > 1 else "output/gemma3-4b-v2.vindex")
print(vindex)
print()

prompts = [
    "The capital of France is",
    "Albert Einstein was a",
    "The programming language Python was created by",
    "The largest planet in our solar system is",
]

for i, prompt in enumerate(prompts):
    t0 = time.time()
    result = vindex.infer(prompt, top_k_predictions=3)
    t1 = time.time()
    top = result[0]
    label = "cold" if i == 0 else "warm"
    print(f"  [{label:4} {t1-t0:.1f}s] {prompt}")
    print(f"    → {top[0]} ({top[1]:.1%})")
    print()
