"""
Benchmark larql Python bindings — speed and memory.

Usage:
    python bench/bench_bindings.py [path/to/model.vindex]
"""

import sys
import os
import time
import resource

VINDEX_PATH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    os.path.dirname(__file__), "..", "..", "..", "output", "gemma3-4b-v2.vindex"
)


def rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def bench(name, fn, n=10):
    fn()
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    mean = sum(times) / len(times)
    mn = min(times)
    print(f"  {name:<35} min={mn*1000:>8.2f}ms  mean={mean*1000:>8.2f}ms")
    return mean


def main():
    import larql

    print(f"Benchmarking: {VINDEX_PATH}")
    print()

    # ── Load ──
    print("=== Load ===")
    r0 = rss_mb()
    t0 = time.perf_counter()
    vindex = larql.load(VINDEX_PATH)
    t1 = time.perf_counter()
    r1 = rss_mb()
    print(f"  {'larql.load()':<35} {(t1-t0)*1000:>8.1f}ms  RSS: {r1:.0f} MB (+{r1-r0:.0f})")
    print(f"  {vindex}")
    print()

    # ── Core operations ──
    print("=== Core Operations ===")
    bands = vindex.layer_bands()
    last_knowledge = bands["knowledge"][1]

    bench("embed('France')", lambda: vindex.embed("France"))
    bench("gate_vector(L27, F0)", lambda: vindex.gate_vector(last_knowledge, 0))
    bench("gate_vectors(L27)", lambda: vindex.gate_vectors(last_knowledge), n=3)
    embed = vindex.embed("France")
    bench("gate_knn(L27, top=10)", lambda: vindex.gate_knn(last_knowledge, embed.tolist(), 10))
    bench("entity_knn('France', L27)", lambda: vindex.entity_knn("France", last_knowledge, 10))
    layers = list(range(bands["knowledge"][0], bands["knowledge"][1] + 1))
    bench("entity_walk('France', knowledge)", lambda: vindex.entity_walk("France", layers, 5))
    bench("describe('France')", lambda: vindex.describe("France"), n=5)
    bench("relations()", lambda: vindex.relations(), n=5)
    bench("cluster_centre('capital')", lambda: vindex.cluster_centre("capital"))
    print()

    # ── Inference: mmap'd (vindex.infer) ──
    print("=== Inference (vindex.infer — mmap'd, cached) ===")
    r2 = rss_mb()
    t2 = time.perf_counter()
    result = vindex.infer("The capital of France is", top_k_predictions=1)
    t3 = time.perf_counter()
    r3 = rss_mb()
    print(f"  {'1st call (cold mmap)':<35} {(t3-t2)*1000:>8.0f}ms  RSS: +{r3-r2:.0f} MB  → {result[0][0]}")

    t4 = time.perf_counter()
    result2 = vindex.infer("The largest planet is", top_k_predictions=1)
    t5 = time.perf_counter()
    print(f"  {'2nd call (warm cache)':<35} {(t5-t4)*1000:>8.0f}ms  → {result2[0][0]}")

    t6 = time.perf_counter()
    result3 = vindex.infer("Water boils at", top_k_predictions=1)
    t7 = time.perf_counter()
    print(f"  {'3rd call (hot cache)':<35} {(t7-t6)*1000:>8.0f}ms  → {result3[0][0]}")
    print()

    # ── WalkModel (zero-copy mmap) ──
    print("=== WalkModel (zero-copy mmap) ===")
    r4 = rss_mb()
    t8 = time.perf_counter()
    wm = larql.WalkModel(VINDEX_PATH, top_k=4096)
    t9 = time.perf_counter()
    r5 = rss_mb()
    print(f"  {'load (mmap)':<35} {(t9-t8)*1000:>8.0f}ms  RSS: {r5:.0f} MB (+{r5-r4:.0f})")

    t10 = time.perf_counter()
    result4 = wm.predict("The capital of France is")
    t11 = time.perf_counter()
    r6 = rss_mb()
    print(f"  {'predict (1st, cold)':<35} {(t11-t10)*1000:>8.0f}ms  RSS: {r6:.0f} MB  → {result4[0][0]}")

    t12 = time.perf_counter()
    result5 = wm.predict("The largest planet is")
    t13 = time.perf_counter()
    print(f"  {'predict (2nd, warm)':<35} {(t13-t12)*1000:>8.0f}ms  → {result5[0][0]}")
    print()

    # ── MLX ──
    print("=== MLX ===")
    try:
        import mlx_lm

        r7 = rss_mb()
        t14 = time.perf_counter()
        model, tokenizer = larql.mlx.load(VINDEX_PATH)
        t15 = time.perf_counter()
        r8 = rss_mb()
        print(f"  {'larql.mlx.load()':<35} {(t15-t14)*1000:>8.0f}ms  RSS: +{r8-r7:.0f} MB")

        t16 = time.perf_counter()
        resp = mlx_lm.generate(model, tokenizer, prompt="The capital of France is", max_tokens=5, verbose=False)
        t17 = time.perf_counter()
        print(f"  {'generate (5 tok)':<35} {(t17-t16)*1000:>8.0f}ms  → {resp.strip()}")
        del model, tokenizer

        import gc; gc.collect()
        r9 = rss_mb()
        t18 = time.perf_counter()
        model2, tok2 = mlx_lm.load("google/gemma-3-4b-it")
        t19 = time.perf_counter()
        r10 = rss_mb()
        print(f"  {'mlx_lm.load() (native)':<35} {(t19-t18)*1000:>8.0f}ms  RSS: +{r10-r9:.0f} MB")

        t20 = time.perf_counter()
        resp2 = mlx_lm.generate(model2, tok2, prompt="The capital of France is", max_tokens=5, verbose=False)
        t21 = time.perf_counter()
        print(f"  {'generate native (5 tok)':<35} {(t21-t20)*1000:>8.0f}ms  → {resp2.strip()}")

    except ImportError:
        print("  (mlx not installed — skipping)")

    # ── Summary ──
    print()
    print("=== Memory Summary ===")
    print(f"  Vindex load (gate+embed):     ~{r1:.0f} MB")
    print(f"  WalkModel load (all mmap'd):  +{r5-r4:.0f} MB  (zero-copy)")
    print(f"  WalkModel predict (paged):    ~{r6:.0f} MB  (OS pages in on demand)")
    print()
    print("Done.")


if __name__ == "__main__":
    main()
