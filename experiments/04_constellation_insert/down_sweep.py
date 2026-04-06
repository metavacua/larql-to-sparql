"""
Experiment 4e: Down vector magnitude sweep.

Uses Poseidon embedding as the down direction, sweeps alpha to find
the right magnitude. Inserts at a single layer first to isolate the effect.
"""

import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import larql

VINDEX_PATH = os.environ.get("VINDEX_PATH", "output/gemma3-4b-f16.vindex")


def run():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else VINDEX_PATH

    prompt = "The capital of Atlantis is"
    france_prompt = "The capital of France is"

    # Get the Poseidon direction (first subtoken)
    v0 = larql.load(vindex_path)
    poseidon_id = v0.tokenize("Poseidon")[0]
    poseidon_dir = np.array(v0.embedding(poseidon_id))  # unit norm
    embed_scale = v0.embed_scale_value
    # Scale by embed_scale so it's in the same space as the residual
    poseidon_vec = poseidon_dir * embed_scale

    print(f"Poseidon direction: norm={np.linalg.norm(poseidon_dir):.3f}")
    print(f"Poseidon vec (scaled): norm={np.linalg.norm(poseidon_vec):.1f}")

    # Capture Atlantis residuals once
    print(f"\nCapturing residuals for: \"{prompt}\"")
    t0 = time.time()
    _, residuals = v0.infer_trace(prompt, top_k_predictions=3)
    print(f"  {time.time()-t0:.1f}s")
    del v0  # Release

    # Sweep alpha values
    target_layer = 26  # Late knowledge layer
    alphas = [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0]

    print(f"\n{'='*70}")
    print(f"  Magnitude sweep at L{target_layer}")
    print(f"  Down vector = alpha * embed(Poseidon) * embed_scale")
    print(f"{'='*70}")
    print(f"  {'alpha':<10s} {'top1':<15s} {'prob':<10s} {'Poseidon?':<12s} {'France ok?':<10s}")
    print(f"  {'-'*60}")

    results = []
    for alpha in alphas:
        # Fresh load each time to avoid accumulation
        v = larql.load(vindex_path)
        atlantis_res = np.array(residuals[target_layer])

        # Gate: match Atlantis residual
        sample_norms = [np.linalg.norm(np.array(v.gate_vector(target_layer, f)))
                        for f in range(min(50, v.num_features(target_layer)))]
        avg_norm = np.mean([n for n in sample_norms if n > 0])
        res_norm = np.linalg.norm(atlantis_res)
        gate_vec = atlantis_res * (avg_norm / res_norm)

        # Down: Poseidon direction * alpha
        down_vec = poseidon_vec * alpha

        # Insert
        free_feat = v.find_free_feature(target_layer)
        v.set_gate_vector(target_layer, free_feat, gate_vec.tolist())
        v.set_down_vector(target_layer, free_feat, down_vec.tolist())
        v.set_feature_meta(target_layer, free_feat, "Poseidon", 0.95)

        # Test
        new_preds = v.infer(prompt, top_k_predictions=10)
        france_preds = v.infer(france_prompt, top_k_predictions=3)

        top1, top1_prob = new_preds[0]
        poseidon_found = any("poseidon" in t.lower() for t, _ in new_preds[:10])
        poseidon_rank = next((i+1 for i, (t, _) in enumerate(new_preds)
                              if "poseidon" in t.lower()), None)
        france_ok = "paris" in france_preds[0][0].lower()

        results.append({
            "alpha": alpha,
            "top1": top1,
            "top1_prob": float(top1_prob),
            "poseidon_found": poseidon_found,
            "poseidon_rank": poseidon_rank,
            "france_ok": france_ok,
            "france_top": france_preds[0][0],
        })

        pos_str = f"rank {poseidon_rank}" if poseidon_rank else "no"
        print(f"  {alpha:<10.3f} {top1:<15s} {top1_prob*100:<10.2f} {pos_str:<12s} "
              f"{'yes' if france_ok else france_preds[0][0]}")

        del v

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04e_down_sweep.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
