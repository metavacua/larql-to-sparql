"""
Experiment 4h: Multi-layer small-alpha INSERT.

Single layer at alpha=5 produces Poseidon but breaks France.
Hypothesis: spread across 8 layers with small alpha, the nudges accumulate
for Atlantis (no competing signal) but are diluted for France (strong Paris signal).
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

    atlantis_prompt = "The capital of Atlantis is"
    france_prompt = "The capital of France is"

    v0 = larql.load(vindex_path)
    poseidon_id = v0.tokenize("Poseidon")[0]
    poseidon_vec = np.array(v0.embedding(poseidon_id)) * v0.embed_scale_value

    print("Capturing residuals...")
    t0 = time.time()
    _, atlantis_res = v0.infer_trace(atlantis_prompt, top_k_predictions=3)
    print(f"  {time.time()-t0:.1f}s")
    del v0

    layers = list(range(20, 28))  # 8 layers
    # Small alpha per layer: total effect = alpha * 8 layers
    # From sweep: alpha=5 at 1 layer = strong effect. So alpha=0.5-1.0 per layer × 8 = total 4-8
    test_alphas = [0.25, 0.5, 0.75, 1.0, 1.5]

    print(f"\n{'='*80}")
    print(f"  Multi-layer INSERT: L{layers[0]}-L{layers[-1]} ({len(layers)} layers)")
    print(f"{'='*80}")
    print(f"  {'alpha/layer':<12s} {'total':<8s} {'Atlantis':<18s} {'%':<8s} {'France':<18s} {'%':<8s}")
    print(f"  {'-'*72}")

    results = []
    for alpha in test_alphas:
        v = larql.load(vindex_path)

        for layer in layers:
            a_res = np.array(atlantis_res[layer])
            sample_norms = [np.linalg.norm(np.array(v.gate_vector(layer, f)))
                            for f in range(min(50, v.num_features(layer)))]
            avg_norm = np.mean([n for n in sample_norms if n > 0])
            gate_vec = a_res * (avg_norm / np.linalg.norm(a_res))
            down_vec = poseidon_vec * alpha

            free_feat = v.find_free_feature(layer)
            v.set_gate_vector(layer, free_feat, gate_vec.tolist())
            v.set_down_vector(layer, free_feat, down_vec.tolist())
            v.set_feature_meta(layer, free_feat, "Poseidon", 0.95)

        a_preds = v.infer(atlantis_prompt, top_k_predictions=10)
        f_preds = v.infer(france_prompt, top_k_predictions=5)

        a_top, a_prob = a_preds[0]
        f_top, f_prob = f_preds[0]
        pose_rank = next((i+1 for i, (t, _) in enumerate(a_preds) if "pose" in t.lower()), None)
        paris_rank = next((i+1 for i, (t, _) in enumerate(f_preds) if "paris" in t.lower()), None)

        total = alpha * len(layers)
        extra = ""
        if pose_rank: extra += f" Pose@{pose_rank}"
        if paris_rank: extra += f" Paris@{paris_rank}"

        results.append({
            "alpha_per_layer": alpha,
            "total_alpha": total,
            "num_layers": len(layers),
            "atlantis_top": a_top, "atlantis_prob": float(a_prob),
            "pose_rank": pose_rank,
            "france_top": f_top, "france_prob": float(f_prob),
            "paris_rank": paris_rank,
        })

        print(f"  {alpha:<12.2f} {total:<8.1f} {a_top:<18s} {a_prob*100:<8.2f} "
              f"{f_top:<18s} {f_prob*100:<8.2f}{extra}")

        del v

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04h_multilayer.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
