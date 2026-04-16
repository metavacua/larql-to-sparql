"""
Experiment 4g: Fine-grained alpha sweep between 1.0 and 5.0.
Looking for the sweet spot where Atlantis→Poseidon and France→Paris.
Uses multiple layers (not just L26) for broader coverage.
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
    poseidon_dir = np.array(v0.embedding(poseidon_id))
    embed_scale = v0.embed_scale_value
    poseidon_vec = poseidon_dir * embed_scale

    print("Capturing residuals...")
    t0 = time.time()
    _, atlantis_res = v0.infer_trace(atlantis_prompt, top_k_predictions=3)
    print(f"  {time.time()-t0:.1f}s")
    del v0

    # Sweep: single layer L26, fine alphas
    alphas = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    target_layer = 26

    print(f"\n{'='*75}")
    print(f"  Fine sweep at L{target_layer}")
    print(f"{'='*75}")
    print(f"  {'alpha':<8s} {'Atlantis':<20s} {'%':<8s} {'France':<20s} {'%':<8s}")
    print(f"  {'-'*65}")

    results = []
    for alpha in alphas:
        v = larql.load(vindex_path)
        a_res = np.array(atlantis_res[target_layer])

        # Gate: full Atlantis residual (not orthogonal — orthogonal didn't help)
        sample_norms = [np.linalg.norm(np.array(v.gate_vector(target_layer, f)))
                        for f in range(min(50, v.num_features(target_layer)))]
        avg_norm = np.mean([n for n in sample_norms if n > 0])
        gate_vec = a_res * (avg_norm / np.linalg.norm(a_res))

        down_vec = poseidon_vec * alpha

        free_feat = v.find_free_feature(target_layer)
        v.set_gate_vector(target_layer, free_feat, gate_vec.tolist())
        v.set_down_vector(target_layer, free_feat, down_vec.tolist())
        v.set_feature_meta(target_layer, free_feat, "Poseidon", 0.95)

        a_preds = v.infer(atlantis_prompt, top_k_predictions=10)
        f_preds = v.infer(france_prompt, top_k_predictions=5)

        a_top, a_prob = a_preds[0]
        f_top, f_prob = f_preds[0]
        pose_rank = next((i+1 for i, (t, _) in enumerate(a_preds) if "pose" in t.lower()), None)
        paris_rank = next((i+1 for i, (t, _) in enumerate(f_preds) if "paris" in t.lower()), None)

        results.append({
            "alpha": alpha,
            "atlantis_top": a_top, "atlantis_prob": float(a_prob),
            "pose_rank": pose_rank,
            "france_top": f_top, "france_prob": float(f_prob),
            "paris_rank": paris_rank,
        })

        a_extra = f" (Pose@{pose_rank})" if pose_rank else ""
        f_extra = f" (Paris@{paris_rank})" if paris_rank else ""
        print(f"  {alpha:<8.1f} {a_top:<20s} {a_prob*100:<8.2f} {f_top:<20s} {f_prob*100:<8.2f}{a_extra}{f_extra}")

        del v

    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04g_fine_sweep.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
