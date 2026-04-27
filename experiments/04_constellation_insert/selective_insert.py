"""
Experiment 4f: Selective INSERT — gate fires for Atlantis but NOT France.

alpha=5-10 produces "Pose" at 27-65%, but also breaks France.
The problem: gate fires for both because cos(France,Atlantis)=0.98 at L26.

Fix: use the orthogonal component of the Atlantis residual as the gate.
  gate = atlantis_residual - project(atlantis_residual, france_residual)
This gate has zero dot product with France but positive with Atlantis.
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
    germany_prompt = "The capital of Germany is"

    # Get Poseidon direction
    v0 = larql.load(vindex_path)
    poseidon_id = v0.tokenize("Poseidon")[0]
    poseidon_dir = np.array(v0.embedding(poseidon_id))
    embed_scale = v0.embed_scale_value
    poseidon_vec = poseidon_dir * embed_scale

    # Capture residuals for Atlantis AND France
    print("Capturing residuals...")
    t0 = time.time()
    _, atlantis_res = v0.infer_trace(atlantis_prompt, top_k_predictions=3)
    _, france_res = v0.infer_trace(france_prompt, top_k_predictions=3)
    print(f"  {time.time()-t0:.1f}s")
    del v0

    target_layer = 26

    a_res = np.array(atlantis_res[target_layer])
    f_res = np.array(france_res[target_layer])

    # Compute the Atlantis-specific direction (orthogonal to France)
    # project(a, f) = f * (a·f / f·f)
    proj_on_france = f_res * (np.dot(a_res, f_res) / (np.dot(f_res, f_res) + 1e-8))
    atlantis_specific = a_res - proj_on_france

    cos_specific_france = np.dot(atlantis_specific, f_res) / (
        np.linalg.norm(atlantis_specific) * np.linalg.norm(f_res) + 1e-8)
    cos_specific_atlantis = np.dot(atlantis_specific, a_res) / (
        np.linalg.norm(atlantis_specific) * np.linalg.norm(a_res) + 1e-8)

    print(f"\nL{target_layer} residual analysis:")
    print(f"  cos(France, Atlantis) = {np.dot(a_res, f_res) / (np.linalg.norm(a_res) * np.linalg.norm(f_res)):.4f}")
    print(f"  Atlantis-specific component:")
    print(f"    norm = {np.linalg.norm(atlantis_specific):.1f}")
    print(f"    cos with France = {cos_specific_france:.6f} (should be ~0)")
    print(f"    cos with Atlantis = {cos_specific_atlantis:.4f}")

    # Sweep with selective gate
    alphas = [1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]

    print(f"\n{'='*75}")
    print(f"  Selective INSERT sweep at L{target_layer}")
    print(f"  Gate = Atlantis-specific (orthogonal to France)")
    print(f"  Down = alpha * embed(Poseidon) * embed_scale")
    print(f"{'='*75}")
    print(f"  {'alpha':<8s} {'Atlantis top1':<18s} {'prob':<8s} {'France top1':<18s} {'prob':<8s}")
    print(f"  {'-'*65}")

    results = []
    for alpha in alphas:
        v = larql.load(vindex_path)

        # Gate: Atlantis-specific direction, normalised
        sample_norms = [np.linalg.norm(np.array(v.gate_vector(target_layer, f)))
                        for f in range(min(50, v.num_features(target_layer)))]
        avg_norm = np.mean([n for n in sample_norms if n > 0])
        spec_norm = np.linalg.norm(atlantis_specific)
        gate_vec = atlantis_specific * (avg_norm / spec_norm) if spec_norm > 0 else atlantis_specific

        # Down: Poseidon direction * alpha
        down_vec = poseidon_vec * alpha

        # Insert
        free_feat = v.find_free_feature(target_layer)
        v.set_gate_vector(target_layer, free_feat, gate_vec.tolist())
        v.set_down_vector(target_layer, free_feat, down_vec.tolist())
        v.set_feature_meta(target_layer, free_feat, "Poseidon", 0.95)

        # Verify selectivity
        a_score = np.dot(gate_vec, a_res)
        f_score = np.dot(gate_vec, f_res)

        # Test
        a_preds = v.infer(atlantis_prompt, top_k_predictions=5)
        f_preds = v.infer(france_prompt, top_k_predictions=3)

        a_top, a_prob = a_preds[0]
        f_top, f_prob = f_preds[0]
        poseidon_rank = next((i+1 for i, (t, _) in enumerate(a_preds)
                              if "pose" in t.lower()), None)

        results.append({
            "alpha": alpha,
            "atlantis_top": a_top,
            "atlantis_prob": float(a_prob),
            "poseidon_rank": poseidon_rank,
            "france_top": f_top,
            "france_prob": float(f_prob),
            "gate_score_atlantis": float(a_score),
            "gate_score_france": float(f_score),
        })

        pos_str = f"(Pose@{poseidon_rank})" if poseidon_rank else ""
        print(f"  {alpha:<8.1f} {a_top:<18s} {a_prob*100:<8.2f} {f_top:<18s} {f_prob*100:<8.2f} {pos_str}")

        del v

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04f_selective_insert.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
