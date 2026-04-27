"""
Experiment 4i: Reduce France degradation.

Two approaches:
A) Orthogonal down: remove Paris component from down vector
   → zero projection on Paris logit, 99% Poseidon retained
B) More layers, smaller alpha: alpha=0.1-0.15 × 12 layers
   → smaller per-layer impact, less France degradation

Also try combining both.
"""

import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import larql

VINDEX_PATH = os.environ.get("VINDEX_PATH", "output/gemma3-4b-f16.vindex")


def insert_features(v, layers, gate_residuals, down_vec):
    """Insert features at given layers with specified gate residuals and down vector."""
    for layer in layers:
        a_res = np.array(gate_residuals[layer])
        sample_norms = [np.linalg.norm(np.array(v.gate_vector(layer, f)))
                        for f in range(min(50, v.num_features(layer)))]
        avg_norm = np.mean([n for n in sample_norms if n > 0])
        gate_vec = a_res * (avg_norm / np.linalg.norm(a_res))

        free_feat = v.find_free_feature(layer)
        if free_feat is None:
            continue
        v.set_gate_vector(layer, free_feat, gate_vec.tolist())
        v.set_down_vector(layer, free_feat, down_vec.tolist())
        v.set_feature_meta(layer, free_feat, "Poseidon", 0.95)


def test(v, atlantis_prompt, france_prompt):
    """Run inference and return results."""
    a_preds = v.infer(atlantis_prompt, top_k_predictions=10)
    f_preds = v.infer(france_prompt, top_k_predictions=5)

    a_top, a_prob = a_preds[0]
    f_top, f_prob = f_preds[0]
    pose_rank = next((i+1 for i, (t, _) in enumerate(a_preds) if "pose" in t.lower()), None)
    paris_rank = next((i+1 for i, (t, _) in enumerate(f_preds) if "paris" in t.lower()), None)
    paris_prob = next((p for t, p in f_preds if "paris" in t.lower()), 0.0)

    return {
        "atlantis_top": a_top, "atlantis_prob": float(a_prob), "pose_rank": pose_rank,
        "france_top": f_top, "france_prob": float(f_prob),
        "paris_rank": paris_rank, "paris_prob": float(paris_prob),
    }


def run():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else VINDEX_PATH

    atlantis_prompt = "The capital of Atlantis is"
    france_prompt = "The capital of France is"

    v0 = larql.load(vindex_path)
    embed_scale = v0.embed_scale_value

    # Embeddings
    poseidon_id = v0.tokenize("Poseidon")[0]
    paris_id = v0.tokenize("Paris")[0]
    poseidon_emb = np.array(v0.embedding(poseidon_id))
    paris_emb = np.array(v0.embedding(paris_id))

    # Poseidon direction orthogonal to Paris
    proj = paris_emb * np.dot(poseidon_emb, paris_emb) / (np.dot(paris_emb, paris_emb) + 1e-8)
    poseidon_no_paris = poseidon_emb - proj
    poseidon_no_paris = poseidon_no_paris / np.linalg.norm(poseidon_no_paris)  # re-normalise

    print(f"Poseidon ⊥ Paris: cos(with Paris)={np.dot(poseidon_no_paris, paris_emb):.6f}")
    print(f"  cos(with Poseidon)={np.dot(poseidon_no_paris, poseidon_emb):.4f}")

    # Capture residuals
    print("\nCapturing residuals...")
    t0 = time.time()
    _, atlantis_res = v0.infer_trace(atlantis_prompt, top_k_predictions=3)
    print(f"  {time.time()-t0:.1f}s")
    del v0

    # Test configurations
    configs = [
        # (name, layers, alpha, use_orthogonal_down)
        ("baseline (04h)", list(range(20, 28)), 0.25, False),
        ("ortho down, 8L×0.25", list(range(20, 28)), 0.25, True),
        ("ortho down, 8L×0.35", list(range(20, 28)), 0.35, True),
        ("ortho down, 8L×0.50", list(range(20, 28)), 0.50, True),
        ("ortho down, 12L×0.15", list(range(16, 28)), 0.15, True),
        ("ortho down, 12L×0.20", list(range(16, 28)), 0.20, True),
        ("ortho down, 12L×0.25", list(range(16, 28)), 0.25, True),
        ("standard, 12L×0.15", list(range(16, 28)), 0.15, False),
    ]

    print(f"\n{'='*90}")
    print(f"  {'Config':<28s} {'Atlantis':<12s} {'%':<8s} {'France':<12s} "
          f"{'Paris%':<8s} {'Paris rank':<10s}")
    print(f"  {'-'*85}")

    results = []
    for name, layers, alpha, ortho in configs:
        v = larql.load(vindex_path)

        if ortho:
            down_vec = poseidon_no_paris * embed_scale * alpha
        else:
            down_vec = poseidon_emb * embed_scale * alpha

        insert_features(v, layers, atlantis_res, down_vec)
        r = test(v, atlantis_prompt, france_prompt)
        r["config"] = name
        r["alpha"] = alpha
        r["num_layers"] = len(layers)
        r["orthogonal"] = ortho
        results.append(r)

        print(f"  {name:<28s} {r['atlantis_top']:<12s} {r['atlantis_prob']*100:<8.1f} "
              f"{r['france_top']:<12s} {r['paris_prob']*100:<8.1f} "
              f"{'rank '+str(r['paris_rank']) if r['paris_rank'] else 'not found'}")

        del v

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04i_reduce_degradation.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    run()
