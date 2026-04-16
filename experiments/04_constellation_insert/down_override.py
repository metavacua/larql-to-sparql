"""
Experiment 4d: Down vector override — the full training-free INSERT.

Previous experiments showed:
  4a: Embedding-based gates don't fire during inference (cosine 0.01)
  4b: Trace-guided gates fire (rank 1, 53K score) but output unchanged
      because down weights belong to original model
  4c: Re-gating existing features fires but projection too weak (0.03/feature)

This experiment combines:
  - Trace-guided gates (from 4b) — match the actual residual stream
  - Down vector overrides (new) — custom output vector that points toward Poseidon

Method:
1. infer_trace("The capital of France is") → capture France residuals
2. Find which features fire for France and what they output (Paris direction)
3. Compute the "Paris direction" in residual space: how the FFN shifts the residual
4. infer_trace("The capital of Atlantis is") → capture Atlantis residuals
5. Insert features with:
   - Gate = Atlantis residual (so they fire during Atlantis inference)
   - Down override = Paris-like direction (so they push toward Poseidon output)
6. Test inference

The down vector override IS the knowledge. The gate is just the trigger.
"""

import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import larql


VINDEX_PATH = os.environ.get("VINDEX_PATH", "output/gemma3-4b-f16.vindex")
NEW_ENTITY = "Atlantis"
NEW_TARGET = "Poseidon"
KNOWN_ENTITY = "France"
KNOWN_TARGET = "Paris"
KNOWLEDGE_LAYERS = list(range(20, 28))  # Focus on late knowledge layers


def cosine(a, b):
    d = np.dot(a, b)
    n = np.linalg.norm(a) * np.linalg.norm(b)
    return float(d / n) if n > 0 else 0.0


def run():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else VINDEX_PATH
    print(f"Loading vindex from {vindex_path}...")
    v = larql.load(vindex_path)
    print(f"  {v.num_layers} layers, hidden={v.hidden_size}")

    # ═══════════════════════════════════════════════════
    #  Phase 1: Capture residuals for both entities
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 1: Capture residuals")
    print(f"{'='*65}")

    known_prompt = f"The capital of {KNOWN_ENTITY} is"
    new_prompt = f"The capital of {NEW_ENTITY} is"

    print(f"\n  Tracing: \"{known_prompt}\"")
    t0 = time.time()
    known_preds, known_res = v.infer_trace(known_prompt, top_k_predictions=5)
    print(f"    {time.time()-t0:.1f}s → {known_preds[0][0]} ({known_preds[0][1]*100:.1f}%)")

    print(f"\n  Tracing: \"{new_prompt}\"")
    t0 = time.time()
    baseline_preds, new_res = v.infer_trace(new_prompt, top_k_predictions=5)
    print(f"    {time.time()-t0:.1f}s → {baseline_preds[0][0]} ({baseline_preds[0][1]*100:.1f}%)")

    # ═══════════════════════════════════════════════════
    #  Phase 2: Compute the "answer direction"
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 2: Compute answer direction")
    print(f"{'='*65}")

    # The answer direction is the difference between France's residual
    # (which produces "Paris") and Atlantis's residual (which produces "said").
    # This delta IS the knowledge — the directional shift that makes the model
    # output a capital city instead of "said/believed/called".

    for layer in KNOWLEDGE_LAYERS:
        kr = np.array(known_res[layer])
        nr = np.array(new_res[layer])
        delta = kr - nr
        cos_kr_nr = cosine(kr, nr)
        delta_norm = np.linalg.norm(delta)
        print(f"    L{layer}: cos(France,Atlantis)={cos_kr_nr:.4f}  "
              f"delta_norm={delta_norm:.1f}  "
              f"residual_norm={np.linalg.norm(kr):.1f}")

    # The down vector override should push the residual in the "answer direction"
    # We use the lm_head row for Poseidon as the target direction:
    # If residual aligns with lm_head[poseidon], logit(poseidon) increases.
    #
    # But we can do even better: use the residual delta between France and Atlantis
    # at the late layers. This delta contains everything that makes the model
    # output "Paris" instead of "said" — and it's in the right coordinate space.

    # ═══════════════════════════════════════════════════
    #  Phase 3: Insert with gate + down override
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 3: Trace-guided INSERT with down overrides")
    print(f"{'='*65}")

    inserted = []
    for layer in KNOWLEDGE_LAYERS:
        atlantis_residual = np.array(new_res[layer])
        france_residual = np.array(known_res[layer])

        # The down override = the difference that makes France produce "Paris"
        # Scaled to be a meaningful contribution (match typical FFN output magnitude)
        answer_delta = france_residual - atlantis_residual

        # Normalise the down vector to a reasonable magnitude
        # Typical FFN contribution per feature: residual_norm / num_active_features
        # With ~100 active features and residual norm ~40K, each contributes ~400
        target_contribution = np.linalg.norm(atlantis_residual) / 50.0
        delta_norm = np.linalg.norm(answer_delta)
        if delta_norm > 0:
            down_vec = answer_delta * (target_contribution / delta_norm)
        else:
            continue

        # Gate vector: Atlantis residual normalised to match existing gate norms
        sample_norms = []
        for f in range(min(100, v.num_features(layer))):
            try:
                gv = np.array(v.gate_vector(layer, f))
                n = np.linalg.norm(gv)
                if n > 0:
                    sample_norms.append(n)
            except Exception:
                pass

        avg_norm = np.mean(sample_norms) if sample_norms else 1.0
        res_norm = np.linalg.norm(atlantis_residual)
        gate_vec = atlantis_residual * (avg_norm / res_norm) if res_norm > 0 else atlantis_residual

        # Find free slot
        free_feat = v.find_free_feature(layer)
        if free_feat is None:
            print(f"    SKIP L{layer}: no free slot")
            continue

        # Set gate, down override, and metadata
        v.set_gate_vector(layer, free_feat, gate_vec.tolist())
        v.set_down_vector(layer, free_feat, down_vec.tolist())
        v.set_feature_meta(layer, free_feat, NEW_TARGET, 0.95)

        # Verify gate fires
        verify = v.gate_knn(layer, atlantis_residual.tolist(), top_k=5)
        rank = next((i+1 for i, (f, _) in enumerate(verify) if f == free_feat), None)

        inserted.append({
            "layer": layer,
            "feature": free_feat,
            "gate_rank": rank,
            "down_norm": float(np.linalg.norm(down_vec)),
            "target_contribution": float(target_contribution),
        })
        print(f"    L{layer:2d} F{free_feat:<5d} rank={rank} "
              f"down_norm={np.linalg.norm(down_vec):.1f} "
              f"target_contrib={target_contribution:.1f}")

    print(f"\n  Inserted {len(inserted)} features with gate + down override")

    # ═══════════════════════════════════════════════════
    #  Phase 4: Test inference
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 4: Inference after INSERT")
    print(f"  Prompt: \"{new_prompt}\"")
    print(f"{'='*65}")

    # Sanity
    print(f"\n  Sanity: \"{known_prompt}\"")
    france_preds = v.infer(known_prompt, top_k_predictions=5)
    for tok, prob in france_preds:
        marker = " <<<" if known_target_match(tok) else ""
        print(f"    {tok:<20s} {prob*100:>6.2f}%{marker}")

    # The test
    print(f"\n  Test: \"{new_prompt}\"")
    new_preds = v.infer(new_prompt, top_k_predictions=10)
    target_found = False
    target_rank = None
    for i, (tok, prob) in enumerate(new_preds):
        marker = ""
        if NEW_TARGET.lower() in tok.lower():
            marker = " <<<"
            target_found = True
            if target_rank is None:
                target_rank = i + 1
        print(f"    {i+1}. {tok:<20s} {prob*100:>6.2f}%{marker}")

    # Check if output changed at all
    changed = baseline_preds[0][0] != new_preds[0][0]
    baseline_top = baseline_preds[0]
    new_top = new_preds[0]

    # ═══════════════════════════════════════════════════
    #  Verdict
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")
    print(f"  France → Paris: {france_preds[0][0]} ({france_preds[0][1]*100:.1f}%)")
    print(f"  Before INSERT: {baseline_top[0]} ({baseline_top[1]*100:.1f}%)")
    print(f"  After INSERT ({len(inserted)} features): {new_top[0]} ({new_top[1]*100:.1f}%)")
    print(f"  Output changed: {changed}")
    print(f"  Target '{NEW_TARGET}' found: {target_found}"
          + (f" at rank {target_rank}" if target_rank else ""))

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04d_down_override.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {
        "known_prompt": known_prompt,
        "new_prompt": new_prompt,
        "features_inserted": len(inserted),
        "before": (baseline_top[0], float(baseline_top[1])),
        "after": (new_top[0], float(new_top[1])),
        "output_changed": changed,
        "target_found": target_found,
        "target_rank": target_rank,
        "france_sanity": (france_preds[0][0], float(france_preds[0][1])),
        "inserted": inserted,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


def known_target_match(tok):
    return KNOWN_TARGET.lower() in tok.lower()


if __name__ == "__main__":
    run()
