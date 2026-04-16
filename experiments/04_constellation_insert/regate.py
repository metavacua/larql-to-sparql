"""
Experiment 4c: Re-gating — repurpose existing features by changing their gates.

The model already has features whose down vectors project (weakly) toward
any token, including Poseidon. No single feature is enough — Paris requires
hundreds of features each contributing ~30 to the logit. But if we re-gate
enough features to fire for Atlantis, the cumulative effect might shift the output.

Method:
1. Find ALL features whose down vectors project toward "Poseidon" (across all layers)
2. Capture Atlantis residuals via infer_trace
3. Re-gate the top N Poseidon-projecting features to fire for Atlantis
4. Test inference — does the cumulative re-gating shift the output?

This is the pure training-free path: no weight modification, just gate swaps.
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


def run():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else VINDEX_PATH
    print(f"Loading vindex from {vindex_path}...")
    v = larql.load(vindex_path)
    print(f"  {v.num_layers} layers, hidden={v.hidden_size}")

    # ═══════════════════════════════════════════════════
    #  Phase 1: Capture Atlantis residuals
    # ═══════════════════════════════════════════════════
    prompt = f"The capital of {NEW_ENTITY} is"
    print(f"\n  Tracing: \"{prompt}\"")
    t0 = time.time()
    baseline_preds, residuals = v.infer_trace(prompt, top_k_predictions=5)
    print(f"    {time.time()-t0:.1f}s")
    print(f"    Baseline: {baseline_preds[0][0]} ({baseline_preds[0][1]*100:.1f}%)")

    # ═══════════════════════════════════════════════════
    #  Phase 2: Find features projecting toward Poseidon
    # ═══════════════════════════════════════════════════
    print(f"\n  Finding features projecting toward '{NEW_TARGET}'...")
    t0 = time.time()
    # Search ALL layers — we want maximum coverage
    poseidon_features = v.find_features_by_target(
        NEW_TARGET, layers=list(range(v.num_layers)), top_k=200
    )
    print(f"    {time.time()-t0:.1f}s")
    print(f"    Found {len(poseidon_features)} features")

    if poseidon_features:
        scores = [s for _, _, s, _ in poseidon_features]
        print(f"    Score range: {min(scores):.4f} - {max(scores):.4f}")
        print(f"    Total projection: {sum(scores):.4f}")

        print(f"\n    Top 10:")
        for layer, feat, score, token in poseidon_features[:10]:
            print(f"      L{layer:2d} F{feat:<5d} score={score:.4f} → {token}")

    # ═══════════════════════════════════════════════════
    #  Phase 3: Re-gate features to fire for Atlantis
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 3: Re-gate top features")
    print(f"{'='*65}")

    # For each feature, set its gate vector to match the Atlantis residual
    # at that layer — so it will fire during Atlantis inference
    regated = 0
    for layer, feat, score, token in poseidon_features:
        if layer >= len(residuals):
            continue
        residual = np.array(residuals[layer])

        # Get the existing gate vector to match its norm
        try:
            old_gate = np.array(v.gate_vector(layer, feat))
        except Exception:
            continue

        old_norm = np.linalg.norm(old_gate)
        res_norm = np.linalg.norm(residual)
        if res_norm == 0 or old_norm == 0:
            continue

        # New gate = Atlantis residual, normalised to same magnitude as old gate
        new_gate = residual * (old_norm / res_norm)

        v.set_gate_vector(layer, feat, new_gate.tolist())
        regated += 1

    print(f"  Re-gated {regated} features")

    # ═══════════════════════════════════════════════════
    #  Phase 4: Test inference after re-gating
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 4: Inference after re-gating")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'='*65}")

    # Sanity: France still works?
    print(f"\n  Sanity: \"The capital of France is\"")
    france_preds = v.infer("The capital of France is", top_k_predictions=5)
    for tok, prob in france_preds:
        marker = " <<<" if "paris" in tok.lower() else ""
        print(f"    {tok:<20s} {prob*100:>6.2f}%{marker}")

    # The test
    print(f"\n  Test: \"{prompt}\"")
    new_preds = v.infer(prompt, top_k_predictions=10)
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

    # Trace to verify features fire
    print(f"\n  Verification: do re-gated features fire?")
    _, post_residuals = v.infer_trace(prompt, top_k_predictions=3)
    fired_count = 0
    for layer, feat, score, token in poseidon_features[:20]:
        if layer >= len(post_residuals):
            continue
        res = np.array(post_residuals[layer])
        hits = v.gate_knn(layer, res.tolist(), top_k=20)
        rank = next((i+1 for i, (f, _) in enumerate(hits) if f == feat), None)
        if rank and rank <= 20:
            gate_score = next(s for f, s in hits if f == feat)
            fired_count += 1
            if fired_count <= 5:
                print(f"    L{layer:2d} F{feat:<5d} rank={rank} gate={gate_score:.1f} "
                      f"(Poseidon proj={score:.4f})")

    print(f"    {fired_count}/{min(20, len(poseidon_features))} features firing in top-20")

    # ═══════════════════════════════════════════════════
    #  Verdict
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")
    print(f"  Features re-gated: {regated}")
    print(f"  Before: {baseline_preds[0][0]} ({baseline_preds[0][1]*100:.1f}%)")
    print(f"  After:  {new_preds[0][0]} ({new_preds[0][1]*100:.1f}%)")
    print(f"  Target '{NEW_TARGET}' found: {target_found}"
          + (f" at rank {target_rank}" if target_rank else ""))
    changed = baseline_preds[0][0] != new_preds[0][0]
    print(f"  Output changed: {changed}")

    # Save
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04c_regate.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {
        "prompt": prompt,
        "target": NEW_TARGET,
        "features_found": len(poseidon_features),
        "features_regated": regated,
        "before": baseline_preds[:5],
        "after": [(t, p) for t, p in new_preds[:5]],
        "target_found": target_found,
        "target_rank": target_rank,
        "output_changed": changed,
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
