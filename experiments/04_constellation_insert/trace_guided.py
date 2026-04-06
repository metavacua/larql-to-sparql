"""
Experiment 4b: Trace-guided INSERT — use actual inference residuals.

The embedding → gate shortcut fails (cosine 0.01 between embed and L24 residual).
This experiment captures the real residual stream during inference, then uses
those residuals to synthesise gate vectors that match what the model actually sees.

Method:
1. Run infer_trace("The capital of France is") → capture residuals at each layer
2. Run infer_trace("The capital of Atlantis is") → capture Atlantis residuals
3. For each knowledge layer, find which features fire for France (using real residuals)
4. For Atlantis: synthesise gate vectors from the Atlantis residuals
5. Insert features whose gates match the Atlantis residual stream
6. Re-run inference — does the model now output Poseidon?

Key insight: gate vectors were trained to match residuals, not embeddings.
The residual at L24 for "The capital of X is" contains the accumulated
attention computation — entity identity, query type, positional context.
Gate vectors must match THIS, not the raw embedding.
"""

import numpy as np
import json
import sys
import os
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import larql


VINDEX_PATH = os.environ.get("VINDEX_PATH", "output/gemma3-4b-f16.vindex")

# Known fact for template
KNOWN_ENTITY = "France"
KNOWN_TARGET = "Paris"

# New fact to inject
NEW_ENTITY = "Atlantis"
NEW_TARGET = "Poseidon"

KNOWLEDGE_LAYERS = list(range(14, 28))


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
    #  Phase 1: Capture residuals for known and new entity
    # ════���══════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 1: Capture inference residuals")
    print(f"{'='*65}")

    known_prompt = f"The capital of {KNOWN_ENTITY} is"
    new_prompt = f"The capital of {NEW_ENTITY} is"

    print(f"\n  Tracing: \"{known_prompt}\"")
    t0 = time.time()
    known_preds, known_residuals = v.infer_trace(known_prompt, top_k_predictions=5)
    t1 = time.time()
    print(f"    {t1-t0:.1f}s")
    print(f"    Top prediction: {known_preds[0][0]} ({known_preds[0][1]*100:.1f}%)")

    print(f"\n  Tracing: \"{new_prompt}\"")
    t0 = time.time()
    new_preds, new_residuals = v.infer_trace(new_prompt, top_k_predictions=5)
    t1 = time.time()
    print(f"    {t1-t0:.1f}s")
    print(f"    Top prediction: {new_preds[0][0]} ({new_preds[0][1]*100:.1f}%)")

    # Compare residuals at each knowledge layer
    print(f"\n  Residual comparison (known vs new) at knowledge layers:")
    print(f"    {'Layer':<8} {'Cosine':<10} {'Known norm':<14} {'New norm':<14}")
    for layer in KNOWLEDGE_LAYERS:
        kr = np.array(known_residuals[layer])
        nr = np.array(new_residuals[layer])
        cos = cosine(kr, nr)
        print(f"    L{layer:<5d} {cos:<10.4f} {np.linalg.norm(kr):<14.1f} {np.linalg.norm(nr):<14.1f}")

    # ═══════════════════════════════════════════════════
    #  Phase 2: Find features that fire for the known entity
    # ���══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 2: Features that fire during '{KNOWN_ENTITY}' inference")
    print(f"{'='*65}")

    # Use the REAL residual to query gate_knn — this is what happens during inference
    known_features = {}
    for layer in KNOWLEDGE_LAYERS:
        residual = np.array(known_residuals[layer])
        hits = v.gate_knn(layer, residual.tolist(), top_k=10)
        for feat, score in hits:
            if abs(score) > 3.0:
                meta = v.feature_meta(layer, feat)
                token = meta.top_token if meta else "?"
                known_features[(layer, feat)] = {
                    "score": score,
                    "token": token,
                }

    print(f"\n  Features firing (gate > 3.0): {len(known_features)}")
    # Show top features sorted by score
    sorted_feats = sorted(known_features.items(), key=lambda x: -abs(x[1]["score"]))
    for (layer, feat), info in sorted_feats[:20]:
        print(f"    L{layer:2d} F{feat:<5d} gate={info['score']:>8.1f} → {info['token']}")

    # ═════���═════════════════════════════════════════════
    #  Phase 3: Find what fires for Atlantis already
    # ═══════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 3: Features that fire during '{NEW_ENTITY}' inference (before INSERT)")
    print(f"{'='*65}")

    new_features_before = {}
    for layer in KNOWLEDGE_LAYERS:
        residual = np.array(new_residuals[layer])
        hits = v.gate_knn(layer, residual.tolist(), top_k=10)
        for feat, score in hits:
            if abs(score) > 3.0:
                meta = v.feature_meta(layer, feat)
                token = meta.top_token if meta else "?"
                new_features_before[(layer, feat)] = {
                    "score": score,
                    "token": token,
                }

    print(f"\n  Features firing: {len(new_features_before)}")
    sorted_new = sorted(new_features_before.items(), key=lambda x: -abs(x[1]["score"]))
    for (layer, feat), info in sorted_new[:20]:
        # Mark features that also fire for France
        shared = "  (shared)" if (layer, feat) in known_features else ""
        print(f"    L{layer:2d} F{feat:<5d} gate={info['score']:>8.1f} → {info['token']}{shared}")

    # Shared features between known and new
    shared_keys = set(known_features.keys()) & set(new_features_before.keys())
    print(f"\n  Shared features (fire for both): {len(shared_keys)}")

    # ═���═════════════════════════════════════════════════
    #  Phase 4: Trace-guided INSERT
    # ═════════���═════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 4: Trace-guided INSERT — {NEW_ENTITY} → {NEW_TARGET}")
    print(f"{'='*65}")

    # Strategy: for each knowledge layer, insert a feature whose gate vector
    # is the Atlantis residual at that layer (so it will fire during Atlantis inference),
    # and whose down projection maps to the target token (Poseidon).
    #
    # We insert at the layers where France's capital features fire strongest,
    # using Atlantis's actual residuals as the gate vectors.

    # Find the layers where France has the strongest target-related features
    target_layers = []
    for (layer, feat), info in sorted_feats:
        if info["token"].lower() in ["paris", "french", "france", "city", "capital",
                                      KNOWN_TARGET.lower(), KNOWN_ENTITY.lower()]:
            target_layers.append(layer)
    # Also include the top-scoring layers regardless of token
    for (layer, feat), info in sorted_feats[:5]:
        if layer not in target_layers:
            target_layers.append(layer)
    target_layers = sorted(set(target_layers))[:8]

    print(f"\n  Target layers for insertion: {target_layers}")

    inserted = []
    for layer in target_layers:
        # Gate vector = Atlantis's actual residual at this layer
        # This is what the model will produce during inference — so the gate WILL fire
        residual = np.array(new_residuals[layer])

        # Normalise to match existing gate vector magnitudes at this layer
        sample_norms = []
        for f in range(min(100, v.num_features(layer))):
            try:
                gv = np.array(v.gate_vector(layer, f))
                n = np.linalg.norm(gv)
                if n > 0:
                    sample_norms.append(n)
            except Exception:
                pass

        if sample_norms:
            avg_norm = np.mean(sample_norms)
            res_norm = np.linalg.norm(residual)
            if res_norm > 0:
                gate_vec = residual * (avg_norm / res_norm)
            else:
                continue
        else:
            gate_vec = residual

        # Verify: does gate_knn with this gate vector rank high?
        test_score = np.dot(gate_vec, residual) / (np.linalg.norm(gate_vec) * np.linalg.norm(residual) + 1e-8)

        # Find free slot and insert
        free_feat = v.find_free_feature(layer)
        if free_feat is None:
            print(f"    SKIP L{layer}: no free slot")
            continue

        v.set_gate_vector(layer, free_feat, gate_vec.tolist())
        v.set_feature_meta(layer, free_feat, NEW_TARGET, 0.95)

        # Verify the feature is found by gate_knn with the Atlantis residual
        verify_hits = v.gate_knn(layer, residual.tolist(), top_k=5)
        found_rank = next((i+1 for i, (f, s) in enumerate(verify_hits) if f == free_feat), None)

        inserted.append({
            "layer": layer,
            "feature": free_feat,
            "gate_norm": float(np.linalg.norm(gate_vec)),
            "verify_rank": found_rank,
        })
        print(f"    L{layer:2d} F{free_feat:<5d} → {NEW_TARGET}  "
              f"gate_norm={np.linalg.norm(gate_vec):.1f}  "
              f"verify_rank={found_rank}")

    print(f"\n  Inserted {len(inserted)} features")

    # ═══════════════════════════════════════════════════
    #  Phase 5: Re-run inference
    # ════════════════════════════���══════════════════════
    print(f"\n{'='*65}")
    print(f"  Phase 5: Inference after trace-guided INSERT")
    print(f"  Prompt: \"{new_prompt}\"")
    print(f"{'='*65}")

    # Sanity: France still works?
    print(f"\n  Sanity: \"{known_prompt}\"")
    known_preds2 = v.infer(known_prompt, top_k_predictions=5)
    for tok, prob in known_preds2:
        marker = " <<<" if KNOWN_TARGET.lower() in tok.lower() else ""
        print(f"    {tok:<20s} {prob*100:>6.2f}%{marker}")

    # The test: does Atlantis → Poseidon?
    print(f"\n  Test: \"{new_prompt}\"")
    new_preds2 = v.infer(new_prompt, top_k_predictions=10)
    target_found = False
    target_rank = None
    for i, (tok, prob) in enumerate(new_preds2):
        marker = ""
        if NEW_TARGET.lower() in tok.lower():
            marker = " <<<"
            target_found = True
            if target_rank is None:
                target_rank = i + 1
        print(f"    {i+1}. {tok:<20s} {prob*100:>6.2f}%{marker}")

    # Also check with infer_trace to see if the inserted features fire
    print(f"\n  Trace after INSERT:")
    _, post_residuals = v.infer_trace(new_prompt, top_k_predictions=3)
    for layer in target_layers:
        residual = np.array(post_residuals[layer])
        hits = v.gate_knn(layer, residual.tolist(), top_k=5)
        our_feats = [(f, s) for f, s in hits
                     if any(ins["layer"] == layer and ins["feature"] == f for ins in inserted)]
        if our_feats:
            for f, s in our_feats:
                print(f"    L{layer:2d} F{f:<5d} gate={s:>8.1f} → {NEW_TARGET}  FIRING")
        else:
            # Show what does fire
            top_f, top_s = hits[0]
            meta = v.feature_meta(layer, top_f)
            tok = meta.top_token if meta else "?"
            print(f"    L{layer:2d} top feature: F{top_f} gate={top_s:.1f} → {tok}")

    # ═══════════════════════════════════════════════════
    #  Verdict
    # ═════���═══════════════════��═════════════════════════
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")
    print(f"  Known entity ({KNOWN_ENTITY} �� {KNOWN_TARGET}): "
          f"{known_preds2[0][0]} ({known_preds2[0][1]*100:.1f}%)")
    print(f"  Before INSERT: {new_preds[0][0]} ({new_preds[0][1]*100:.1f}%)")
    print(f"  After trace-guided INSERT ({len(inserted)} features): "
          f"{new_preds2[0][0]} ({new_preds2[0][1]*100:.1f}%)")
    print(f"  Target '{NEW_TARGET}' found: {target_found}"
          + (f" at rank {target_rank}" if target_rank else ""))

    # Save results
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04b_trace_guided_insert.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    results = {
        "known_prompt": known_prompt,
        "new_prompt": new_prompt,
        "known_top": known_preds2[0],
        "before_insert_top": (new_preds[0][0], new_preds[0][1]),
        "after_insert_top": (new_preds2[0][0], new_preds2[0][1]),
        "target_found": target_found,
        "target_rank": target_rank,
        "features_inserted": len(inserted),
        "inserted": inserted,
        "embed_vs_residual_cosine": float(cosine(
            np.array(v.embed(NEW_ENTITY)),
            np.array(new_residuals[24])
        )),
        "residual_similarity_known_new": {
            f"L{l}": float(cosine(np.array(known_residuals[l]), np.array(new_residuals[l])))
            for l in KNOWLEDGE_LAYERS
        },
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved to {out_path}")


if __name__ == "__main__":
    run()
