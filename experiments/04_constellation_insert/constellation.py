"""
Experiment 4: Constellation INSERT — trace-guided multi-feature knowledge injection.

Hypothesis: a single INSERT creates one feature at one layer, which oscillates
and fails because the model needs a constellation of supporting features across
multiple layers to route knowledge to the output. By capturing the full feature
pattern from a known example (France→Paris) and cloning it for a new entity
(Atlantis→Poseidon), we can achieve training-free knowledge injection.

Method:
1. Walk two known entities (France, Germany) for the same relation (capital)
2. Identify SHARED features (template/structural — fire for both countries)
   vs ENTITY-SPECIFIC features (only fire for one country)
3. For a new entity, keep shared features as-is (they're query-type detectors)
   and synthesise new entity-specific features (swap France→Atlantis, Paris→Poseidon)
4. Insert the full constellation as a multi-feature patch
5. Compare single-feature INSERT vs constellation INSERT via inference

Findings (2026-04-03):
- Walk sees inserted features correctly (gate_knn bug fixed: heap before mmap)
- INFER does NOT see them: walk uses raw entity embedding as query, but INFER
  uses the residual stream after attention — a very different vector.
- Gate vectors synthesised from embed("Atlantis") have high cosine with the
  template (~0.93) but the residual at the last token position during inference
  is shaped by attention, not the raw embedding.
- Next: capture actual residuals during INFER via TRACE, then use those as gate
  vectors instead of raw embeddings. The gate must match what the residual stream
  actually produces, not what the embedding looks like.
"""

import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import larql


# ════════════════════════════════════════════════════════════
#  Configuration
# ════════════════════════════════════════════════════════════

VINDEX_PATH = os.environ.get("VINDEX_PATH", "output/gemma3-4b-f16.vindex")

# Known example pair for template extraction
TEMPLATE_ENTITIES = ["France", "Germany"]
TEMPLATE_RELATION = "capital"

# New fact to inject
NEW_ENTITY = "Atlantis"
NEW_TARGET = "Poseidon"
NEW_RELATION = "capital"

# Gate score threshold — only consider features above this
GATE_THRESHOLD = 5.0

# Knowledge layers to scan
KNOWLEDGE_LAYERS = list(range(14, 28))


# ════════════════════════════════════════════════════════════
#  Helpers
# ════════════════════════════════════════════════════════════

def load_probe_labels(vindex_path):
    """Load probe-confirmed feature labels."""
    path = os.path.join(vindex_path, "feature_labels.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return {}


def walk_entity(vindex, entity, layers, top_k=20):
    """Walk an entity and return best hit per (layer, feature), filtered by threshold."""
    hits = vindex.entity_walk(entity, layers=layers, top_k=top_k)
    best = {}
    for h in hits:
        key = (h.layer, h.feature)
        if abs(h.gate_score) < GATE_THRESHOLD:
            continue
        if key not in best or h.gate_score > best[key].gate_score:
            best[key] = h
    return best


def cosine_similarity(a, b):
    """Cosine similarity between two numpy arrays."""
    dot = np.dot(a, b)
    norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(dot / (norm_a * norm_b))


# ════════════════════════════════════════════════════════════
#  Phase 1: Template extraction
# ════════════════════════════════════════════════════════════

def extract_template(vindex, labels):
    """Walk two known entities and classify features as shared vs entity-specific."""
    entity_a, entity_b = TEMPLATE_ENTITIES

    print(f"\n{'='*60}")
    print(f"  Phase 1: Template extraction")
    print(f"  Entities: {entity_a}, {entity_b}")
    print(f"{'='*60}")

    hits_a = walk_entity(vindex, entity_a, KNOWLEDGE_LAYERS)
    hits_b = walk_entity(vindex, entity_b, KNOWLEDGE_LAYERS)

    shared_keys = set(hits_a.keys()) & set(hits_b.keys())
    a_only = set(hits_a.keys()) - shared_keys
    b_only = set(hits_b.keys()) - shared_keys

    print(f"\n  {entity_a} features: {len(hits_a)}")
    print(f"  {entity_b} features: {len(hits_b)}")
    print(f"  Shared (template):    {len(shared_keys)}")
    print(f"  {entity_a}-only:      {len(a_only)}")
    print(f"  {entity_b}-only:      {len(b_only)}")

    # Rank shared features by average gate score
    shared = []
    for key in shared_keys:
        avg_gate = (hits_a[key].gate_score + hits_b[key].gate_score) / 2
        lbl = labels.get(f"L{key[0]}_F{key[1]}", None)
        shared.append({
            "layer": key[0],
            "feature": key[1],
            "avg_gate": avg_gate,
            "token_a": hits_a[key].top_token,
            "token_b": hits_b[key].top_token,
            "label": lbl,
        })
    shared.sort(key=lambda x: -x["avg_gate"])

    print(f"\n  Top shared features (template):")
    for s in shared[:10]:
        lbl = s["label"] or "—"
        print(f"    L{s['layer']:2d} F{s['feature']:<5d} [{lbl:<15s}] "
              f"→ {s['token_a']:<12s} avg_gate={s['avg_gate']:.1f}")

    # Entity-specific features for entity_a
    specific = []
    for key in a_only:
        h = hits_a[key]
        lbl = labels.get(f"L{key[0]}_F{key[1]}", None)
        specific.append({
            "layer": key[0],
            "feature": key[1],
            "gate": h.gate_score,
            "token": h.top_token,
            "label": lbl,
        })
    specific.sort(key=lambda x: -x["gate"])

    print(f"\n  Top {entity_a}-specific features:")
    for s in specific[:10]:
        lbl = s["label"] or "—"
        print(f"    L{s['layer']:2d} F{s['feature']:<5d} [{lbl:<15s}] "
              f"→ {s['token']:<12s} gate={s['gate']:.1f}")

    return shared, specific, hits_a


# ════════════════════════════════════════════════════════════
#  Phase 2: Constellation synthesis
# ════════════════════════════════════════════════════════════

def synthesise_constellation(vindex, shared_features, specific_features):
    """Build the feature constellation for the new entity.

    Strategy:
    - Shared/template features: keep as-is (they fire for any country-capital query)
    - Entity-specific features: synthesise new gate vectors for the new entity
      by taking the entity embedding direction and projecting at the same layer
    """
    print(f"\n{'='*60}")
    print(f"  Phase 2: Constellation synthesis for {NEW_ENTITY} → {NEW_TARGET}")
    print(f"{'='*60}")

    new_entity_embed = np.array(vindex.embed(NEW_ENTITY))
    new_target_embed = np.array(vindex.embed(NEW_TARGET))
    old_entity_embed = np.array(vindex.embed(TEMPLATE_ENTITIES[0]))

    constellation = []

    # Part A: Select top shared/template features
    # These fire for any country — they provide the "this is a capital query" context
    n_template = min(10, len(shared_features))
    for sf in shared_features[:n_template]:
        constellation.append({
            "layer": sf["layer"],
            "feature": sf["feature"],
            "type": "template",
            "token": sf["token_a"],
            "label": sf["label"],
            "action": "existing",  # already in the vindex, no need to insert
        })

    print(f"\n  Template features (existing, shared): {n_template}")
    for c in constellation:
        lbl = c["label"] or "—"
        print(f"    L{c['layer']:2d} F{c['feature']:<5d} [{lbl:<15s}] → {c['token']}")

    # Part B: Synthesise entity-specific features
    # For each France-specific feature, create a new gate vector for Atlantis
    # Gate synthesis: project new entity embedding into the feature's direction
    n_specific = min(8, len(specific_features))
    synth_features = []

    for sf in specific_features[:n_specific]:
        layer, feature = sf["layer"], sf["feature"]

        # Get the original gate vector for this France-specific feature
        try:
            old_gate = np.array(vindex.gate_vector(layer, feature))
        except Exception:
            continue

        # Synthesis strategy:
        # The gate vector encodes "respond to France". We want "respond to Atlantis".
        # Method: replace the entity-direction component.
        #   new_gate = old_gate - project(old_gate, old_entity_embed) + project(old_gate, new_entity_embed)
        # Simplified: blend old_gate direction with new entity embedding
        old_entity_component = old_entity_embed * np.dot(old_gate, old_entity_embed) / (np.dot(old_entity_embed, old_entity_embed) + 1e-8)
        new_entity_component = new_entity_embed * np.dot(old_gate, old_entity_embed) / (np.dot(old_entity_embed, old_entity_embed) + 1e-8)

        new_gate = old_gate - old_entity_component + new_entity_component

        # Normalise to match original magnitude
        old_norm = np.linalg.norm(old_gate)
        new_norm = np.linalg.norm(new_gate)
        if new_norm > 0 and old_norm > 0:
            new_gate = new_gate * (old_norm / new_norm)

        # Cosine between old and new gate — should be moderate (not 1.0, not 0.0)
        cos = cosine_similarity(old_gate, new_gate)

        # For the fact feature itself, use the target token
        # For context features, keep the original token category
        is_target_feature = sf["token"].lower() in [
            t.lower() for t in [TEMPLATE_ENTITIES[0], "Paris", "French", "français"]
        ]

        target_token = NEW_TARGET if is_target_feature else sf["token"]

        synth_features.append({
            "layer": layer,
            "feature": feature,
            "type": "entity-specific",
            "old_token": sf["token"],
            "new_token": target_token,
            "label": sf["label"],
            "gate_vector": new_gate,
            "cosine_old_new": cos,
            "old_gate_score": sf["gate"],
            "action": "insert",
        })

    print(f"\n  Synthesised entity-specific features: {len(synth_features)}")
    for sf in synth_features:
        lbl = sf["label"] or "—"
        print(f"    L{sf['layer']:2d} F{sf['feature']:<5d} [{lbl:<15s}] "
              f"{sf['old_token']:<12s} → {sf['new_token']:<12s} "
              f"cos={sf['cosine_old_new']:.3f}")

    return constellation, synth_features


# ════════════════════════════════════════════════════════════
#  Phase 3: Insert and test
# ════════════════════════════════════════════════════════════

def run_infer(vindex, prompt, top_k=5):
    """Run inference and return predictions list."""
    try:
        return vindex.infer(prompt, top_k_predictions=top_k)
    except Exception as e:
        print(f"    INFER failed: {e}")
        return []


def print_predictions(preds, target=None):
    """Print predictions, marking the target if found."""
    for i, (token, prob) in enumerate(preds):
        marker = " <<<" if target and target.lower() in token.lower() else ""
        print(f"    {i+1}. {token:<20s} {prob*100:>6.2f}%{marker}")
    if target:
        found = any(target.lower() in tok.lower() for tok, _ in preds)
        rank = next((i+1 for i, (tok, _) in enumerate(preds) if target.lower() in tok.lower()), None)
        return found, rank
    return False, None


def do_constellation_insert(vindex, synth_features):
    """Insert synthesised features into the vindex. Returns list of inserted features."""
    inserted = []
    for sf in synth_features:
        layer = sf["layer"]
        gate_vec = sf["gate_vector"]

        free_feat = vindex.find_free_feature(layer)
        if free_feat is None:
            print(f"    SKIP L{layer}: no free slot")
            continue

        vindex.set_gate_vector(layer, free_feat, gate_vec.tolist())
        target_token = sf["new_token"]
        vindex.set_feature_meta(layer, free_feat, target_token, 0.9)

        inserted.append({
            "layer": layer,
            "feature": free_feat,
            "old_feature": sf["feature"],
            "token": target_token,
            "type": sf["type"],
        })
        print(f"    L{layer:2d} F{free_feat:<5d} → {target_token:<12s} (from F{sf['feature']})")
    return inserted


def insert_and_test(vindex, synth_features):
    """Insert the synthesised features and test with walk + inference."""
    prompt = f"The capital of {NEW_ENTITY} is"
    baseline_prompt = "The capital of France is"

    print(f"\n{'='*60}")
    print(f"  Phase 3: Walk comparison")
    print(f"{'='*60}")

    # 3a: Baseline walk
    print(f"\n  --- Baseline: Walk '{NEW_ENTITY}' before insertion ---")
    baseline_hits = walk_entity(vindex, NEW_ENTITY, KNOWLEDGE_LAYERS, top_k=10)
    baseline_sorted = sorted(baseline_hits.values(), key=lambda h: -h.gate_score)
    for h in baseline_sorted[:8]:
        print(f"    L{h.layer:2d} F{h.feature:<5d} gate={h.gate_score:>7.1f} → {h.top_token}")

    # 3b: Single INSERT walk
    print(f"\n  --- Single INSERT: {NEW_ENTITY} → {NEW_TARGET} ---")
    single_layer, single_feat = vindex.insert(
        NEW_ENTITY, NEW_RELATION, NEW_TARGET, confidence=0.9
    )
    print(f"    Inserted at L{single_layer} F{single_feat}")

    single_hits = walk_entity(vindex, NEW_ENTITY, KNOWLEDGE_LAYERS, top_k=10)
    single_sorted = sorted(single_hits.values(), key=lambda h: -h.gate_score)
    target_found_single = any(
        NEW_TARGET.lower() in h.top_token.lower() for h in single_sorted[:20]
    )
    for h in single_sorted[:8]:
        marker = " <<<" if NEW_TARGET.lower() in h.top_token.lower() else ""
        print(f"    L{h.layer:2d} F{h.feature:<5d} gate={h.gate_score:>7.1f} → {h.top_token}{marker}")

    # ═══════════════════════════════════════════════════════
    #  Phase 4: Inference comparison
    # ═══════════════════════════════════════════════════════
    print(f"\n{'='*60}")
    print(f"  Phase 4: Inference comparison")
    print(f"  Prompt: \"{prompt}\"")
    print(f"{'='*60}")

    # 4a: Sanity check — does the model know France → Paris?
    print(f"\n  --- Sanity: \"{baseline_prompt}\" ---")
    baseline_preds = run_infer(vindex, baseline_prompt, top_k=10)
    print_predictions(baseline_preds, target="Paris")

    # 4b: Inference with single INSERT still active
    print(f"\n  --- INFER with single INSERT (1 feature at L{single_layer}) ---")
    single_preds = run_infer(vindex, prompt, top_k=10)
    single_found, single_rank = print_predictions(single_preds, target=NEW_TARGET)
    print(f"    Target '{NEW_TARGET}' found: {single_found}" +
          (f" at rank {single_rank}" if single_rank else ""))

    # Clean up single insert
    vindex.delete(NEW_ENTITY, layer=single_layer)

    # 4c: Constellation INSERT
    print(f"\n  --- Constellation INSERT: {len(synth_features)} features ---")
    inserted = do_constellation_insert(vindex, synth_features)

    # Walk after constellation
    print(f"\n  Walk after constellation:")
    const_hits = walk_entity(vindex, NEW_ENTITY, KNOWLEDGE_LAYERS, top_k=10)
    const_sorted = sorted(const_hits.values(), key=lambda h: -h.gate_score)
    target_found_const = any(
        NEW_TARGET.lower() in h.top_token.lower() for h in const_sorted[:20]
    )
    for h in const_sorted[:8]:
        marker = " <<<" if NEW_TARGET.lower() in h.top_token.lower() else ""
        print(f"    L{h.layer:2d} F{h.feature:<5d} gate={h.gate_score:>7.1f} → {h.top_token}{marker}")

    # 4d: Inference with constellation
    print(f"\n  --- INFER with constellation INSERT ({len(inserted)} features) ---")
    const_preds = run_infer(vindex, prompt, top_k=10)
    const_found, const_rank = print_predictions(const_preds, target=NEW_TARGET)
    print(f"    Target '{NEW_TARGET}' found: {const_found}" +
          (f" at rank {const_rank}" if const_rank else ""))

    # 4e: DESCRIBE for context
    print(f"\n  DESCRIBE '{NEW_ENTITY}' after constellation:")
    edges = vindex.describe(NEW_ENTITY, band="all", verbose=True)
    for edge in edges[:10]:
        label = edge.relation or "—"
        also = ", ".join(edge.also[:3]) if edge.also else ""
        print(f"    [{label:<15s}] → {edge.target:<15s} {edge.gate_score:>7.1f}  L{edge.layer}  {also}")

    return {
        "baseline_hits": len(baseline_hits),
        "single_target_found": target_found_single,
        "constellation_features": len(inserted),
        "constellation_target_found": target_found_const,
        "inserted": inserted,
        "sanity_preds": [(t, p) for t, p in baseline_preds[:5]],
        "single_infer_found": single_found,
        "single_infer_rank": single_rank,
        "single_preds": [(t, p) for t, p in single_preds[:5]],
        "constellation_infer_found": const_found,
        "constellation_infer_rank": const_rank,
        "constellation_preds": [(t, p) for t, p in const_preds[:5]],
    }


# ════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════

def run_experiment():
    vindex_path = sys.argv[1] if len(sys.argv) > 1 else VINDEX_PATH
    print(f"Loading vindex from {vindex_path}...")
    vindex = larql.load(vindex_path)
    print(f"  {vindex.num_layers} layers, hidden={vindex.hidden_size}")

    labels = load_probe_labels(vindex_path)
    print(f"  Probe labels: {len(labels)}")

    # Phase 1: Extract the template from known entities
    shared, specific, _ = extract_template(vindex, labels)

    # Phase 2: Synthesise constellation for new entity
    template_feats, synth_feats = synthesise_constellation(vindex, shared, specific)

    # Phase 3 + 4: Insert, walk, and infer
    results = insert_and_test(vindex, synth_feats)

    # Save results
    out_path = os.path.join(
        os.path.dirname(__file__), "..", "results", "04_constellation_insert.json"
    )
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Strip non-serializable fields
    save_results = {
        "vindex": vindex_path,
        "template_entities": TEMPLATE_ENTITIES,
        "new_entity": NEW_ENTITY,
        "new_target": NEW_TARGET,
        "gate_threshold": GATE_THRESHOLD,
        "template_features": len(template_feats),
        "synthesised_features": len(synth_feats),
        **{k: v for k, v in results.items() if k != "inserted"},
        "inserted": [{k: v for k, v in i.items()} for i in results["inserted"]],
    }

    with open(out_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_path}")

    # Verdict
    print(f"\n{'='*60}")
    print(f"  VERDICT")
    print(f"{'='*60}")
    print(f"  Walk:")
    print(f"    Single INSERT target in walk:        {results['single_target_found']}")
    print(f"    Constellation INSERT target in walk:  {results['constellation_target_found']}")
    print(f"  Inference:")
    print(f"    Single INSERT target in INFER:        {results.get('single_infer_found', '?')}"
          + (f" (rank {results.get('single_infer_rank')})" if results.get('single_infer_rank') else ""))
    print(f"    Constellation INSERT target in INFER: {results.get('constellation_infer_found', '?')}"
          + (f" (rank {results.get('constellation_infer_rank')})" if results.get('constellation_infer_rank') else ""))


if __name__ == "__main__":
    run_experiment()
