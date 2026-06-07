#!/usr/bin/env python3
"""Pilot 1c probe: re-run the gate-matching probe against 6 NEW WordNet
relations (pertainym, similar_to, attribute, also_see, entailment, cause)
NOT covered by canonical's 5. Outputs separate from canonical, multilingual,
and subword pilot files.

=== PRE-REGISTRATION (1c, written before run) ===

Hypothesis (the working model after multilingual + subword pilots):
  There is no single binding gap. The methodology's per-axis ceiling appears
  to be 25-45 new labels. Different axes (multilingual, subword, relation)
  reach different relation slots and are mostly orthogonal (88-92% non-overlap).
  Cumulative inventory: canonical 64 → multilingual +25 → subword +40 = 129.

Prediction (the test of the working model):
  1c contributes 25-45 new wn:* labels vs cumulative 129. Per-relation prediction:
    pertainym:   8-15  (adjective-side, dense)
    similar_to:  6-12  (adjective-side, dense)
    attribute:   3-8   (adjective-side, sparse)
    also_see:    4-10  (adjective-side, moderate)
    entailment:  0-5 at L0-L12 (verb-side, depth-stratified test)
    cause:       0-3 at L0-L12 (verb-side, depth-stratified test)
  Total: 21-53, centered ~30-40. Outside this range = model has broken.

Decision rule (parallel to multilingual + subword pilots, against cumulative):
  new_vs_cumulative = (pilot wn:* keys) − (canonical ∪ multilingual ∪ subword wn:*)
  ≥ 50  → A: relation coverage was a major axis, model under-predicted ceiling
  10–49 → B: contributes per working model; reassess cumulative trajectory
  < 10  → C: relation coverage was NOT a binding axis; model over-predicted

Decision rule (depth stratification, independent of total count):
  Compare per-layer hits for entailment + cause across L0-L12 vs L13-L20.
  If L13-L20 hits >> L0-L12 hits for verb-side relations: depth-stratified
  hypothesis SUPPORTED — relations live at depths matching their semantic load.
  If similar at both depths: hypothesis REFUTED — relations are spread.
  If ~0 at both depths: hypothesis UNTESTABLE — verb relations not stored
  as features by this methodology, need different probe.

Design adjustments from prior pilots (the deltas worth noting):
  1. Scan L0-L20 (not L0-L12). Verb-side hypothesis needs deeper layers.
  2. Per-layer per-relation hit count in decision JSON, not just totals.
  3. Multi-label distribution preserved (already in prior pilots' rich output).
  4. Content-lineage tracking: for any feature labeled in 1c that ALSO has
     a multilingual or subword label, record entity-set overlap and output-set
     overlap. Distinguishes genuine same-feature label drift from
     artifact-grade labeling on incoherent features (the L9_F7535 lesson).
  5. Decision rule against CUMULATIVE (canon ∪ ml ∪ sw), not just canonical.
     Same thresholds (50/10) — absolute counts of independent contribution.
  6. Auxiliary "high-stability" subset filtered at ≥3 hits AND ≥3 unique entities.
     Primary numbers stay at ≥2 hits for cross-pilot comparability; the
     ≥3-stability subset is what to use for any downstream analysis.

Output paths (all under larql/output/gemma3-4b-v2.vindex/):
  feature_labels_extended_pilot.json           — primary labels (≥2 hits filter)
  feature_labels_extended_pilot_rich.json      — entities, outputs, relations, layer
  feature_labels_extended_pilot_stable.json    — auxiliary ≥3-hit ≥3-entity subset
  feature_labels_extended_pilot_decision.json  — branch fired, per-relation, depth-strat
"""

import argparse
import json
import sys
import time
import numpy as np
from collections import defaultdict, Counter
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import probe_mlx as pm  # noqa: E402

_KNOWLEDGE_DIR = _SCRIPT_DIR.parent
_DEFAULT_EXTENDED = _KNOWLEDGE_DIR / "data" / "wordnet_extended_relations.json"

ADJECTIVE_SIDE = {"wn:pertainym", "wn:similar_to", "wn:attribute", "wn:also_see"}
VERB_SIDE = {"wn:entailment", "wn:cause"}


def load_extended(path: Path) -> dict:
    with open(path) as f:
        raw = json.load(f)
    return {f"wn:{rel}": data for rel, data in raw.items()}


def count_wn_labels(labels_dict: dict) -> set:
    return {k for k, v in labels_dict.items()
            if isinstance(v, str) and v.startswith("wn:")}


def parse_args():
    p = argparse.ArgumentParser(description="Pilot 1c probe: extended WordNet relations")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--vindex", default=None)
    p.add_argument("--extended", type=str, default=str(_DEFAULT_EXTENDED))
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-gate-score", type=float, default=5.0)
    p.add_argument("--scan-end-layer", type=int, default=21,
                   help="Scan L0..scan_end_layer-1. Default 21 (i.e. L0-L20).")
    p.add_argument("--offline", action="store_true", default=True)
    p.add_argument("--limit-subjects", type=int, default=None)
    p.add_argument("--output-suffix", type=str, default="",
                   help="Suffix appended to output filenames (e.g., '_l33_signed')")
    return p.parse_args()


def main():
    args = parse_args()
    model_id = args.model
    model_slug = pm._model_slug(model_id)

    vindex_path = args.vindex
    if vindex_path is None:
        output_root = _KNOWLEDGE_DIR.parent / "output"
        canonical = output_root / "gemma3-4b-v2.vindex"
        slug_default = output_root / f"{model_slug}.vindex"
        if canonical.exists():
            vindex_path = str(canonical)
        elif slug_default.exists():
            vindex_path = str(slug_default)
    if not vindex_path or not Path(vindex_path).exists():
        print(f"ERROR: vindex not found. Pass --vindex explicitly.", file=sys.stderr)
        sys.exit(1)

    print(f"Loading vindex: {vindex_path}")
    config, gates, down_meta = pm.load_vindex_gates_and_meta(vindex_path)
    num_layers = config["num_layers"]
    print(f"  {num_layers} layers, {config['hidden_size']} hidden, {len(down_meta)} features")

    scan_layers = list(range(0, min(args.scan_end_layer, num_layers)))
    print(f"  Scanning L0-L{scan_layers[-1]} ({len(scan_layers)} layers) — extended from canonical's L0-L12")

    extended_path = Path(args.extended)
    if not extended_path.exists():
        print(f"ERROR: extended data not found at {extended_path}", file=sys.stderr)
        print("Run fetch_wordnet_extended_relations.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading extended data: {extended_path}")
    syntax_data = load_extended(extended_path)
    total_pairs = sum(len(d.get("pairs", [])) for d in syntax_data.values())
    print(f"  {len(syntax_data)} relations, {total_pairs} pairs")
    for rel, data in syntax_data.items():
        n = len(data.get("pairs", []))
        side = "adj" if rel in ADJECTIVE_SIDE else ("verb" if rel in VERB_SIDE else "?")
        print(f"    {rel:<20s} [{side}] {n} pairs")

    syntax_index = pm.build_match_index(syntax_data)
    print(f"  Match index: {len(syntax_index)} entries")

    TEMPLATES = {rel: ["{X}"] for rel in syntax_data}

    print(f"Loading MLX model: {model_id}...")
    import os
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_id)
    print("  Model loaded")

    start_time = time.time()

    print("Phase 1: collecting unique subjects...")
    rel_to_subjects = {}
    unique_subjects_set = set()
    for rel_name in TEMPLATES:
        if rel_name not in syntax_data:
            continue
        rel_subjs = list({
            pair[0] for pair in syntax_data[rel_name].get("pairs", [])
            if len(pair) >= 2 and 2 <= len(pair[0]) <= 30
        })
        if args.limit_subjects:
            rel_subjs = rel_subjs[: args.limit_subjects]
        rel_to_subjects[rel_name] = rel_subjs
        unique_subjects_set.update(rel_subjs)

    unique_subjects = sorted(unique_subjects_set, key=lambda s: (len(s.split()), len(s)))
    pair_level_probes = sum(len(s) for s in rel_to_subjects.values())
    print(f"  {len(unique_subjects)} unique subjects across {len(rel_to_subjects)} relations"
          f" (pair-level: {pair_level_probes} - cache saves {pair_level_probes - len(unique_subjects)})")

    print("Phase 2: encoding...")
    residual_cache = {}
    encode_start = time.time()
    for i, subj in enumerate(unique_subjects):
        residuals, _ = pm.get_residuals_and_logits(model, tokenizer, subj)
        if residuals is None:
            continue
        residual_cache[subj] = {l: residuals[l] for l in scan_layers if l in residuals}
        if (i + 1) % 50 == 0 or (i + 1) == len(unique_subjects):
            el = time.time() - encode_start
            rate = (i + 1) / max(el, 0.1)
            eta = max(0, (len(unique_subjects) - i - 1) / max(rate, 0.1))
            sys.stdout.write(
                f"\r  encoded {i+1}/{len(unique_subjects)} ({rate:.1f}/s, ETA {eta:.0f}s)  "
            )
            sys.stdout.flush()
    print(f"\n  Encoded {len(residual_cache)} subjects in {time.time() - encode_start:.0f}s")

    print("Phase 3: gate matching (per-layer per-relation tracking)...")
    feature_hits = defaultdict(lambda: defaultdict(int))
    feature_entities = defaultdict(lambda: defaultdict(set))
    feature_outputs = defaultdict(lambda: defaultdict(set))
    feature_layer = {}  # first layer where feature was hit, for layer attribution
    # per-relation per-layer hit count for depth stratification
    rel_layer_hits = defaultdict(lambda: defaultdict(int))
    match_attempts = 0

    for rel_name, rel_subjs in rel_to_subjects.items():
        gate_matched = 0
        for subject in rel_subjs:
            if subject not in residual_cache:
                continue
            residuals = residual_cache[subject]
            match_attempts += 1
            subj_key = subject.lower().strip()

            for layer in scan_layers:
                if layer not in residuals or layer not in gates:
                    continue
                r = residuals[layer]
                scores = gates[layer] @ r
                top_indices = np.argsort(-scores)[:args.top_k]

                for feat_idx in top_indices:
                    score = float(scores[feat_idx])
                    if score < args.min_gate_score:
                        continue
                    tokens = down_meta.get((layer, int(feat_idx)), [])
                    if not tokens:
                        continue

                    feat_key = f"L{layer}_F{feat_idx}"
                    for target in tokens:
                        if len(target) < 2:
                            continue
                        tgt_lower = target.lower().strip()
                        if syntax_index.get((subj_key, tgt_lower)) == rel_name:
                            feature_hits[feat_key][rel_name] += 1
                            feature_entities[feat_key][rel_name].add(subject)
                            feature_outputs[feat_key][rel_name].add(tgt_lower)
                            rel_layer_hits[rel_name][layer] += 1
                            if feat_key not in feature_layer:
                                feature_layer[feat_key] = layer
                            gate_matched += 1
                            break

        print(f"  {rel_name:<25s} {len(rel_subjs):5d} subjects -> {gate_matched:5d} hits")

    elapsed = time.time() - start_time
    inference_count = len(residual_cache)
    print(f"\nTotal: {inference_count} forward passes, {match_attempts} match attempts in {elapsed:.0f}s")
    print(f"Features with hits: {len(feature_hits)}")

    # Primary labels (≥2 hits + conf > 0.5, parallel to prior pilots)
    pilot_labels = {}
    label_details = {}
    relation_totals = Counter()
    for feat_key, rel_counts in feature_hits.items():
        total_hits = sum(rel_counts.values())
        primary_rel = max(rel_counts, key=rel_counts.get)
        primary_count = rel_counts[primary_rel]
        confidence = primary_count / total_hits
        if primary_count >= 2 and confidence > 0.5:
            pilot_labels[feat_key] = primary_rel
            relation_totals[primary_rel] += 1
            entities = sorted(feature_entities[feat_key].get(primary_rel, set()))
            outputs = sorted(feature_outputs[feat_key].get(primary_rel, set()))
            label_details[feat_key] = {
                "primary": primary_rel,
                "confidence": round(confidence, 3),
                "hits": total_hits,
                "entity_count": len(entities),
                "entities": entities[:20],
                "outputs": outputs[:10],
                "relations": {r: c for r, c in sorted(rel_counts.items(), key=lambda x: -x[1])},
                "first_layer": feature_layer.get(feat_key),
            }
    print(f"Labeled (>=2 hits, conf>0.5): {len(pilot_labels)} features"
          f" ({len(feature_hits) - len(pilot_labels)} dropped)")

    # Auxiliary stability filter (≥3 hits + ≥3 unique entities) — applied to
    # the same label set, not a re-decision. Use for downstream quality work.
    stable_labels = {}
    for feat_key, label in pilot_labels.items():
        det = label_details[feat_key]
        primary = det["primary"]
        ent_count = len(feature_entities[feat_key].get(primary, set()))
        if det["hits"] >= 3 and ent_count >= 3:
            stable_labels[feat_key] = label
    print(f"Stable subset (>=3 hits, >=3 entities): {len(stable_labels)} features")

    if relation_totals:
        print(f"\nRelation distribution ({len(relation_totals)} relations):")
        for rel, count in relation_totals.most_common():
            side = "adj" if rel in ADJECTIVE_SIDE else ("verb" if rel in VERB_SIDE else "?")
            stable_count = sum(1 for k, v in stable_labels.items() if v == rel)
            print(f"  {rel:<25s} [{side}] {count:4d}  ({stable_count} stable)")

    suffix = getattr(args, 'output_suffix', '')
    pilot_path = Path(vindex_path) / f"feature_labels_extended_pilot{suffix}.json"
    with open(pilot_path, "w") as f:
        json.dump(pilot_labels, f, indent=2, ensure_ascii=False)
    print(f"\nPilot labels -> {pilot_path}")

    details_path = Path(vindex_path) / f"feature_labels_extended_pilot{suffix}_rich.json"
    with open(details_path, "w") as f:
        json.dump(label_details, f, indent=2, ensure_ascii=False)
    print(f"Pilot details -> {details_path}")

    stable_path = Path(vindex_path) / f"feature_labels_extended_pilot{suffix}_stable.json"
    with open(stable_path, "w") as f:
        json.dump(stable_labels, f, indent=2, ensure_ascii=False)
    print(f"Pilot stable subset -> {stable_path}")

    # Decision rule: new vs cumulative (canon ∪ multilingual ∪ subword)
    canonical_path = Path(vindex_path) / "feature_labels.json"
    ml_path = Path(vindex_path) / "feature_labels_multilingual_pilot.json"
    sw_path = Path(vindex_path) / "feature_labels_subword_pilot.json"

    canonical = json.load(open(canonical_path)) if canonical_path.exists() else {}
    ml = json.load(open(ml_path)) if ml_path.exists() else {}
    sw = json.load(open(sw_path)) if sw_path.exists() else {}

    canonical_wn = count_wn_labels(canonical)
    ml_wn = count_wn_labels(ml)
    sw_wn = count_wn_labels(sw)
    cumulative_wn = canonical_wn | ml_wn | sw_wn

    pilot_wn = count_wn_labels(pilot_labels)
    new_vs_canonical = pilot_wn - canonical_wn
    new_vs_cumulative = pilot_wn - cumulative_wn

    new_by_relation = Counter(pilot_labels[k] for k in new_vs_cumulative if k in pilot_labels)

    # Content-lineage tracking for any features that overlap multilingual or subword
    ml_rich = json.load(open(Path(vindex_path) / "feature_labels_multilingual_pilot_rich.json")) if (Path(vindex_path) / "feature_labels_multilingual_pilot_rich.json").exists() else {}
    sw_rich = json.load(open(Path(vindex_path) / "feature_labels_subword_pilot_rich.json")) if (Path(vindex_path) / "feature_labels_subword_pilot_rich.json").exists() else {}

    content_lineage = {}
    for feat_key in pilot_wn & (set(ml_rich.keys()) | set(sw_rich.keys())):
        det = label_details.get(feat_key, {})
        ext_ents = set(e.lower() for e in det.get("entities", []))
        ext_outs = set(o.lower() for o in det.get("outputs", []))
        lineage = {"extended_primary": det.get("primary"), "extended_entities": list(ext_ents)[:10], "extended_outputs": list(ext_outs)[:5]}
        if feat_key in ml_rich:
            m_ents = set(e.lower() for e in ml_rich[feat_key].get("entities", []))
            m_outs = set(o.lower() for o in ml_rich[feat_key].get("outputs", []))
            lineage["multilingual"] = {
                "primary": ml_rich[feat_key]["primary"],
                "entity_overlap_with_extended": list(ext_ents & m_ents),
                "output_overlap_with_extended": list(ext_outs & m_outs),
            }
        if feat_key in sw_rich:
            s_ents = set(e.lower() for e in sw_rich[feat_key].get("entities", []))
            s_outs = set(o.lower() for o in sw_rich[feat_key].get("outputs", []))
            lineage["subword"] = {
                "primary": sw_rich[feat_key]["primary"],
                "entity_overlap_with_extended": list(ext_ents & s_ents),
                "output_overlap_with_extended": list(ext_outs & s_outs),
            }
        content_lineage[feat_key] = lineage

    # Depth stratification analysis
    depth_strat = {}
    for rel in syntax_data:
        layer_hits = rel_layer_hits.get(rel, {})
        early = sum(c for l, c in layer_hits.items() if 0 <= l <= 12)
        late = sum(c for l, c in layer_hits.items() if 13 <= l <= 20)
        depth_strat[rel] = {
            "L0_L12_hits": early,
            "L13_L20_hits": late,
            "early_vs_late_ratio": round(early / max(late, 1), 3),
            "side": "adj" if rel in ADJECTIVE_SIDE else ("verb" if rel in VERB_SIDE else "?"),
            "per_layer": dict(sorted(layer_hits.items())),
        }

    print("\n" + "=" * 60)
    print("DECISION RULE EVALUATION")
    print("=" * 60)
    print(f"Canonical wn:* labels:                64")
    print(f"Multilingual pilot wn:* labels:       30")
    print(f"Subword pilot wn:* labels:            52")
    print(f"Extended (1c) pilot wn:* labels:      {len(pilot_wn)}")
    print(f"  Stable subset:                      {len(stable_labels)}")
    print(f"New vs canonical:                     {len(new_vs_canonical)}")
    print(f"New vs cumulative (canon u ml u sw):  {len(new_vs_cumulative)}")
    print(f"\nNew vs cumulative, by relation:")
    for rel in syntax_data:
        count = new_by_relation.get(rel, 0)
        side = "adj" if rel in ADJECTIVE_SIDE else ("verb" if rel in VERB_SIDE else "?")
        print(f"  {rel:<25s} [{side}] {count:4d}")

    print(f"\nDepth stratification (verb-side hypothesis test):")
    for rel in sorted(syntax_data, key=lambda r: (r not in VERB_SIDE, r)):
        ds = depth_strat[rel]
        print(f"  {rel:<25s} [{ds['side']}] L0-L12={ds['L0_L12_hits']:5d}  L13-L20={ds['L13_L20_hits']:5d}  ratio={ds['early_vs_late_ratio']}")

    n_new_cum = len(new_vs_cumulative)
    if n_new_cum >= 50:
        branch = "A: relation coverage was a major axis; per-axis ceiling higher than model predicted"
        next_action = "Reassess working model. Three axes contributed; possible 4th axis worth probing."
    elif n_new_cum >= 10:
        branch = "B: relation coverage contributes per working model"
        next_action = "Three axes saturate. Cumulative inventory now {} wn:*. Closing the lexical labeling program for L0-L12 is reasonable. Pivot to 2c semantic coverage or structural-feature substrate question.".format(len(cumulative_wn) + n_new_cum)
    else:
        branch = "C: relation coverage was NOT a binding axis"
        next_action = "Model over-predicted; the methodology is closer to saturation than thought. Strong argument for closing lexical program."

    # Depth stratification decision
    verb_early = sum(depth_strat[r]["L0_L12_hits"] for r in VERB_SIDE if r in syntax_data)
    verb_late = sum(depth_strat[r]["L13_L20_hits"] for r in VERB_SIDE if r in syntax_data)
    adj_early = sum(depth_strat[r]["L0_L12_hits"] for r in ADJECTIVE_SIDE if r in syntax_data)
    if verb_early + verb_late < 20:
        depth_decision = "UNTESTABLE: verb-side relations produced <20 total hits. Methodology may not detect them; need different probe."
    elif verb_late > 2 * max(verb_early, 1):
        depth_decision = "SUPPORTED: verb-side hits cluster at L13-L20 (ratio >2:1 vs L0-L12). Relations are depth-stratified by semantic load."
    elif verb_early > 2 * max(verb_late, 1):
        depth_decision = "REFUTED-INVERSE: verb-side hits cluster at L0-L12, not L13+. Hypothesis was backwards."
    else:
        depth_decision = "REFUTED: verb-side hits spread roughly equally across L0-L20. Relations are NOT depth-stratified by semantic load."

    print(f"\nBranch fired:       {branch}")
    print(f"Next action:        {next_action}")
    print(f"Depth stratification: {depth_decision}")

    decision_path = Path(vindex_path) / f"feature_labels_extended_pilot{suffix}_decision.json"
    with open(decision_path, "w") as f:
        json.dump({
            "pilot_status": "PILOT_RESULT - NOT MERGED INTO CANONICAL",
            "pilot_name": "1c_relation_coverage",
            "canonical_wn_count": len(canonical_wn),
            "multilingual_wn_count": len(ml_wn),
            "subword_wn_count": len(sw_wn),
            "extended_wn_count": len(pilot_wn),
            "extended_stable_count": len(stable_labels),
            "new_vs_canonical": len(new_vs_canonical),
            "new_vs_cumulative": len(new_vs_cumulative),
            "new_vs_cumulative_by_relation": dict(new_by_relation),
            "depth_stratification": depth_strat,
            "content_lineage": content_lineage,
            "branch": branch,
            "next_action": next_action,
            "depth_decision": depth_decision,
            "inference_count": inference_count,
            "pair_level_probes": pair_level_probes,
            "match_attempts": match_attempts,
            "elapsed_seconds": int(elapsed),
            "scan_layers": [scan_layers[0], scan_layers[-1]],
            "pre_registered_prediction": "21-53 new vs cumulative, centered ~30-40 if meta-pattern holds",
        }, f, indent=2, ensure_ascii=False)
    print(f"Decision record -> {decision_path}")


if __name__ == "__main__":
    main()
