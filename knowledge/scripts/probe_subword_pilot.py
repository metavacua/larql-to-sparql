#!/usr/bin/env python3
"""Pilot probe: re-run the L0-L13 gate-matching probe against long-tail
multi-piece WordNet pairs (produced by fetch_wordnet_subword_pilot.py) and write
labels to a separate output file. Does NOT merge into canonical feature_labels.json.

Decision rule (pre-registered, parallel to multilingual pilot):
  new_wn_labels = (wn:* feature keys in pilot output) - (wn:* keys in canonical)
  >= 50  -> subword fragmentation (long-tail multi-piece) was the binding gap
  10-50  -> contributes but not dominant; pilot relation coverage next
  < 10   -> not binding; relation coverage or methodology audit (2a)

Structure mirrors probe_multilingual_pilot.py exactly.
"""

import argparse
import json
import sys
import time
import numpy as np
from collections import defaultdict
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

import probe_mlx as pm  # noqa: E402

_KNOWLEDGE_DIR = _SCRIPT_DIR.parent
_DEFAULT_SUBWORD = _KNOWLEDGE_DIR / "data" / "wordnet_subword_pilot.json"


def load_subword(path: Path) -> dict:
    """Load long-tail multi-piece relations and wrap with wn: prefix."""
    with open(path) as f:
        raw = json.load(f)
    return {f"wn:{rel}": data for rel, data in raw.items()}


def count_wn_labels(labels_dict: dict) -> set:
    return {k for k, v in labels_dict.items()
            if isinstance(v, str) and v.startswith("wn:")}


def parse_args():
    p = argparse.ArgumentParser(description="Pilot probe: long-tail multi-piece WordNet")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--vindex", default=None)
    p.add_argument("--subword", type=str, default=str(_DEFAULT_SUBWORD))
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-gate-score", type=float, default=5.0)
    p.add_argument("--offline", action="store_true", default=True)
    p.add_argument("--limit-subjects", type=int, default=None)
    p.add_argument("--scan-end-layer", type=int, default=None,
                   help="Override scan range to L0..scan_end_layer-1.")
    p.add_argument("--output-suffix", type=str, default="",
                   help="Suffix appended to output filenames (e.g., '_l20')")
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

    if args.scan_end_layer is not None:
        syntax_end = min(args.scan_end_layer, num_layers)
    elif "layer_bands" in config and config["layer_bands"]:
        bands = config["layer_bands"]
        syntax_end = bands.get("knowledge_start", num_layers * 2 // 5)
    else:
        syntax_end = num_layers * 2 // 5
    scan_layers = list(range(0, syntax_end))
    print(f"  Scanning syntax layers: L0-L{syntax_end - 1} ({len(scan_layers)} layers)")

    subword_path = Path(args.subword)
    if not subword_path.exists():
        print(f"ERROR: subword data not found at {subword_path}", file=sys.stderr)
        print("Run fetch_wordnet_subword_pilot.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading subword data: {subword_path}")
    syntax_data = load_subword(subword_path)
    total_pairs = sum(len(d.get("pairs", [])) for d in syntax_data.values())
    print(f"  {len(syntax_data)} relations, {total_pairs} pairs")

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
          f" (pair-level: {pair_level_probes} - cache saves {pair_level_probes - len(unique_subjects)}"
          f" redundant forward passes)")

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

    print("Phase 3: gate matching...")
    feature_hits = defaultdict(lambda: defaultdict(int))
    feature_entities = defaultdict(lambda: defaultdict(set))
    feature_outputs = defaultdict(lambda: defaultdict(set))
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
                            gate_matched += 1
                            break

        print(f"  {rel_name:<25s} {len(rel_subjs):5d} subjects -> {gate_matched:5d} hits")

    elapsed = time.time() - start_time
    inference_count = len(residual_cache)
    print(f"\nTotal: {inference_count} forward passes (vs {pair_level_probes} naive),"
          f" {match_attempts} match attempts in {elapsed:.0f}s")
    print(f"Features with hits: {len(feature_hits)}")

    pilot_labels = {}
    label_details = {}
    relation_totals = defaultdict(int)
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
            }
    print(f"Labeled: {len(pilot_labels)} features"
          f" ({len(feature_hits) - len(pilot_labels)} dropped by confidence filter)")

    if relation_totals:
        print(f"\nRelation distribution ({len(relation_totals)} relations):")
        for rel, count in sorted(relation_totals.items(), key=lambda x: -x[1]):
            print(f"  {rel:<25s} {count:4d}")

    suffix = getattr(args, 'output_suffix', '')
    pilot_path = Path(vindex_path) / f"feature_labels_subword_pilot{suffix}.json"
    with open(pilot_path, "w") as f:
        json.dump(pilot_labels, f, indent=2, ensure_ascii=False)
    print(f"\nPilot labels -> {pilot_path}")

    details_path = Path(vindex_path) / f"feature_labels_subword_pilot{suffix}_rich.json"
    with open(details_path, "w") as f:
        json.dump(label_details, f, indent=2, ensure_ascii=False)
    print(f"Pilot details -> {details_path}")

    # Decision rule
    canonical_path = Path(vindex_path) / "feature_labels.json"
    multilingual_path = Path(vindex_path) / "feature_labels_multilingual_pilot.json"
    if not canonical_path.exists():
        print(f"\nWARNING: canonical feature_labels.json not found at {canonical_path}")
        print("Cannot evaluate decision rule.")
        return

    with open(canonical_path) as f:
        canonical = json.load(f)
    canonical_wn = count_wn_labels(canonical)
    pilot_wn = count_wn_labels(pilot_labels)
    new_vs_canonical = pilot_wn - canonical_wn

    # Also compute vs canonical ∪ multilingual (the cumulative "what we already have")
    multilingual_wn = set()
    if multilingual_path.exists():
        with open(multilingual_path) as f:
            ml = json.load(f)
        multilingual_wn = count_wn_labels(ml)
    cumulative_wn = canonical_wn | multilingual_wn
    new_vs_cumulative = pilot_wn - cumulative_wn

    from collections import Counter as _Counter
    new_by_relation = _Counter(
        pilot_labels[k] for k in new_vs_canonical if k in pilot_labels
    )
    new_cum_by_relation = _Counter(
        pilot_labels[k] for k in new_vs_cumulative if k in pilot_labels
    )

    print("\n" + "=" * 60)
    print("DECISION RULE EVALUATION")
    print("=" * 60)
    print(f"Canonical wn:* labels:                 {len(canonical_wn)}")
    print(f"Multilingual pilot wn:* labels:        {len(multilingual_wn)}")
    print(f"Subword pilot wn:* labels:             {len(pilot_wn)}")
    print(f"New vs canonical (pilot - canon):      {len(new_vs_canonical)}")
    print(f"New vs cumulative (pilot - canon u ml): {len(new_vs_cumulative)}")
    if new_by_relation:
        print("\nNew vs canonical, by relation:")
        for rel, count in new_by_relation.most_common():
            share = count / max(len(new_vs_canonical), 1)
            print(f"  {rel:<20s} {count:4d}  ({share:>5.1%})")
    if new_cum_by_relation:
        print("\nNew vs cumulative, by relation:")
        for rel, count in new_cum_by_relation.most_common():
            share = count / max(len(new_vs_cumulative), 1)
            print(f"  {rel:<20s} {count:4d}  ({share:>5.1%})")

    # Decision rule uses vs-canonical (parallel to multilingual pilot's rule)
    n_new = len(new_vs_canonical)
    print()
    if n_new >= 50:
        branch = "A: subword fragmentation (long-tail multi-piece) was the binding gap"
        next_action = "Scale long-tail expansion; investigate the specific lexicon strata that produced new labels."
    elif n_new >= 10:
        branch = "B: subword contributes but not dominant"
        next_action = "Pilot relation coverage (1c) next; long-tail alone insufficient."
    else:
        branch = "C: subword (long-tail multi-piece) was not binding"
        next_action = "Likely cumulative pattern points at relation coverage; revisit methodology audit (2a) findings before pre-registering 1c."
    print(f"Branch fired: {branch}")
    print(f"Next action:  {next_action}")

    decision_path = Path(vindex_path) / f"feature_labels_subword_pilot{suffix}_decision.json"
    with open(decision_path, "w") as f:
        json.dump({
            "pilot_status": "PILOT_RESULT - NOT MERGED INTO CANONICAL",
            "pilot_name": "1b_subword_fragmentation",
            "canonical_wn_count": len(canonical_wn),
            "multilingual_wn_count": len(multilingual_wn),
            "pilot_wn_count": len(pilot_wn),
            "new_vs_canonical": len(new_vs_canonical),
            "new_vs_cumulative": len(new_vs_cumulative),
            "new_vs_canonical_by_relation": dict(new_by_relation),
            "new_vs_cumulative_by_relation": dict(new_cum_by_relation),
            "new_wn_keys_vs_canonical": sorted(new_vs_canonical),
            "new_wn_keys_vs_cumulative": sorted(new_vs_cumulative),
            "branch": branch,
            "next_action": next_action,
            "inference_count": inference_count,
            "pair_level_probes": pair_level_probes,
            "match_attempts": match_attempts,
            "elapsed_seconds": int(elapsed),
            "design_note": "long-tail WordNet, canonical-skip, multi-piece filter (>=2 BPE pieces)",
        }, f, indent=2, ensure_ascii=False)
    print(f"Decision record -> {decision_path}")


if __name__ == "__main__":
    main()
