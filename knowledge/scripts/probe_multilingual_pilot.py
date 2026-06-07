#!/usr/bin/env python3
"""Pilot probe: re-run the L0-L13 gate-matching probe against multilingual
WordNet pairs (produced by fetch_wordnet_multilingual_pilot.py) and write
labels to a separate output file. Does NOT merge into canonical feature_labels.json.

Decision rule (pre-registered):
  new_wn_labels = (wn:* feature keys in pilot output) − (wn:* keys in canonical)
  ≥ 50  → multilingual was the binding gap; scale that direction
  10-50 → multilingual matters but not dominant; pilot subword next
  < 10  → multilingual wasn't binding; pilot subword fragmentation / relation coverage

Architecture: imports loaders, vindex helpers, and inference from probe_mlx.py
and re-implements the probe loop with three changes:
  1. Loads only multilingual data (data/wordnet_multilingual_pilot.json)
  2. Scans only syntax layers (L0 .. syntax_end-1)
  3. Writes to feature_labels_multilingual_pilot.json (separate from canonical)
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
_DEFAULT_MULTILINGUAL = _KNOWLEDGE_DIR / "data" / "wordnet_multilingual_pilot.json"


def load_multilingual(path: Path) -> dict:
    """Load multilingual relations and wrap with wn: prefix (matches probe convention)."""
    with open(path) as f:
        raw = json.load(f)
    # Same prefix convention as load_syntax_data() in probe_mlx.py
    return {f"wn:{rel}": data for rel, data in raw.items()}


def count_wn_labels(labels_dict: dict) -> set:
    """Return set of feature keys whose label is wn:*."""
    return {k for k, v in labels_dict.items()
            if isinstance(v, str) and v.startswith("wn:")}


def parse_args():
    p = argparse.ArgumentParser(description="Pilot probe: multilingual WordNet only")
    p.add_argument("--model", default="google/gemma-3-4b-it")
    p.add_argument("--vindex", default=None,
                   help="Path to vindex (default: <repo>/output/<model-slug>.vindex)")
    p.add_argument("--multilingual", type=str, default=str(_DEFAULT_MULTILINGUAL),
                   help="Path to multilingual pairs JSON")
    p.add_argument("--top-k", type=int, default=50)
    p.add_argument("--min-gate-score", type=float, default=5.0)
    p.add_argument("--offline", action="store_true", default=True)
    p.add_argument("--limit-subjects", type=int, default=None,
                   help="Cap subjects per relation for fast smoke-test")
    p.add_argument("--scan-end-layer", type=int, default=None,
                   help="Override scan range to L0..scan_end_layer-1.")
    p.add_argument("--output-suffix", type=str, default="",
                   help="Suffix appended to output filenames (e.g., '_l20')")
    return p.parse_args()


def main():
    args = parse_args()
    model_id = args.model
    model_slug = pm._model_slug(model_id)

    # ── Locate vindex ──
    # Canonical for gemma-3-4b-it is `gemma3-4b-v2.vindex` (not <model-slug>.vindex).
    # Falls back to probe_mlx.py's default if the canonical isn't present.
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

    # ── Load multilingual data ──
    multilingual_path = Path(args.multilingual)
    if not multilingual_path.exists():
        print(f"ERROR: multilingual data not found at {multilingual_path}", file=sys.stderr)
        print(f"Run fetch_wordnet_multilingual_pilot.py first.", file=sys.stderr)
        sys.exit(1)
    print(f"Loading multilingual data: {multilingual_path}")
    syntax_data = load_multilingual(multilingual_path)
    total_pairs = sum(len(d.get("pairs", [])) for d in syntax_data.values())
    print(f"  {len(syntax_data)} relations, {total_pairs} pairs")

    # ── Build syntax match index ──
    syntax_index = pm.build_match_index(syntax_data)
    print(f"  Match index: {len(syntax_index)} entries")

    # ── Probe templates: identity (matches probe_mlx.py syntax mode) ──
    TEMPLATES = {rel: ["{X}"] for rel in syntax_data}

    # ── Load model ──
    print(f"Loading MLX model: {model_id}...")
    import os
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(model_id)
    print("  Model loaded")

    # ── Three-phase loop: collect → encode-once → gate-match ──
    # Identity templates ({X}) mean the prompt IS the subject. Many subjects appear
    # in multiple relations (synonym/hypernym/antonym/meronym/derivation), so naively
    # iterating (rel × subject) re-encodes the same prompt up to 5x. Caching by
    # subject brings the forward-pass count down to len(unique subjects).
    start_time = time.time()

    # Phase 1: collect unique subjects + per-relation subject lists
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
          f" (pair-level: {pair_level_probes} — cache saves {pair_level_probes - len(unique_subjects)}"
          f" redundant forward passes)")

    # Phase 2: encode each unique subject once, cache only syntax-layer residuals
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

    # Phase 3: gate-matching against cached residuals
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

    # ── Label confidence filter (matches probe_mlx.py:715-723) ──
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

    # ── Save (separate file — does NOT merge into canonical) ──
    suffix = getattr(args, 'output_suffix', '')
    pilot_path = Path(vindex_path) / f"feature_labels_multilingual_pilot{suffix}.json"
    with open(pilot_path, "w") as f:
        json.dump(pilot_labels, f, indent=2, ensure_ascii=False)
    print(f"\nPilot labels -> {pilot_path}")

    details_path = Path(vindex_path) / f"feature_labels_multilingual_pilot{suffix}_rich.json"
    with open(details_path, "w") as f:
        json.dump(label_details, f, indent=2, ensure_ascii=False)
    print(f"Pilot details -> {details_path}")

    # ── Decision rule ──
    canonical_path = Path(vindex_path) / "feature_labels.json"
    if not canonical_path.exists():
        print(f"\nWARNING: canonical feature_labels.json not found at {canonical_path}")
        print("Cannot evaluate decision rule.")
        return

    with open(canonical_path) as f:
        canonical = json.load(f)
    canonical_wn = count_wn_labels(canonical)
    pilot_wn = count_wn_labels(pilot_labels)
    new_wn = pilot_wn - canonical_wn

    # Break down NEW labels by their wn:* sub-relation. Antonym pairs carry
    # the most translation noise (~17% of source pairs are inherited via
    # English synset antonymy and may not preserve antonymy in the target
    # language); a new-label count dominated by wn:antonym should be read with
    # caution even if the branch fired is positive.
    from collections import Counter as _Counter
    new_by_relation = _Counter(
        pilot_labels[k] for k in new_wn if k in pilot_labels
    )

    print("\n" + "=" * 60)
    print("DECISION RULE EVALUATION")
    print("=" * 60)
    print(f"Canonical wn:* labels:          {len(canonical_wn)}")
    print(f"Pilot wn:* labels:              {len(pilot_wn)}")
    print(f"New wn:* labels (pilot−canon):  {len(new_wn)}")
    if new_by_relation:
        print("New labels by relation:")
        for rel, count in new_by_relation.most_common():
            share = count / max(len(new_wn), 1)
            flag = "  ← noise-prone" if rel == "wn:antonym" else ""
            print(f"  {rel:<20s} {count:4d}  ({share:>5.1%}){flag}")
    print()
    if len(new_wn) >= 50:
        branch = "A: multilingual was the binding gap"
        next_action = "Scale the multilingual direction — more languages, more pairs."
    elif len(new_wn) >= 10:
        branch = "B: multilingual matters but not dominant"
        next_action = "Pilot subword fragmentation next; multilingual alone insufficient."
    else:
        branch = "C: multilingual wasn't binding"
        next_action = "Pilot subword fragmentation or relation coverage; investigate filter."
    print(f"Branch fired: {branch}")
    print(f"Next action:  {next_action}")
    if new_by_relation.get("wn:antonym", 0) / max(len(new_wn), 1) > 0.4:
        print("  CAVEAT: ≥40% of new labels are wn:antonym — translation-noise dominated.")
        print("  Read the branch with caution; antonym source pairs carry ~17% noise.")

    # Persist decision record for the writeup
    decision_path = Path(vindex_path) / f"feature_labels_multilingual_pilot{suffix}_decision.json"
    with open(decision_path, "w") as f:
        json.dump({
            "pilot_status": "PILOT_RESULT — NOT MERGED INTO CANONICAL",
            "canonical_wn_count": len(canonical_wn),
            "pilot_wn_count": len(pilot_wn),
            "new_wn_count": len(new_wn),
            "new_wn_by_relation": dict(new_by_relation),
            "new_wn_keys": sorted(new_wn),
            "branch": branch,
            "next_action": next_action,
            "inference_count": inference_count,
            "pair_level_probes": pair_level_probes,
            "match_attempts": match_attempts,
            "elapsed_seconds": int(elapsed),
            "languages": ["fra", "ita", "por", "spa", "nld"],
        }, f, indent=2, ensure_ascii=False)
    print(f"Decision record -> {decision_path}")


if __name__ == "__main__":
    main()
