#!/usr/bin/env python3
"""Probe model features with MLX — full forward pass with real attention.

Two matching strategies:
  1. Gate matching: project residuals through gate vectors, check if the
     down_meta top-K tokens match a triple object for this (subject, relation).
  2. Prediction matching: project the final residual through the LM head,
     check if the model's top predicted tokens match a triple object.

Supports multiple data sources:
  - Wikidata triples (knowledge layers L14-27)
  - WordNet relations (syntax layers L0-13)
  - Morphological pairs (syntax layers L0-7)
  - AST pairs (syntax layers L0-13)

Model-agnostic: works with any MLX-compatible model.
Decoupled from vindex: prediction matching works without one.
Resumable: saves progress per (relation, entity) pair.

Usage:
    # Full probe (knowledge + syntax)
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b-f16.vindex

    # Knowledge only (default)
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b-f16.vindex --layers knowledge

    # Syntax only
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b-f16.vindex --layers syntax

    # Specific relations
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b-f16.vindex --relations capital,language,continent

    # Resume interrupted probe
    python3 scripts/probe_mlx.py --model google/gemma-3-4b-it --vindex output/gemma3-4b-f16.vindex --resume
"""

import argparse
import json
import sys
import struct
import time
import numpy as np
from pathlib import Path
from collections import defaultdict

try:
    import mlx.core as mx
    import mlx.nn as nn
    from mlx_lm import load as mlx_load
except ImportError as e:
    print(f"Install MLX: pip install mlx mlx-lm ({e})", file=sys.stderr)
    sys.exit(1)


# Resolve paths relative to the monorepo root
_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent.parent
_KNOWLEDGE_DIR = _SCRIPT_DIR.parent

_DEFAULT_TRIPLES = str(_KNOWLEDGE_DIR / "data" / "wikidata_triples.json")
_DEFAULT_PROBES = str(_KNOWLEDGE_DIR / "probes")

_STOP_WORDS = frozenset({
    "the", "of", "and", "in", "to", "for", "is", "on", "at", "by",
    "an", "or", "de", "la", "le", "el", "du", "des", "von", "van",
    "al", "bin", "ibn", "di", "da", "do", "das", "den", "der", "het",
})


# ── Data loading ──────────────────────────────────────────────────────

def normalize_object(obj: str) -> list[str]:
    """Expand a triple object into all matchable forms."""
    obj_lower = obj.lower().strip()
    forms = {obj_lower}
    words = obj_lower.split()
    if len(words) > 1:
        for word in words:
            if word.endswith("'s"):
                clean = word[:-2]
            elif word.endswith("\u2019s"):
                clean = word[:-2]
            else:
                clean = word
            if clean not in _STOP_WORDS and len(clean) >= 4:
                forms.add(clean)
    return list(forms)


def build_match_index(triples: dict) -> dict[tuple[str, str], str]:
    """Build (subject, object_form) -> relation lookup with normalized forms."""
    index = {}
    for rel_name, rel_data in triples.items():
        for pair in rel_data.get("pairs", []):
            if len(pair) < 2:
                continue
            subj = pair[0].lower().strip()
            for form in normalize_object(pair[1]):
                key = (subj, form)
                if key not in index:
                    index[key] = rel_name
    return index


def load_templates(templates_path: str | None = None) -> dict[str, list[str]]:
    """Load probe templates. Returns {relation: [template, ...]}."""
    if templates_path is None:
        templates_path = str(_KNOWLEDGE_DIR / "data" / "probe_templates.json")
    path = Path(templates_path)
    if not path.exists():
        print(f"ERROR: Templates not found at {path}")
        sys.exit(1)
    with open(path) as f:
        raw = json.load(f)
    return {rel: (variants if isinstance(variants, list) else [variants])
            for rel, variants in raw.items()}


def load_syntax_data() -> dict:
    """Load WordNet, morphological, and AST data as a unified triples dict."""
    syntax_triples = {}
    data_dir = _KNOWLEDGE_DIR / "data"

    # WordNet
    wordnet_path = data_dir / "wordnet_relations.json"
    if wordnet_path.exists():
        with open(wordnet_path) as f:
            wn = json.load(f)
        for rel_name, rel_data in wn.items():
            syntax_triples[f"wn:{rel_name}"] = rel_data

    # Morphological
    morph_path = data_dir / "morphological_relations.json"
    if morph_path.exists():
        with open(morph_path) as f:
            m = json.load(f)
        for rel_name, rel_data in m.get("relations", {}).items():
            syntax_triples[f"morph:{rel_name}"] = rel_data

    # AST
    ast_dir = data_dir / "ast"
    if ast_dir.exists():
        for ast_file in ast_dir.glob("*.json"):
            with open(ast_file) as f:
                d = json.load(f)
            lang = d.get("language", ast_file.stem.replace("_ast", ""))
            for rel_name, rel_data in d.get("relations", {}).items():
                key = rel_name if ":" in rel_name else f"{lang}:{rel_name}"
                syntax_triples[key] = rel_data

    return syntax_triples


# ── Vindex loading ────────────────────────────────────────────────────

def _load_down_meta_jsonl(vindex_dir: Path) -> dict[tuple[int, int], list[str]]:
    """Load down_meta from JSONL format (legacy)."""
    down_meta = {}
    with open(vindex_dir / "down_meta.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            key = (obj.get("l", 0), obj.get("f", 0))
            tokens = []
            top_token = obj.get("t", "")
            if top_token:
                tokens.append(top_token)
            for entry in obj.get("k", []):
                tok = entry.get("t", "")
                if tok and tok not in tokens:
                    tokens.append(tok)
            down_meta[key] = tokens
    return down_meta


def _load_down_meta_bin(vindex_dir: Path, config: dict) -> dict[tuple[int, int], list[str]]:
    """Load down_meta from binary format. Token IDs resolved via tokenizer."""
    tokenizer_path = vindex_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        print(f"  WARNING: No tokenizer.json, cannot read down_meta.bin")
        return {}
    try:
        from tokenizers import Tokenizer
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    except ImportError:
        print("  WARNING: 'tokenizers' not installed, cannot read down_meta.bin")
        return {}

    data = (vindex_dir / "down_meta.bin").read_bytes()
    pos = 0
    magic, version, num_layers, top_k_count = struct.unpack_from("<IIII", data, pos)
    pos += 16

    if magic != 0x444D4554:
        print(f"  WARNING: Invalid down_meta.bin magic: 0x{magic:08X}")
        return {}

    record_size = 8 + top_k_count * 8

    def decode(tid):
        if tid == 0:
            return ""
        try:
            return tokenizer.decode([tid], skip_special_tokens=True).strip()
        except Exception:
            return ""

    down_meta = {}
    for layer_idx in range(num_layers):
        if pos + 4 > len(data):
            break
        nf = struct.unpack_from("<I", data, pos)[0]
        pos += 4
        for feat_idx in range(nf):
            if pos + record_size > len(data):
                break
            top_tid = struct.unpack_from("<I", data, pos)[0]
            tokens = []
            top_str = decode(top_tid)
            if top_str:
                tokens.append(top_str)
            for k in range(top_k_count):
                offset = pos + 8 + k * 8
                tid = struct.unpack_from("<I", data, offset)[0]
                if tid > 0:
                    s = decode(tid)
                    if s and s not in tokens:
                        tokens.append(s)
            pos += record_size
            if tokens:
                down_meta[(layer_idx, feat_idx)] = tokens

    return down_meta


def load_vindex_gates_and_meta(vindex_dir):
    """Load gate vectors and down_meta. Supports f16/f32 gates, bin/jsonl meta."""
    vindex_dir = Path(vindex_dir)
    with open(vindex_dir / "index.json") as f:
        config = json.load(f)

    hidden_size = config["hidden_size"]
    gate_path = vindex_dir / "gate_vectors.bin"
    gate_file_size = gate_path.stat().st_size
    total_elements = sum(li["num_features"] for li in config["layers"]) * hidden_size

    if gate_file_size == total_elements * 2:
        gate_dtype, bpe = np.float16, 2
    else:
        gate_dtype, bpe = np.float32, 4

    gate_raw = np.fromfile(gate_path, dtype=gate_dtype)
    gates = {}
    for layer_info in config["layers"]:
        layer = layer_info["layer"]
        nf = layer_info["num_features"]
        offset = layer_info["offset"] // bpe
        chunk = gate_raw[offset:offset + nf * hidden_size].reshape(nf, hidden_size)
        gates[layer] = chunk.astype(np.float32) if gate_dtype != np.float32 else chunk

    if (vindex_dir / "down_meta.bin").exists():
        print("  Reading down_meta.bin (binary format)...")
        down_meta = _load_down_meta_bin(vindex_dir, config)
    elif (vindex_dir / "down_meta.jsonl").exists():
        print("  Reading down_meta.jsonl (legacy format)...")
        down_meta = _load_down_meta_jsonl(vindex_dir)
    else:
        print("  WARNING: No down_meta found — gate matching disabled")
        down_meta = {}

    return config, gates, down_meta


# ── Model detection ───────────────────────────────────────────────────

def _find_model_parts(model):
    """Auto-detect model internals for any MLX architecture."""
    try:
        lm = model['language_model']
        inner = lm['model']
        embed_fn = inner.embed_tokens
        def lm_head(h):
            return h @ embed_fn.weight.T
        return embed_fn, inner.layers, inner.norm, lm_head, True
    except (KeyError, TypeError, AttributeError):
        pass

    inner = getattr(model, 'model', None)
    if inner and hasattr(inner, 'embed_tokens') and hasattr(inner, 'layers'):
        embed_fn = inner.embed_tokens
        if hasattr(model, 'lm_head'):
            lm_head_layer = model.lm_head
            def lm_head(h):
                return lm_head_layer(h)
        else:
            def lm_head(h):
                return h @ embed_fn.weight.T
        model_type = getattr(getattr(model, 'config', None), 'model_type', '')
        needs_scale = 'gemma' in str(model_type).lower()
        return embed_fn, inner.layers, inner.norm, lm_head, needs_scale

    raise RuntimeError("Could not detect model structure.")


def get_residuals_and_logits(model, tokenizer, prompt, _cache={}):
    """Run forward pass, capture per-layer residuals AND top predictions."""
    if 'parts' not in _cache:
        _cache['parts'] = _find_model_parts(model)
    embed_fn, layers, norm, lm_head, needs_scale = _cache['parts']

    tokens = tokenizer.encode(prompt)
    input_ids = mx.array([tokens])

    try:
        h = embed_fn(input_ids)
        if needs_scale:
            import math
            h = h * math.sqrt(h.shape[-1])

        seq_len = h.shape[1]
        mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len)
        mask = mask.astype(h.dtype)

        residuals = {}
        for i, layer in enumerate(layers):
            h = layer(h, mask=mask)
            mx.eval(h)
            residuals[i] = np.array(h[0, -1, :].astype(mx.float32))

        h_normed = norm(h[:, -1:, :])
        logits = lm_head(h_normed)
        mx.eval(logits)

        logits_np = np.array(logits[0, 0, :].astype(mx.float32))
        top_indices = np.argsort(-logits_np)[:20]
        top_logits = logits_np[top_indices]
        top_logits = top_logits - top_logits.max()
        probs = np.exp(top_logits)
        probs = probs / probs.sum()

        top_predictions = []
        for idx, prob in zip(top_indices, probs):
            token_str = tokenizer.decode([int(idx)]).strip()
            if len(token_str) >= 2:
                top_predictions.append((token_str.lower(), float(prob)))

        return residuals, top_predictions
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None, None


# ── Resume support ────────────────────────────────────────────────────

def load_progress(progress_path: Path) -> set[tuple[str, str]]:
    """Load set of (relation, entity) pairs already probed."""
    if not progress_path.exists():
        return set()
    completed = set()
    with open(progress_path) as f:
        for line in f:
            line = line.strip()
            if line:
                parts = line.split("\t", 1)
                if len(parts) == 2:
                    completed.add((parts[0], parts[1]))
    return completed


def save_progress_batch(progress_path: Path, items: list[tuple[str, str]]) -> None:
    """Append completed (relation, entity) pairs to progress file."""
    with open(progress_path, "a") as f:
        for rel, entity in items:
            f.write(f"{rel}\t{entity}\n")


# ── CLI ───────────────────────────────────────────────────────────────

def _model_slug(model_id: str) -> str:
    return model_id.split("/")[-1]


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Probe model features with MLX inference",
    )
    parser.add_argument(
        "--model", type=str, default="google/gemma-3-4b-it",
        help="HuggingFace model ID (default: google/gemma-3-4b-it)",
    )
    parser.add_argument(
        "--vindex", type=str, default=None,
        help="Path to vindex directory",
    )
    parser.add_argument(
        "--triples", type=str, default=_DEFAULT_TRIPLES,
        help="Path to combined triples JSON",
    )
    parser.add_argument(
        "--templates", type=str, default=None,
        help="Path to probe templates JSON",
    )
    parser.add_argument(
        "--output", type=str, default=_DEFAULT_PROBES,
        help="Probes output directory",
    )
    parser.add_argument(
        "--layers", type=str, default="knowledge",
        choices=["knowledge", "syntax", "all"],
        help="Which layer bands to probe (default: knowledge)",
    )
    parser.add_argument(
        "--relations", type=str, default=None,
        help="Comma-separated list of relations to probe (default: all)",
    )
    parser.add_argument(
        "--skip-relations", type=str, default=None,
        help="Comma-separated list of relations to skip",
    )
    parser.add_argument(
        "--top-k", type=int, default=50,
        help="Top-K gate features to check per layer (default: 50)",
    )
    parser.add_argument(
        "--min-gate-score", type=float, default=5.0,
        help="Minimum |gate score| threshold (default: 5.0)",
    )
    parser.add_argument(
        "--resume", action="store_true",
        help="Resume from last checkpoint (skip already-probed entities)",
    )
    parser.add_argument(
        "--offline", action="store_true", default=True,
        help="Use cached model only (default: true)",
    )
    return parser.parse_args()


# ── Main ──────────────────────────────────────────────────────────────

def main() -> None:
    """Run the probe."""
    args = parse_args()
    model_id = args.model
    model_slug = _model_slug(model_id)

    # ── Load vindex ──
    vindex_path = args.vindex
    gates, down_meta, has_vindex = {}, {}, False
    num_layers = 34  # fallback

    if vindex_path is None:
        default_vindex = _REPO_ROOT / "output" / f"{model_slug}.vindex"
        if default_vindex.exists():
            vindex_path = str(default_vindex)

    if vindex_path and Path(vindex_path).exists():
        print("Loading vindex gates and metadata...")
        config, gates, down_meta = load_vindex_gates_and_meta(vindex_path)
        num_layers = config["num_layers"]
        has_vindex = True
        print(f"  {num_layers} layers, {config['hidden_size']} hidden, {len(down_meta)} features")
    else:
        print("No vindex found — prediction-only mode")

    # ── Compute layer ranges ──
    # Use vindex layer_bands if available, otherwise derive from num_layers
    if has_vindex and "layer_bands" in config and config["layer_bands"]:
        bands = config["layer_bands"]
        syntax_end = bands.get("knowledge_start", num_layers * 2 // 5)
        knowledge_start = syntax_end
        knowledge_end = bands.get("output_start", num_layers * 4 // 5)
    else:
        # Default: 0-40% syntax, 40-80% knowledge, 80-100% output
        knowledge_start = num_layers * 2 // 5
        knowledge_end = num_layers * 4 // 5
        syntax_end = knowledge_start

    if args.layers == "knowledge":
        scan_layers = list(range(knowledge_start, knowledge_end))
    elif args.layers == "syntax":
        scan_layers = list(range(0, syntax_end))
    else:  # all
        scan_layers = list(range(0, knowledge_end))

    print(
        f"  Layer bands: syntax L0-L{syntax_end - 1},"
        f" knowledge L{knowledge_start}-L{knowledge_end - 1}"
    )
    print(f"  Scanning: L{scan_layers[0]}-L{scan_layers[-1]} ({args.layers} mode)")

    # ── Load data sources ──
    print("Loading triples...")
    with open(args.triples) as f:
        triples = json.load(f)

    syntax = {}
    if args.layers in ("syntax", "all"):
        print("Loading syntax data (WordNet, morphological, AST)...")
        syntax = load_syntax_data()
        print(f"  {len(syntax)} syntax relations loaded")

    all_data = {**triples, **syntax}

    # Build separate match indexes per layer band to avoid cross-contamination:
    # syntax layers only match against syntax data, knowledge layers only against knowledge data
    print("Building match indexes...")
    knowledge_index = build_match_index(triples)
    syntax_index = build_match_index(syntax) if syntax else {}
    print(
        f"  knowledge: {len(knowledge_index)} entries,"
        f" syntax: {len(syntax_index)} entries"
    )

    # ── Load templates ──
    TEMPLATES = load_templates(args.templates)

    # For syntax relations, auto-generate identity templates
    # (WordNet/morph pairs are word→word, not prompted)
    if args.layers in ("syntax", "all"):
        for rel_name in syntax:
            if rel_name not in TEMPLATES:
                # Identity template: just the word itself
                TEMPLATES[rel_name] = ["{X}"]

    # ── Filter relations ──
    if args.relations:
        wanted = set(args.relations.split(","))
        TEMPLATES = {r: t for r, t in TEMPLATES.items() if r in wanted}
    if args.skip_relations:
        skip = set(args.skip_relations.split(","))
        TEMPLATES = {r: t for r, t in TEMPLATES.items() if r not in skip}

    # Only keep relations that have data
    TEMPLATES = {r: t for r, t in TEMPLATES.items() if r in all_data}
    print(f"Probing {len(TEMPLATES)} relations")

    # ── Load model ──
    print(f"Loading MLX model: {model_id}...")
    import os
    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
    model, tokenizer = mlx_load(model_id)
    print("  Model loaded")

    # ── Quick test ──
    print("\nTest: 'The capital of France is'")
    residuals, predictions = get_residuals_and_logits(
        model, tokenizer, "The capital of France is"
    )
    if residuals is None:
        print("ERROR: Could not capture residuals")
        sys.exit(1)
    print(f"  Residuals at {len(residuals)} layers, predictions: {predictions[:3]}")

    # ── Resume support ──
    probe_dir = Path(args.output) / model_slug
    probe_dir.mkdir(parents=True, exist_ok=True)
    progress_path = probe_dir / "probe_progress.tsv"

    completed = set()
    if args.resume:
        completed = load_progress(progress_path)
        if completed:
            print(f"  Resuming: {len(completed)} (relation, entity) pairs already done")

    # ── Probe loop ──
    feature_hits = defaultdict(lambda: defaultdict(int))
    feature_entities = defaultdict(lambda: defaultdict(set))
    feature_outputs = defaultdict(lambda: defaultdict(set))
    total_probes = 0
    total_skipped = 0
    start_time = time.time()
    progress_batch = []

    for rel_name, templates in TEMPLATES.items():
        if rel_name not in all_data:
            continue

        all_subjects = list(set(
            pair[0] for pair in all_data[rel_name].get("pairs", [])
            if len(pair) >= 2 and 2 <= len(pair[0]) <= 30
        ))
        all_subjects.sort(key=lambda s: (len(s.split()), len(s)))

        if not all_subjects:
            continue

        gate_matched = 0
        pred_matched = 0
        rel_start = time.time()
        probe_items = [(s, t) for s in all_subjects for t in templates]

        for pi, (subject, template) in enumerate(probe_items):
            # Resume: skip already-probed pairs
            if (rel_name, subject) in completed:
                total_skipped += 1
                continue

            prompt = template.replace("{X}", subject)
            residuals, predictions = get_residuals_and_logits(
                model, tokenizer, prompt
            )
            if residuals is None:
                continue

            total_probes += 1
            subj_key = subject.lower().strip()

            if (pi + 1) % 50 == 0:
                elapsed_rel = time.time() - rel_start
                rate = max((pi + 1 - total_skipped) / elapsed_rel, 0.1) if elapsed_rel > 0 else 1
                eta = max(0, (len(probe_items) - pi - 1) / rate)
                sys.stdout.write(
                    f"\r  {rel_name:<20s} {pi+1}/{len(probe_items)}"
                    f" ({gate_matched + pred_matched} hits, {rate:.0f}/s,"
                    f" ETA {eta:.0f}s)"
                )
                sys.stdout.flush()

            # ── Gate matching ──
            if has_vindex:
                for layer in scan_layers:
                    if layer not in residuals or layer not in gates:
                        continue
                    # Use the right index for this layer band
                    layer_index = syntax_index if layer < syntax_end else knowledge_index
                    r = residuals[layer]
                    scores = gates[layer] @ r
                    top_indices = np.argsort(-np.abs(scores))[:args.top_k]

                    for feat_idx in top_indices:
                        score = float(scores[feat_idx])
                        if abs(score) < args.min_gate_score:
                            continue
                        tokens = down_meta.get((layer, int(feat_idx)), [])
                        if not tokens:
                            continue

                        feat_key = f"L{layer}_F{feat_idx}"
                        for target in tokens:
                            if len(target) < 2:
                                continue
                            tgt_lower = target.lower().strip()
                            key = (subj_key, tgt_lower)
                            if layer_index.get(key) == rel_name:
                                feature_hits[feat_key][rel_name] += 1
                                feature_entities[feat_key][rel_name].add(subject)
                                feature_outputs[feat_key][rel_name].add(tgt_lower)
                                gate_matched += 1
                                break

            # ── Prediction matching ──
            # Predictions come from the LM head (full model output).
            # Match against knowledge index — predictions are knowledge-level.
            if predictions and has_vindex:
                for pred_token, pred_prob in predictions[:10]:
                    if pred_prob < 0.01:
                        break
                    key = (subj_key, pred_token)
                    if knowledge_index.get(key) == rel_name:
                        # Attribute to top features at peak scan layers
                        for layer in scan_layers[-4:]:
                            if layer not in residuals or layer not in gates:
                                continue
                            r = residuals[layer]
                            scores = gates[layer] @ r
                            top_idx = int(np.argmax(np.abs(scores)))
                            feat_key = f"L{layer}_F{top_idx}"
                            feature_hits[feat_key][rel_name] += 1
                        pred_matched += 1
                        break

            # Track progress
            progress_batch.append((rel_name, subject))
            if len(progress_batch) >= 100:
                save_progress_batch(progress_path, progress_batch)
                progress_batch.clear()

        # Flush remaining progress
        if progress_batch:
            save_progress_batch(progress_path, progress_batch)
            progress_batch.clear()

        elapsed = time.time() - start_time
        rate = total_probes / elapsed if elapsed > 0 else 0
        total_matched = gate_matched + pred_matched
        print(
            f"  {rel_name:<20s} {len(all_subjects):3d} entities"
            f" x {len(templates)} templates -> {total_matched:3d} hits"
            f"  (gate={gate_matched}, pred={pred_matched})"
            f"  ({rate:.1f} probes/s)"
        )

    elapsed = time.time() - start_time
    print(
        f"\nTotal: {total_probes} probes in {elapsed:.0f}s"
        f" ({total_probes / max(elapsed, 1):.1f}/s)"
    )
    if total_skipped:
        print(f"  Skipped {total_skipped} already-probed pairs (resume)")
    print(f"Features with hits: {len(feature_hits)}")

    # ── Resolve multi-label features ──
    all_labels = {}
    label_details = {}
    relation_totals = defaultdict(int)

    for feat_key, rel_counts in feature_hits.items():
        total_hits = sum(rel_counts.values())
        primary_rel = max(rel_counts, key=rel_counts.get)
        primary_count = rel_counts[primary_rel]
        confidence = primary_count / total_hits

        if primary_count >= 2 and confidence > 0.5:
            all_labels[feat_key] = primary_rel
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
                "relations": {
                    r: c for r, c in sorted(
                        rel_counts.items(), key=lambda x: -x[1]
                    )
                },
            }

    print(
        f"Labeled: {len(all_labels)} features"
        f" ({len(feature_hits) - len(all_labels)} dropped)"
    )

    if relation_totals:
        print(f"\nRelation distribution ({len(relation_totals)} relations):")
        for rel, count in sorted(relation_totals.items(), key=lambda x: -x[1]):
            print(f"  {rel:<25s} {count:4d}")

    # ── Save ──
    if has_vindex:
        vindex_labels = Path(vindex_path) / "feature_labels.json"
        # Merge with existing (don't overwrite labels from other runs)
        existing = {}
        if vindex_labels.exists():
            with open(vindex_labels) as f:
                existing = json.load(f)
        for key, rel in all_labels.items():
            existing[key] = rel  # new labels overwrite old for same feature
        with open(vindex_labels, "w") as f:
            json.dump(existing, f, indent=2)
        print(f"\nVindex: {len(existing)} total labels -> {vindex_labels}")

    probe_path = probe_dir / "feature_labels.json"
    # Merge with existing probe labels too
    existing_probes = {}
    if probe_path.exists():
        with open(probe_path) as f:
            existing_probes = json.load(f)
    for key, rel in all_labels.items():
        existing_probes[key] = rel
    with open(probe_path, "w") as f:
        json.dump(existing_probes, f, indent=2)
    print(f"Probes: {len(existing_probes)} total labels -> {probe_path}")

    details_path = probe_dir / "feature_labels_rich.json"
    with open(details_path, "w") as f:
        json.dump(label_details, f, indent=2)
    print(f"Details: {len(label_details)} entries -> {details_path}")


if __name__ == "__main__":
    main()
