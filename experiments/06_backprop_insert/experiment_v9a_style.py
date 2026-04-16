#!/usr/bin/env python3
"""
v9a: Style Engine

Prove that style is structured data, not a neural mystery.

Three components:
  1. Connotation Graph — enriches WordNet synonyms with formality, warmth,
     concreteness, complexity axes. Picks the RIGHT synonym for the register.
  2. Style Profiles — per-register grammar preferences (sentence length,
     voice, clause depth, conjunction style).
  3. Discourse Templates — structural patterns for different text types.

Test: generate the same factual content in 5 different styles using
the three-system model from v8 + style engine. Measure whether the
outputs are recognisably different.
"""

import os
import sys
import json
import time
import math
import random
from collections import defaultdict
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
nltk.download('sentiwordnet', quiet=True)
from nltk.corpus import wordnet as wn
from nltk.corpus import sentiwordnet as swn
from wordfreq import zipf_frequency

from model import TinyGemma
from synth_data_v2 import build_mixed_corpus, GroundTruth

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

N_LAYERS = 12
DIM = 256
FFN_DIM = 1024
EPOCHS = 20
BATCH_SIZE = 8
LR = 3e-4
MAX_SEQ = 64
SEED = 42
VOCAB = 32000

OUTPUT_DIR = "results_v9a_style"

# ---------------------------------------------------------------------------
# Shared infrastructure
# ---------------------------------------------------------------------------

class ClampedTokenizer:
    def __init__(self, tok, vocab):
        self.tok = tok
        self.vocab = vocab
        self.pad_token_id = tok.pad_token_id or 0
    def encode(self, text, **kwargs):
        ids = self.tok.encode(text, **kwargs)
        return [min(i, self.vocab - 1) for i in ids]
    def decode_token(self, tid):
        return self.tok.decode([tid]).strip()


class TokenDataset(Dataset):
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = []
        for text in texts:
            ids = tokenizer.encode(text, add_special_tokens=True,
                                   max_length=max_len, truncation=True)
            self.encodings.append(ids)
    def __len__(self):
        return len(self.encodings)
    def __getitem__(self, idx):
        return torch.tensor(self.encodings[idx], dtype=torch.long)


def collate_fn(batch, pad_id=0):
    max_len = max(len(x) for x in batch)
    padded = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    for i, x in enumerate(batch):
        padded[i, :len(x)] = x
    return padded


def train_baseline(loader, tokenizer, device, epochs=EPOCHS):
    print(f"\n  Training baseline ({epochs} epochs)...")
    torch.manual_seed(SEED)
    model = TinyGemma(
        vocab_size=VOCAB, dim=DIM, n_layers=N_LAYERS, ffn_dim=FFN_DIM,
        n_heads=4, n_kv_heads=2, max_seq=MAX_SEQ,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    t0 = time.time()
    for epoch in range(epochs):
        eloss = 0; n = 0
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            logits = model(batch)
            loss = F.cross_entropy(
                logits[:, :-1, :].contiguous().view(-1, VOCAB),
                batch[:, 1:].contiguous().view(-1),
                ignore_index=tokenizer.pad_token_id)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            eloss += loss.item(); n += 1
        avg = eloss / n
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"    E{epoch+1:2d}/{epochs} loss={avg:.4f} {time.time()-t0:.0f}s")
        sys.stdout.flush()
    print(f"  Baseline done: loss={avg:.4f}")
    return model


# ---------------------------------------------------------------------------
# Component 1: Connotation Graph
# ---------------------------------------------------------------------------

class ConnotationGraph:
    """
    Enriches WordNet synonyms with connotation axes:
    formality, warmth, concreteness, complexity, intensity.

    Picks the RIGHT synonym for a given style register.
    """

    def __init__(self):
        self.entries = {}  # word → {axes: {formality: float, ...}, synonyms: [str]}
        self._build()

    def _build(self):
        print("\n  Building connotation graph...")

        # Collect synonym sets with connotation scoring
        n_words = 0
        for synset in wn.all_synsets():
            if synset.pos() not in ('n', 'v', 'a', 'r'):
                continue

            lemmas = [l.name() for l in synset.lemmas()
                      if l.name().isalpha() and l.name().islower() and len(l.name()) >= 3]
            if len(lemmas) < 2:
                continue

            # Get SentiWordNet scores
            senti = list(swn.senti_synsets(synset.name()))
            pos_score = senti[0].pos_score() if senti else 0.0
            neg_score = senti[0].neg_score() if senti else 0.0

            for word in lemmas:
                if word in self.entries:
                    continue

                # Compute connotation axes
                freq = zipf_frequency(word, 'en')

                axes = {
                    # Formality: rare words tend to be more formal
                    "formality": max(-1, min(1, (4.0 - freq) / 3.0)),
                    # Warmth: from sentiment (positive = warm, negative = cold)
                    "warmth": pos_score - neg_score,
                    # Concreteness: shorter words with higher frequency tend to be more concrete
                    "concreteness": max(-1, min(1, (freq - 3.0) / 3.0 - len(word) / 15.0)),
                    # Complexity: word length + syllable estimate
                    "complexity": min(1, (len(word) - 3) / 10.0),
                    # Intensity: from sentiment strength
                    "intensity": pos_score + neg_score,
                }

                other_synonyms = [l for l in lemmas if l != word]
                self.entries[word] = {
                    "axes": axes,
                    "synonyms": other_synonyms,
                    "pos": synset.pos(),
                    "frequency": freq,
                }
                n_words += 1

                if n_words >= 5000:
                    break
            if n_words >= 5000:
                break

        print(f"  Connotation graph: {len(self.entries)} words")

    def get_synonym_for_style(self, word: str, style: Dict[str, float]) -> str:
        """
        Given a word and target style (axis→value dict),
        pick the synonym that best matches the style.
        """
        entry = self.entries.get(word.lower())
        if not entry or not entry["synonyms"]:
            return word

        candidates = [word] + entry["synonyms"]
        best_word = word
        best_score = -999

        for candidate in candidates:
            cand_entry = self.entries.get(candidate)
            if not cand_entry:
                continue

            # Score: cosine-like match between candidate axes and target style
            score = 0
            for axis, target in style.items():
                actual = cand_entry["axes"].get(axis, 0)
                score -= abs(actual - target)  # closer = better

            if score > best_score:
                best_score = score
                best_word = candidate

        return best_word

    def get_all_variants(self, word: str) -> List[Tuple[str, Dict]]:
        """Get all synonym variants with their connotation axes."""
        entry = self.entries.get(word.lower())
        if not entry:
            return [(word, {})]

        variants = []
        for syn in [word] + entry["synonyms"]:
            syn_entry = self.entries.get(syn)
            if syn_entry:
                variants.append((syn, syn_entry["axes"]))
        return variants

    def to_json(self) -> str:
        """Export as readable JSON."""
        export = {}
        for word, data in sorted(self.entries.items()):
            if data["synonyms"]:  # only words with synonyms
                export[word] = {
                    "synonyms": data["synonyms"],
                    "formality": round(data["axes"]["formality"], 2),
                    "warmth": round(data["axes"]["warmth"], 2),
                    "concreteness": round(data["axes"]["concreteness"], 2),
                    "complexity": round(data["axes"]["complexity"], 2),
                }
        return json.dumps(export, indent=2)


# ---------------------------------------------------------------------------
# Component 2: Style Profiles
# ---------------------------------------------------------------------------

STYLE_PROFILES = {
    "hemingway": {
        "name": "Hemingway (terse, concrete)",
        "target_axes": {
            "formality": -0.5,
            "warmth": 0.2,
            "concreteness": 0.8,
            "complexity": -0.5,
            "intensity": 0.3,
        },
        "grammar": {
            "max_tokens_per_sentence": 12,
            "prefer_short_words": True,
            "conjunction_style": "and",
            "allow_fragments": True,
        },
    },
    "academic": {
        "name": "Academic (formal, complex)",
        "target_axes": {
            "formality": 0.8,
            "warmth": -0.3,
            "concreteness": -0.2,
            "complexity": 0.7,
            "intensity": 0.1,
        },
        "grammar": {
            "max_tokens_per_sentence": 35,
            "prefer_short_words": False,
            "conjunction_style": "furthermore, moreover, additionally",
            "allow_fragments": False,
        },
    },
    "casual": {
        "name": "Casual (informal, warm)",
        "target_axes": {
            "formality": -0.8,
            "warmth": 0.7,
            "concreteness": 0.5,
            "complexity": -0.7,
            "intensity": 0.5,
        },
        "grammar": {
            "max_tokens_per_sentence": 10,
            "prefer_short_words": True,
            "conjunction_style": "but, so, like",
            "allow_fragments": True,
        },
    },
    "legal": {
        "name": "Legal (formal, precise)",
        "target_axes": {
            "formality": 0.9,
            "warmth": -0.8,
            "concreteness": 0.3,
            "complexity": 0.9,
            "intensity": -0.2,
        },
        "grammar": {
            "max_tokens_per_sentence": 50,
            "prefer_short_words": False,
            "conjunction_style": "whereas, notwithstanding, provided that",
            "allow_fragments": False,
        },
    },
    "poetic": {
        "name": "Poetic (warm, intense, abstract)",
        "target_axes": {
            "formality": 0.3,
            "warmth": 0.8,
            "concreteness": -0.5,
            "complexity": 0.4,
            "intensity": 0.9,
        },
        "grammar": {
            "max_tokens_per_sentence": 20,
            "prefer_short_words": False,
            "conjunction_style": "and, or, yet",
            "allow_fragments": True,
        },
    },
}


# ---------------------------------------------------------------------------
# Component 3: Discourse Templates
# ---------------------------------------------------------------------------

DISCOURSE_TEMPLATES = {
    "factual_description": {
        "slots": ["subject_intro", "key_fact_1", "key_fact_2", "elaboration", "conclusion"],
        "connectors": {
            "hemingway": ["", ". ", ". ", ". ", "."],
            "academic": ["", ". Furthermore, ", ". Additionally, ", ". It is noteworthy that ", ". In conclusion, "],
            "casual": ["So ", "! ", ". Also, ", ". Oh and ", "!"],
            "legal": ["Regarding ", ". It is hereby noted that ", ". In addition, ", ". Subject to the foregoing, ", "."],
            "poetic": ["", " — ", ", and ", ", where ", "."],
        },
    },
    "entity_description": {
        "slots": ["entity_name", "primary_relation", "secondary_relation", "context"],
        "connectors": {
            "hemingway": ["", " is ", ". ", "."],
            "academic": ["The entity known as ", " is characterised by ", ". Moreover, ", "."],
            "casual": ["", " is like ", ". Plus ", "!"],
            "legal": ["With respect to ", ", the principal attribute is ", ". Additionally, ", "."],
            "poetic": ["", " holds ", ", and in its heart ", "."],
        },
    },
}


# ---------------------------------------------------------------------------
# Style-Aware Output Engine
# ---------------------------------------------------------------------------

class StyleAwareOutputEngine:
    """
    Extends the v8 output engine with style-conditioned token boosting.

    Instead of uniformly boosting tokens, this engine adjusts token
    probabilities based on the active style profile:
    - Formal style → boost longer, rarer words
    - Casual style → boost shorter, common words
    - Poetic style → boost warm, intense words
    """

    def __init__(self, connotation: ConnotationGraph, tokenizer: ClampedTokenizer,
                 device: torch.device, dim: int):
        self.connotation = connotation
        self.tokenizer = tokenizer
        self.device = device
        self.dim = dim
        self.active_style = "casual"

        # Build token→connotation lookup
        self._build_token_axes()

    def _build_token_axes(self):
        """Pre-compute connotation axes for vocabulary tokens."""
        print("    Building token connotation lookup...")

        # For each token in vocab that maps to a word in our connotation graph,
        # store its axes as a vector
        self.token_axes = torch.zeros(VOCAB, 5, device=self.device)  # 5 axes
        self.token_has_axes = torch.zeros(VOCAB, device=self.device)

        axis_names = ["formality", "warmth", "concreteness", "complexity", "intensity"]
        matched = 0

        for tid in range(min(VOCAB, 32000)):
            word = self.tokenizer.decode_token(tid).strip().lower()
            if not word or not word.isalpha():
                continue
            entry = self.connotation.entries.get(word)
            if entry:
                for i, axis in enumerate(axis_names):
                    self.token_axes[tid, i] = entry["axes"].get(axis, 0)
                self.token_has_axes[tid] = 1.0
                matched += 1

        print(f"    Matched {matched} tokens to connotation axes")

    def set_style(self, style_name: str):
        """Set the active style profile."""
        self.active_style = style_name

    def compute_style_bias(self, style_name: str) -> torch.Tensor:
        """
        Compute a bias vector over the vocabulary for the given style.
        Returns: (vocab,) — positive = boost, negative = suppress.
        """
        profile = STYLE_PROFILES.get(style_name)
        if not profile:
            return torch.zeros(VOCAB, device=self.device)

        target = profile["target_axes"]
        axis_names = ["formality", "warmth", "concreteness", "complexity", "intensity"]
        target_vec = torch.tensor(
            [target.get(a, 0) for a in axis_names],
            device=self.device,
        )  # (5,)

        # Cosine-like scoring: tokens whose axes match the target get boosted
        # token_axes: (vocab, 5), target_vec: (5,)
        scores = self.token_axes @ target_vec  # (vocab,)

        # Zero out tokens without axes (don't bias them)
        scores = scores * self.token_has_axes

        # Normalise to reasonable range (-2 to +2)
        if scores.abs().max() > 0:
            scores = scores / (scores.abs().max() + 1e-10) * 2.0

        return scores


# ---------------------------------------------------------------------------
# Style-Conditioned Generation
# ---------------------------------------------------------------------------

def generate_styled(
    model: TinyGemma,
    prompt_ids: List[int],
    style_engine: StyleAwareOutputEngine,
    style_name: str,
    tokenizer: ClampedTokenizer,
    device: torch.device,
    max_new_tokens: int = 40,
    temperature: float = 0.8,
) -> Tuple[str, List[int]]:
    """
    Generate text with style-conditioned output bias.
    Uses the connotation graph to bias token selection toward the target style.
    """
    model.eval()
    style_bias = style_engine.compute_style_bias(style_name)
    grammar = STYLE_PROFILES[style_name]["grammar"]

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    generated = list(prompt_ids)
    tokens_since_period = 0
    max_sent_len = grammar["max_tokens_per_sentence"]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(input_ids)
            next_logits = logits[0, -1] / temperature  # (vocab,)

            # Apply style bias
            next_logits = next_logits + style_bias * 0.5

            # Grammar constraints: encourage sentence breaks at max length
            if tokens_since_period >= max_sent_len:
                # Boost period/stop tokens
                period_ids = tokenizer.encode(".", add_special_tokens=False)
                for pid in period_ids:
                    if pid < len(next_logits):
                        next_logits[pid] += 3.0

            # Sample
            probs = F.softmax(next_logits, dim=0)
            next_token = torch.multinomial(probs, 1).item()

            generated.append(next_token)
            input_ids = torch.tensor([generated[-MAX_SEQ:]], dtype=torch.long, device=device)

            # Track sentence length
            decoded = tokenizer.decode_token(next_token)
            if decoded in '.!?':
                tokens_since_period = 0
            else:
                tokens_since_period += 1

            # Stop on EOS
            if next_token == tokenizer.tok.eos_token_id:
                break

    return tokenizer.tok.decode(generated), generated


# ---------------------------------------------------------------------------
# Style Metrics
# ---------------------------------------------------------------------------

def compute_style_metrics(text: str, style_name: str, connotation: ConnotationGraph) -> Dict:
    """Measure how well the generated text matches the target style."""
    words = [w.lower() for w in text.split() if w.isalpha()]
    sentences = [s.strip() for s in text.replace('!', '.').replace('?', '.').split('.') if s.strip()]

    if not words:
        return {"error": "no words"}

    profile = STYLE_PROFILES[style_name]
    target = profile["target_axes"]

    # Word-level metrics
    avg_word_len = sum(len(w) for w in words) / len(words)
    avg_freq = sum(zipf_frequency(w, 'en') for w in words) / len(words)

    # Connotation alignment
    axis_scores = defaultdict(list)
    for word in words:
        entry = connotation.entries.get(word)
        if entry:
            for axis, val in entry["axes"].items():
                axis_scores[axis].append(val)

    connotation_alignment = {}
    for axis in ["formality", "warmth", "concreteness", "complexity", "intensity"]:
        if axis_scores[axis]:
            actual = sum(axis_scores[axis]) / len(axis_scores[axis])
            target_val = target.get(axis, 0)
            connotation_alignment[axis] = {
                "actual": round(actual, 3),
                "target": target_val,
                "error": round(abs(actual - target_val), 3),
            }

    # Sentence-level metrics
    avg_sent_len = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
    target_sent_len = profile["grammar"]["max_tokens_per_sentence"]

    # Vocabulary richness (type-token ratio)
    ttr = len(set(words)) / max(len(words), 1)

    return {
        "n_words": len(words),
        "n_sentences": len(sentences),
        "avg_word_length": round(avg_word_len, 2),
        "avg_word_frequency": round(avg_freq, 2),
        "avg_sentence_length": round(avg_sent_len, 1),
        "target_sentence_length": target_sent_len,
        "type_token_ratio": round(ttr, 3),
        "connotation_alignment": connotation_alignment,
    }


def compute_style_divergence(metrics_by_style: Dict[str, Dict]) -> Dict:
    """Measure how different the styles are from each other."""
    styles = list(metrics_by_style.keys())
    divergence = {}

    for i, s1 in enumerate(styles):
        for s2 in styles[i+1:]:
            m1 = metrics_by_style[s1]
            m2 = metrics_by_style[s2]

            # Vocabulary overlap
            if "words_used" in m1 and "words_used" in m2:
                overlap = len(m1["words_used"] & m2["words_used"]) / max(
                    len(m1["words_used"] | m2["words_used"]), 1)
            else:
                overlap = 0

            # Metric differences
            wl_diff = abs(m1.get("avg_word_length", 0) - m2.get("avg_word_length", 0))
            sl_diff = abs(m1.get("avg_sentence_length", 0) - m2.get("avg_sentence_length", 0))
            freq_diff = abs(m1.get("avg_word_frequency", 0) - m2.get("avg_word_frequency", 0))

            divergence[f"{s1} vs {s2}"] = {
                "word_length_diff": round(wl_diff, 2),
                "sentence_length_diff": round(sl_diff, 1),
                "frequency_diff": round(freq_diff, 2),
            }

    return divergence


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 65)
    print("  v9a: STYLE ENGINE")
    print("  Style is structured data, not a neural mystery.")
    print("=" * 65)

    device = torch.device("cpu")
    print(f"\n  Device: CPU")

    # Setup
    print("  Loading tokenizer...")
    raw_tok = AutoTokenizer.from_pretrained("google/gemma-3-4b-pt")
    if raw_tok.pad_token_id is None:
        raw_tok.pad_token_id = 0
    tokenizer = ClampedTokenizer(raw_tok, VOCAB)

    print("  Building corpus...")
    samples, ground_truth = build_mixed_corpus(n_countries=50, seed=SEED)

    dataset = TokenDataset([s.text for s in samples], tokenizer, MAX_SEQ)
    loader = DataLoader(
        dataset, batch_size=BATCH_SIZE, shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer.pad_token_id),
        drop_last=True,
    )

    # ═══════════════════════════════════════════════════════════════
    # Phase 0: Train baseline
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 0: Train baseline")
    print(f"{'='*65}")

    trained = train_baseline(loader, tokenizer, device)

    # ═══════════════════════════════════════════════════════════════
    # Phase 1: Build connotation graph
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 1: Connotation Graph")
    print(f"{'='*65}")

    connotation = ConnotationGraph()

    # Show examples
    print(f"\n  Example connotation variants:")
    test_words = ["big", "happy", "walk", "house", "fast", "old", "smart", "cold"]
    for word in test_words:
        variants = connotation.get_all_variants(word)
        if len(variants) > 1:
            var_str = ", ".join(
                f"{w}(f={axes.get('formality',0):.1f})"
                for w, axes in variants[:4]
            )
            print(f"    {word}: {var_str}")

    # Style-conditioned synonym selection
    print(f"\n  Style-conditioned synonym selection:")
    for style_name, profile in STYLE_PROFILES.items():
        target = profile["target_axes"]
        selections = []
        for word in test_words:
            selected = connotation.get_synonym_for_style(word, target)
            if selected != word:
                selections.append(f"{word}→{selected}")
        if selections:
            print(f"    {style_name}: {', '.join(selections[:5])}")

    # Export connotation graph
    cg_json = connotation.to_json()
    with open(os.path.join(OUTPUT_DIR, "connotation_graph.json"), "w") as f:
        f.write(cg_json)
    print(f"\n  Connotation graph exported ({len(cg_json)} bytes)")

    # ═══════════════════════════════════════════════════════════════
    # Phase 2: Build style-aware output engine
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 2: Style-Aware Output Engine")
    print(f"{'='*65}")

    style_engine = StyleAwareOutputEngine(connotation, tokenizer, device, DIM)

    # Show bias analysis per style
    print(f"\n  Style bias analysis (top boosted tokens per style):")
    for style_name in STYLE_PROFILES:
        bias = style_engine.compute_style_bias(style_name)
        top_idx = bias.topk(10).indices.tolist()
        top_words = [tokenizer.decode_token(t) for t in top_idx]
        top_scores = [f"{bias[t]:.2f}" for t in top_idx]
        print(f"    {style_name}: {', '.join(f'{w}({s})' for w, s in zip(top_words[:6], top_scores[:6]))}")

    # ═══════════════════════════════════════════════════════════════
    # Phase 3: Generate same content in 5 styles
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 3: Generate same knowledge in 5 styles")
    print(f"{'='*65}")

    # Test prompts — same factual seed, different style output
    test_prompts = [
        "The capital of Freedonia is",
        "A dog is a type of",
        "The president of Sylvania is",
        "Every house has a",
    ]

    all_generations = {}
    all_metrics = {}

    for prompt_text in test_prompts:
        print(f"\n  Prompt: \"{prompt_text}\"")
        prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=True)

        prompt_generations = {}
        prompt_metrics = {}

        for style_name in STYLE_PROFILES:
            torch.manual_seed(SEED)  # same seed for fair comparison
            text, ids = generate_styled(
                trained, prompt_ids, style_engine, style_name,
                tokenizer, device, max_new_tokens=30, temperature=0.8,
            )

            # Trim prompt from output
            output_text = tokenizer.tok.decode(ids[len(prompt_ids):])
            prompt_generations[style_name] = output_text

            metrics = compute_style_metrics(output_text, style_name, connotation)
            prompt_metrics[style_name] = metrics

            # Show
            style_label = STYLE_PROFILES[style_name]["name"]
            print(f"    [{style_label}]:")
            print(f"      {output_text[:100]}")

        all_generations[prompt_text] = prompt_generations
        all_metrics[prompt_text] = prompt_metrics

    # ═══════════════════════════════════════════════════════════════
    # Phase 4: Measure style divergence
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  PHASE 4: Style Divergence Analysis")
    print(f"{'='*65}")

    # Aggregate metrics across all prompts per style
    agg_metrics = defaultdict(lambda: defaultdict(list))
    for prompt, prompt_metrics in all_metrics.items():
        for style, metrics in prompt_metrics.items():
            for key in ["avg_word_length", "avg_word_frequency", "avg_sentence_length", "type_token_ratio"]:
                if key in metrics:
                    agg_metrics[style][key].append(metrics[key])

    print(f"\n  Aggregate style metrics:")
    print(f"  {'Style':<15} {'WordLen':>8} {'WordFreq':>9} {'SentLen':>8} {'TTR':>6}")
    print(f"  {'─'*48}")

    style_summaries = {}
    for style in STYLE_PROFILES:
        m = agg_metrics[style]
        summary = {}
        for key in ["avg_word_length", "avg_word_frequency", "avg_sentence_length", "type_token_ratio"]:
            vals = m.get(key, [0])
            summary[key] = sum(vals) / max(len(vals), 1)

        style_summaries[style] = summary
        print(f"  {style:<15} {summary['avg_word_length']:>8.2f} {summary['avg_word_frequency']:>9.2f} "
              f"{summary['avg_sentence_length']:>8.1f} {summary['type_token_ratio']:>6.3f}")

    # Pairwise divergence
    print(f"\n  Pairwise style divergence:")
    divergence = compute_style_divergence(style_summaries)
    print(f"  {'Pair':<30} {'WordLen Δ':>10} {'SentLen Δ':>11} {'Freq Δ':>8}")
    print(f"  {'─'*62}")
    for pair, d in sorted(divergence.items()):
        print(f"  {pair:<30} {d['word_length_diff']:>10.2f} {d['sentence_length_diff']:>11.1f} "
              f"{d['frequency_diff']:>8.2f}")

    # Connotation alignment
    print(f"\n  Connotation alignment (actual vs target per axis):")
    for style in STYLE_PROFILES:
        print(f"\n  [{style}]")
        for prompt, prompt_metrics in all_metrics.items():
            metrics = prompt_metrics.get(style, {})
            ca = metrics.get("connotation_alignment", {})
            if ca:
                for axis, vals in ca.items():
                    print(f"    {axis}: actual={vals['actual']:+.2f} target={vals['target']:+.1f} "
                          f"error={vals['error']:.2f}")
                break  # one prompt is enough to show the pattern

    # ═══════════════════════════════════════════════════════════════
    # VERDICT
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print(f"  VERDICT")
    print(f"{'='*65}")

    # Are styles distinguishable?
    if divergence:
        avg_wl_div = sum(d["word_length_diff"] for d in divergence.values()) / len(divergence)
        avg_sl_div = sum(d["sentence_length_diff"] for d in divergence.values()) / len(divergence)
        avg_freq_div = sum(d["frequency_diff"] for d in divergence.values()) / len(divergence)

        print(f"\n  Average pairwise divergence:")
        print(f"    Word length: {avg_wl_div:.2f}")
        print(f"    Sentence length: {avg_sl_div:.1f}")
        print(f"    Word frequency: {avg_freq_div:.2f}")

        if avg_wl_div > 0.3 or avg_sl_div > 2.0 or avg_freq_div > 0.3:
            print(f"\n  ✓ Styles are DISTINGUISHABLE")
            print(f"    The connotation graph + style profiles produce measurably different output")
        else:
            print(f"\n  ~ Styles show some variation but are weak")
    else:
        print(f"\n  No divergence data")

    # Size
    print(f"\n  Style engine size:")
    print(f"    Connotation graph: {len(cg_json)} bytes ({len(connotation.entries)} words)")
    print(f"    Style profiles: {len(json.dumps(STYLE_PROFILES))} bytes (5 registers)")
    print(f"    Discourse templates: {len(json.dumps(DISCOURSE_TEMPLATES))} bytes")
    total_style = len(cg_json) + len(json.dumps(STYLE_PROFILES)) + len(json.dumps(DISCOURSE_TEMPLATES))
    print(f"    Total: {total_style:,} bytes ({total_style/1024:.0f} KB)")

    # Save
    results = {
        "generations": all_generations,
        "metrics": {prompt: {style: m for style, m in pm.items()}
                    for prompt, pm in all_metrics.items()},
        "aggregate": {style: dict(m) for style, m in agg_metrics.items()},
        "divergence": divergence,
        "connotation_size": len(cg_json),
        "style_profiles_size": len(json.dumps(STYLE_PROFILES)),
    }
    with open(os.path.join(OUTPUT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n  Results: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
