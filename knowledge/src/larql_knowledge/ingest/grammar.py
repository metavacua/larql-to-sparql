"""English grammar pair extraction.

Extracts syntactic bigram pairs from English text, grouped by
grammatical category:

    determiner -> noun      (the -> cat, a -> dog)
    preposition -> noun     (in -> house, on -> table)
    copula -> adjective     (is -> big, was -> happy)
    auxiliary -> verb        (can -> run, will -> go)

Uses known word lists rather than requiring an NLP library.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Word lists (curated for single-token entities)
# ---------------------------------------------------------------------------

DETERMINERS = {"the", "a", "an", "this", "that", "these", "those", "my",
               "your", "his", "her", "its", "our", "their", "some", "any",
               "no", "every", "each", "all", "both", "few", "many", "much",
               "several", "such"}

PREPOSITIONS = {"in", "on", "at", "to", "for", "with", "from", "by", "of",
                "about", "into", "through", "during", "before", "after",
                "above", "below", "between", "under", "over", "near",
                "behind", "beside", "against", "around", "among", "along",
                "across", "beyond", "within", "without", "upon", "towards"}

COPULAS = {"is", "are", "was", "were", "be", "been", "being", "am"}

AUXILIARIES = {"can", "could", "will", "would", "shall", "should", "may",
               "might", "must", "do", "does", "did", "have", "has", "had"}

NOUNS = {
    "cat", "dog", "house", "tree", "car", "book", "table", "chair", "door",
    "window", "river", "mountain", "city", "country", "school", "church",
    "bridge", "road", "garden", "forest", "ocean", "island", "valley",
    "castle", "village", "market", "station", "park", "field", "lake",
    "hill", "tower", "wall", "roof", "floor", "room", "hall", "street",
    "corner", "path", "world", "sky", "sun", "moon", "star", "cloud",
    "rain", "wind", "snow", "fire", "water", "stone", "gold", "silver",
    "iron", "wood", "glass", "paper", "cloth", "silk", "sand", "dust",
    "child", "man", "woman", "king", "queen", "prince", "knight", "lord",
    "soldier", "farmer", "teacher", "doctor", "artist", "writer", "singer",
    "brother", "sister", "mother", "father", "friend", "horse", "bird",
    "fish", "wolf", "bear", "lion", "eagle", "snake", "rabbit", "deer",
    "food", "bread", "milk", "wine", "meat", "fruit", "cake", "soup",
    "time", "day", "night", "year", "morning", "evening", "summer", "winter",
    "hand", "head", "heart", "eye", "face", "voice", "mind", "soul",
    "light", "shadow", "sound", "color", "shape", "edge", "surface",
}

ADJECTIVES = {
    "big", "small", "tall", "short", "long", "wide", "narrow", "deep",
    "high", "low", "fast", "slow", "hot", "cold", "warm", "cool", "dry",
    "wet", "hard", "soft", "loud", "quiet", "bright", "dark", "light",
    "heavy", "thick", "thin", "strong", "weak", "rich", "poor", "old",
    "young", "new", "clean", "dirty", "safe", "dangerous", "happy", "sad",
    "angry", "calm", "brave", "afraid", "kind", "cruel", "wise", "foolish",
    "true", "false", "real", "fake", "good", "bad", "right", "wrong",
    "full", "empty", "open", "closed", "free", "busy", "easy", "difficult",
    "simple", "complex", "clear", "vague", "sharp", "dull", "smooth",
    "rough", "sweet", "bitter", "sour", "fresh", "ripe", "raw", "wild",
    "tame", "rare", "common", "strange", "normal", "beautiful", "ugly",
    "plain", "fancy", "round", "flat", "straight", "curved", "alive",
    "dead", "awake", "asleep", "aware", "blind", "deaf", "mute", "sick",
}

VERBS = {
    "run", "walk", "go", "come", "see", "look", "hear", "listen", "speak",
    "talk", "say", "tell", "read", "write", "eat", "drink", "sleep", "wake",
    "work", "play", "make", "build", "break", "fix", "cut", "draw", "paint",
    "sing", "dance", "jump", "fly", "swim", "climb", "fall", "sit", "stand",
    "lie", "move", "stop", "start", "begin", "end", "open", "close", "turn",
    "push", "pull", "hold", "drop", "throw", "catch", "hit", "kick",
    "fight", "win", "lose", "find", "hide", "show", "teach", "learn",
    "think", "know", "feel", "want", "need", "try", "help", "give", "take",
    "bring", "send", "leave", "stay", "wait", "change", "grow", "die",
    "live", "love", "hate", "like", "fear", "trust", "believe", "hope",
    "remember", "forget", "understand", "explain", "agree", "refuse",
}


# ---------------------------------------------------------------------------
# Extraction from text
# ---------------------------------------------------------------------------

def _tokenize(text: str) -> list[str]:
    """Simple whitespace + punctuation tokenizer."""
    return re.findall(r"[a-zA-Z]+", text.lower())


def extract_grammar_pairs_from_text(text: str) -> dict[str, list[list[str]]]:
    """Extract grammatical bigram pairs from English text.

    Returns {category: [[word_a, word_b], ...]} where categories are
    determiner_noun, preposition_noun, copula_adjective, auxiliary_verb.
    """
    tokens = _tokenize(text)
    pairs: dict[str, list[list[str]]] = {
        "determiner_noun": [],
        "preposition_noun": [],
        "copula_adjective": [],
        "auxiliary_verb": [],
    }
    seen: dict[str, set[tuple[str, str]]] = {k: set() for k in pairs}

    for i in range(len(tokens) - 1):
        a, b = tokens[i], tokens[i + 1]

        if a in DETERMINERS and b in NOUNS:
            if (a, b) not in seen["determiner_noun"]:
                pairs["determiner_noun"].append([a, b])
                seen["determiner_noun"].add((a, b))

        if a in PREPOSITIONS and b in NOUNS:
            if (a, b) not in seen["preposition_noun"]:
                pairs["preposition_noun"].append([a, b])
                seen["preposition_noun"].add((a, b))

        if a in COPULAS and b in ADJECTIVES:
            if (a, b) not in seen["copula_adjective"]:
                pairs["copula_adjective"].append([a, b])
                seen["copula_adjective"].add((a, b))

        if a in AUXILIARIES and b in VERBS:
            if (a, b) not in seen["auxiliary_verb"]:
                pairs["auxiliary_verb"].append([a, b])
                seen["auxiliary_verb"].add((a, b))

    return pairs


def generate_grammar_pairs() -> dict[str, list[list[str]]]:
    """Generate grammar pairs from known word lists (no input text needed).

    Produces a curated cross-product of common bigrams.
    """
    pairs: dict[str, list[list[str]]] = {}

    # Determiners + nouns: pick common combos
    det_nouns: list[list[str]] = []
    common_dets = ["the", "a", "an", "this", "that", "some", "every", "each",
                   "my", "your", "no", "any"]
    common_nouns = sorted(NOUNS)[:40]
    seen: set[tuple[str, str]] = set()
    for det in common_dets:
        for noun in common_nouns:
            # Skip "a" before vowels, "an" before consonants
            if det == "a" and noun[0] in "aeiou":
                continue
            if det == "an" and noun[0] not in "aeiou":
                continue
            if (det, noun) not in seen:
                det_nouns.append([det, noun])
                seen.add((det, noun))
    pairs["determiner_noun"] = det_nouns

    # Prepositions + nouns
    prep_nouns: list[list[str]] = []
    common_preps = ["in", "on", "at", "to", "for", "with", "from", "by",
                    "of", "about", "into", "through", "near", "under", "over"]
    seen2: set[tuple[str, str]] = set()
    for prep in common_preps:
        for noun in common_nouns[:20]:
            if (prep, noun) not in seen2:
                prep_nouns.append([prep, noun])
                seen2.add((prep, noun))
    pairs["preposition_noun"] = prep_nouns

    # Copula + adjective
    cop_adj: list[list[str]] = []
    common_cops = ["is", "are", "was", "were"]
    common_adjs = sorted(ADJECTIVES)[:30]
    seen3: set[tuple[str, str]] = set()
    for cop in common_cops:
        for adj in common_adjs:
            if (cop, adj) not in seen3:
                cop_adj.append([cop, adj])
                seen3.add((cop, adj))
    pairs["copula_adjective"] = cop_adj

    # Auxiliary + verb
    aux_verb: list[list[str]] = []
    common_auxs = ["can", "could", "will", "would", "shall", "should",
                   "may", "might", "must"]
    common_verbs = sorted(VERBS)[:20]
    seen4: set[tuple[str, str]] = set()
    for aux in common_auxs:
        for verb in common_verbs:
            if (aux, verb) not in seen4:
                aux_verb.append([aux, verb])
                seen4.add((aux, verb))
    pairs["auxiliary_verb"] = aux_verb

    return pairs


def save_grammar_pairs(output_path: Path) -> dict[str, int]:
    """Generate and save English grammar pairs to JSON.

    Returns {category: num_pairs}.
    """
    pairs = generate_grammar_pairs()

    data: dict[str, Any] = {
        "source": "english_grammar",
        "description": "Syntactic bigram pairs grouped by grammatical category",
        "relations": {
            category: {"pairs": pair_list}
            for category, pair_list in pairs.items()
        },
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    return {cat: len(pl) for cat, pl in pairs.items()}
