#!/usr/bin/env python3
"""Generate morphological (base_form, inflected_form) pairs using lemminflect.

Covers 10 relation types: plural, gerund, past_tense, third_person,
comparative, superlative, agent_noun, nominalization, adverb, negation_prefix.

Uses lemminflect for accurate inflection (handles irregulars), with inflect
as a fallback.

Output: data/morphological_relations.json

Usage:
    pip install lemminflect
    python3 scripts/fetch_morphological.py
    python3 scripts/fetch_morphological.py --output data/morph.json --max-per-relation 200
"""

import argparse
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Library imports with graceful fallback
# ---------------------------------------------------------------------------

BACKEND = None

try:
    import lemminflect  # noqa: F401
    BACKEND = "lemminflect"
except Exception:
    pass

if BACKEND is None:
    try:
        import inflect as _inflect_mod  # noqa: F401
        BACKEND = "inflect"
    except ImportError:
        print(
            "Install lemminflect (preferred) or inflect:\n"
            "  pip install lemminflect\n"
            "  pip install inflect",
            file=sys.stderr,
        )
        sys.exit(1)

# ---------------------------------------------------------------------------
# Common word lists (top ~500 each, sourced from frequency corpora)
# ---------------------------------------------------------------------------

COMMON_NOUNS = [
    "time", "year", "people", "way", "day", "man", "woman", "child", "world",
    "life", "hand", "part", "place", "case", "week", "company", "system",
    "program", "question", "work", "government", "number", "night", "point",
    "home", "water", "room", "mother", "area", "money", "story", "fact",
    "month", "lot", "right", "study", "book", "eye", "job", "word", "business",
    "issue", "side", "kind", "head", "house", "service", "friend", "father",
    "power", "hour", "game", "line", "end", "member", "law", "car", "city",
    "community", "name", "president", "team", "minute", "idea", "body",
    "information", "back", "parent", "face", "thing", "student", "farm",
    "group", "country", "problem", "plan", "family", "teacher", "door",
    "reason", "moment", "person", "girl", "boy", "table", "market", "class",
    "bed", "war", "history", "party", "result", "change", "morning", "road",
    "report", "form", "decision", "food", "ground", "office", "role", "view",
    "level", "heart", "son", "daughter", "art", "experience", "death",
    "effect", "use", "field", "development", "process", "state", "action",
    "model", "force", "education", "foot", "age", "policy", "music",
    "interest", "bank", "period", "building", "film", "rate", "dog", "cat",
    "bird", "fish", "tree", "flower", "river", "mountain", "stone", "star",
    "sun", "moon", "wind", "rain", "snow", "fire", "cup", "box", "ball",
    "wall", "window", "step", "test", "picture", "voice", "answer", "letter",
    "price", "land", "note", "paper", "page", "song", "baby", "type",
    "center", "church", "event", "term", "color", "figure", "key", "surface",
    "record", "king", "doctor", "village", "animal", "plant", "machine",
    "ship", "horse", "brother", "sister", "army", "bridge", "garden",
    "island", "kitchen", "seat", "shape", "sign", "tool", "town", "wave",
    "chair", "corner", "coat", "dress", "hat", "milk", "oil", "meat",
    "bread", "fruit", "egg", "glass", "bag", "bottle", "ring", "clock",
    "knife", "bridge", "radio", "camera", "dream", "nose", "lip", "finger",
    "leg", "shoulder", "bone", "brain", "tooth", "nail", "wing", "tail",
    "neck", "skin", "blood", "stomach", "knee", "ear", "tongue", "muscle",
    "lung", "throat", "hair", "cell", "circle", "square", "triangle",
    "wheel", "engine", "screen", "wire", "battery", "switch", "pipe",
    "chain", "gate", "fence", "pole", "rope", "thread", "needle", "button",
    "pocket", "shelf", "drawer", "blanket", "pillow", "towel", "mirror",
    "brush", "basket", "bucket", "ladder", "cage", "tent", "flag", "stamp",
    "ticket", "coin", "medal", "badge", "label", "card", "map", "chart",
    "graph", "list", "menu", "recipe", "lesson", "speech", "poem", "novel",
    "chapter", "verse", "phrase", "sentence", "paragraph", "column", "row",
    "border", "edge", "tip", "peak", "root", "branch", "leaf", "seed",
    "stem", "bark", "berry", "crop", "harvest", "soil", "dust", "mud",
    "sand", "rock", "cliff", "cave", "valley", "hill", "lake", "ocean",
    "beach", "coast", "harbor", "canal", "dam", "pond", "spring", "storm",
    "cloud", "thunder", "lightning", "fog", "frost", "ice", "flame",
    "smoke", "ash", "coal", "metal", "iron", "steel", "copper", "gold",
    "silver", "diamond", "crystal", "brick", "cement", "paint", "cloth",
    "silk", "cotton", "wool", "leather", "rubber", "plastic", "poison",
    "drug", "pill", "wound", "scar", "fever", "cough", "disease", "virus",
    "cancer", "treatment", "surgery", "nurse", "patient", "hospital",
    "prison", "trial", "judge", "witness", "victim", "weapon", "bullet",
    "bomb", "sword", "shield", "helmet", "uniform", "soldier", "captain",
    "crowd", "passenger", "guest", "customer", "driver", "pilot", "sailor",
    "hunter", "farmer", "baker", "painter", "actor", "singer", "dancer",
    "writer", "poet", "hero", "villain", "monster", "giant", "angel",
    "ghost", "shadow", "spirit", "soul", "brain", "thought", "memory",
    "emotion", "anger", "fear", "joy", "pride", "shame", "guilt", "love",
    "hate", "trust", "doubt", "hope", "wish", "gift", "prize", "reward",
    "profit", "debt", "tax", "wage", "salary", "budget", "loan", "rent",
    "fee", "cost", "bill", "check", "trade", "deal", "contract", "sale",
    "stock", "supply", "demand", "growth", "decline", "crisis", "disaster",
    "accident", "mistake", "failure", "success", "victory", "defeat",
    "struggle", "challenge", "opportunity", "risk", "threat", "danger",
    "safety", "peace", "freedom", "justice", "truth", "lie", "secret",
    "mystery", "puzzle", "trick", "joke", "game", "sport", "race", "match",
    "score", "goal", "rule", "league", "champion", "coach", "fan",
    "audience", "concert", "festival", "ceremony", "tradition", "culture",
    "religion", "temple", "prayer", "god", "heaven", "hell", "myth",
    "legend", "tale", "adventure", "journey", "trip", "flight", "voyage",
    "mission", "task", "project", "skill", "talent", "habit", "pattern",
    "trend", "fashion", "style", "design", "structure", "frame", "layer",
    "block", "section", "zone", "region", "district", "neighborhood",
    "street", "path", "trail", "track", "route", "mile", "inch", "pound",
    "gallon", "degree",
]

COMMON_VERBS = [
    "be", "have", "do", "say", "go", "get", "make", "know", "think", "take",
    "see", "come", "want", "look", "use", "find", "give", "tell", "work",
    "call", "try", "ask", "need", "feel", "become", "leave", "put", "mean",
    "keep", "let", "begin", "seem", "help", "show", "hear", "play", "run",
    "move", "live", "believe", "hold", "bring", "happen", "write", "provide",
    "sit", "stand", "lose", "pay", "meet", "include", "continue", "set",
    "learn", "change", "lead", "understand", "watch", "follow", "stop",
    "create", "speak", "read", "allow", "add", "spend", "grow", "open",
    "walk", "win", "offer", "remember", "love", "consider", "appear", "buy",
    "wait", "serve", "die", "send", "expect", "build", "stay", "fall",
    "cut", "reach", "kill", "remain", "suggest", "raise", "pass", "sell",
    "require", "report", "decide", "pull", "develop", "break", "receive",
    "agree", "support", "hit", "produce", "eat", "cover", "catch", "draw",
    "choose", "cause", "point", "listen", "realize", "place", "enter",
    "carry", "talk", "share", "pick", "drop", "plan", "push", "close",
    "drive", "ride", "wear", "hang", "throw", "fight", "teach", "sing",
    "shake", "sleep", "dance", "fly", "swim", "climb", "dig", "fix",
    "smile", "cry", "laugh", "jump", "kick", "bend", "bite", "blow",
    "burn", "cook", "cross", "fill", "grab", "hide", "hunt", "kiss",
    "lift", "lock", "mix", "pour", "pray", "press", "print", "protect",
    "prove", "race", "roll", "rush", "save", "search", "shoot", "sign",
    "skip", "slide", "slip", "sort", "split", "spread", "steal", "stick",
    "stretch", "strike", "study", "suffer", "supply", "taste", "tie",
    "touch", "trade", "train", "trap", "treat", "trick", "trust", "turn",
    "type", "visit", "wake", "warn", "wash", "watch", "wave", "wish",
    "wonder", "worry", "wrap", "yell", "check", "clean", "collect",
    "compare", "complain", "connect", "control", "copy", "count", "damage",
    "deliver", "demand", "depend", "describe", "design", "destroy",
    "discover", "discuss", "divide", "doubt", "drag", "dream", "dress",
    "drink", "earn", "enjoy", "escape", "examine", "exist", "explain",
    "explore", "express", "extend", "fail", "fear", "feed", "finish",
    "float", "flow", "fold", "force", "form", "freeze", "gain", "gather",
    "guess", "handle", "hate", "heal", "hire", "hope", "hurry", "imagine",
    "improve", "increase", "inform", "introduce", "invent", "invest",
    "invite", "involve", "join", "judge", "land", "last", "launch", "lean",
    "lend", "limit", "link", "load", "manage", "mark", "measure", "mention",
    "miss", "name", "notice", "observe", "obtain", "operate", "order",
    "organize", "paint", "perform", "permit", "plant", "possess", "practice",
    "predict", "prefer", "prepare", "present", "prevent", "promise",
    "publish", "punish", "purchase", "recover", "reduce", "reflect",
    "refuse", "relate", "release", "rely", "remove", "repair", "repeat",
    "replace", "request", "rescue", "resolve", "respond", "rest", "restore",
    "return", "reveal", "ring", "rise", "risk", "rob", "rule", "satisfy",
    "scare", "score", "select", "settle", "shape", "shout", "shut",
    "signal", "snap", "solve", "sound", "spare", "spin", "spot", "squeeze",
    "stare", "step", "store", "struggle", "succeed", "suggest", "surprise",
    "surround", "survive", "suspect", "swing", "tend", "test", "thank",
    "threaten", "toss", "track", "transfer", "transform", "travel",
    "twist", "unite", "value", "vary", "view", "vote", "wander", "weigh",
    "whisper", "wipe", "witness", "attack", "balance", "ban", "bark",
    "bat", "beg", "blame", "bless", "block", "boil", "bomb", "borrow",
    "bounce", "bow", "brake", "breathe", "broadcast", "brush", "bury",
    "calculate", "camp", "care", "carve", "celebrate", "charge", "chase",
    "cheat", "cheer", "chip", "choke", "clap", "click", "coach", "combine",
    "comfort", "command", "communicate", "compete", "compose", "concern",
    "concentrate", "conclude", "conduct", "confess", "confirm", "confuse",
    "consider", "construct", "consult", "consume", "contain", "convince",
    "crash", "crawl", "cure", "cycle",
]

COMMON_ADJECTIVES = [
    "good", "new", "first", "last", "long", "great", "little", "own", "other",
    "old", "right", "big", "high", "different", "small", "large", "next",
    "early", "young", "important", "few", "public", "bad", "same", "able",
    "free", "sure", "real", "full", "special", "easy", "clear", "close",
    "recent", "possible", "common", "strong", "whole", "simple", "past",
    "hard", "low", "late", "general", "specific", "open", "short", "true",
    "natural", "similar", "final", "fast", "main", "wide", "local", "dark",
    "various", "entire", "nice", "poor", "happy", "best", "serious",
    "ready", "single", "deep", "available", "likely", "hot", "cold",
    "wrong", "major", "heavy", "fresh", "blue", "warm", "red", "green",
    "white", "black", "brown", "yellow", "gray", "pink", "rich", "huge",
    "thin", "thick", "tall", "quiet", "soft", "loud", "bright", "sharp",
    "rough", "smooth", "flat", "round", "empty", "dry", "wet", "clean",
    "dirty", "safe", "dangerous", "angry", "calm", "brave", "proud",
    "fair", "strange", "wild", "blind", "deaf", "dumb", "sick", "tired",
    "weak", "fit", "narrow", "gentle", "rare", "tough", "tiny", "vast",
    "neat", "loose", "tight", "raw", "ripe", "pure", "plain", "steep",
    "mild", "harsh", "tender", "hollow", "solid", "bitter", "sweet", "sour",
    "crisp", "dense", "pale", "dull", "vivid", "fierce", "eager", "keen",
    "lazy", "modest", "noble", "odd", "polite", "rude", "shy", "clever",
    "cruel", "elegant", "fancy", "fierce", "fond", "fragile", "grim",
    "humble", "intense", "jealous", "lame", "lean", "loyal", "mature",
    "merry", "nasty", "nervous", "obedient", "patient", "pleasant",
    "precise", "prompt", "rapid", "rigid", "robust", "sacred", "selfish",
    "severe", "shallow", "silly", "sincere", "slender", "slim", "sly",
    "solemn", "splendid", "stiff", "subtle", "superb", "swift", "vague",
    "violent", "vivid", "wicked", "wise", "worthy", "absent", "abstract",
    "abundant", "accurate", "adequate", "anxious", "apparent", "automatic",
    "awful", "bare", "blank", "blunt", "bold", "brief", "broad", "capable",
    "casual", "cautious", "cheerful", "civil", "clumsy", "coarse",
    "compact", "complex", "conscious", "constant", "content", "convenient",
    "cool", "costly", "cozy", "crazy", "curious", "cute", "decent",
    "delicate", "desperate", "dim", "distinct", "double", "dreadful",
    "dull", "eager", "elaborate", "elegant", "enormous", "essential",
    "evident", "evil", "exact", "excellent", "excessive", "exotic",
    "explicit", "extreme", "faint", "faithful", "false", "familiar",
    "fatal", "favorable", "flexible", "fluent", "foolish", "formal",
    "fortunate", "frequent", "friendly", "frightened", "fruitful", "funny",
    "genuine", "glad", "global", "golden", "graceful", "grand", "grateful",
    "greedy", "guilty", "handsome", "helpful", "hopeful", "horrible",
    "hostile", "hungry", "ideal", "illegal", "immediate", "immense",
    "implicit", "impressive", "incredible", "independent", "indirect",
    "inferior", "infinite", "informal", "inner", "innocent", "intimate",
    "invisible", "junior", "just", "legal", "liberal", "literal", "logical",
    "lonely", "lovely", "lucky", "magnificent", "massive", "maximum",
    "meaningful", "mental", "mere", "minimum", "minor", "miserable",
    "mobile", "moderate", "mutual", "naked", "native", "neat", "neutral",
    "normal", "notorious", "obvious", "outer", "overall", "partial",
    "passive", "peculiar", "permanent", "personal", "physical", "plastic",
    "positive", "powerful", "practical", "precious", "pregnant", "previous",
    "primary", "primitive", "principal", "private", "productive",
    "professional", "profound", "progressive", "prominent", "proper",
    "proud", "purple", "radical", "random", "rational", "reasonable",
    "regular", "relative", "relevant", "reluctant", "remarkable",
    "remote", "responsible", "romantic", "royal", "rude", "rural", "sad",
    "secure", "senior", "sensitive", "separate", "sharp", "sheer", "silent",
    "smooth", "sober", "solar", "sole", "sophisticated", "spare",
    "spiritual", "stable", "standard", "static", "steady", "steep",
    "sticky", "straight", "strict", "stupid", "subsequent", "substantial",
    "sufficient", "suitable", "superior", "supreme", "suspicious",
    "sympathetic", "temporary", "terrible", "thorough", "tidy", "tight",
    "total", "tremendous", "tropical", "ugly", "ultimate", "unable",
    "uncertain", "unique", "universal", "unlikely", "unusual", "upper",
    "upset", "urban", "urgent", "useful", "usual", "valid", "valuable",
    "visible", "visual", "vital", "vivid", "voluntary", "vulnerable",
    "weird", "welcome", "wooden", "worthy",
]

# Words known to take un- prefix
UN_WORDS = [
    "happy", "fair", "kind", "able", "aware", "certain", "clear", "common",
    "comfortable", "conscious", "controlled", "convinced", "cool", "covered",
    "decided", "defined", "done", "dressed", "easy", "equal", "even",
    "expected", "familiar", "fashionable", "fit", "fold", "fortunate",
    "friendly", "healthy", "helpful", "impressed", "just", "known", "lawful",
    "like", "likely", "limited", "lock", "lucky", "natural", "necessary",
    "official", "pack", "paid", "pleasant", "popular", "predictable",
    "prepared", "productive", "professional", "profitable", "real",
    "reasonable", "related", "reliable", "resolved", "rest", "safe",
    "satisfactory", "settled", "skilled", "stable", "steady", "successful",
    "sure", "surprised", "sympathetic", "tidy", "tie", "true", "typical",
    "usual", "well", "willing", "wise", "worthy", "wrap",
]


# ---------------------------------------------------------------------------
# Inflection engines
# ---------------------------------------------------------------------------

def _inflect_lemminflect():
    """Use lemminflect for all inflections."""
    from lemminflect import getInflection  # noqa: F811

    def _get(word, tag):
        """Get inflection for word with given Penn tag."""
        result = getInflection(word, tag=tag, inflect_oov=True)
        if result:
            return result[0]
        return None

    def plural(word):
        return _get(word, "NNS")

    def gerund(word):
        return _get(word, "VBG")

    def past_tense(word):
        return _get(word, "VBD")

    def third_person(word):
        return _get(word, "VBZ")

    def comparative(word):
        return _get(word, "JJR")

    def superlative(word):
        return _get(word, "JJS")

    return {
        "plural": plural,
        "gerund": gerund,
        "past_tense": past_tense,
        "third_person": third_person,
        "comparative": comparative,
        "superlative": superlative,
    }


def _inflect_inflect():
    """Fallback using inflect library."""
    import inflect as _inf
    eng = _inf.engine()

    def plural(word):
        result = eng.plural_noun(word)
        return result if result and result != word else None

    def gerund(word):
        # inflect doesn't do verb forms well; use basic rules
        if word.endswith("e") and not word.endswith("ee"):
            return word[:-1] + "ing"
        if (len(word) >= 3 and word[-1] not in "aeiouywx"
                and word[-2] in "aeiou" and word[-3] not in "aeiou"):
            return word + word[-1] + "ing"
        return word + "ing"

    def past_tense(word):
        if word.endswith("e"):
            return word + "d"
        if word.endswith("y") and word[-2] not in "aeiou":
            return word[:-1] + "ied"
        return word + "ed"

    def third_person(word):
        if word.endswith(("s", "sh", "ch", "x", "z")):
            return word + "es"
        if word.endswith("y") and word[-2] not in "aeiou":
            return word[:-1] + "ies"
        return word + "s"

    def comparative(word):
        if word.endswith("e"):
            return word + "r"
        if word.endswith("y"):
            return word[:-1] + "ier"
        if (len(word) >= 3 and word[-1] not in "aeiouywx"
                and word[-2] in "aeiou" and word[-3] not in "aeiou"):
            return word + word[-1] + "er"
        return word + "er"

    def superlative(word):
        if word.endswith("e"):
            return word + "st"
        if word.endswith("y"):
            return word[:-1] + "iest"
        if (len(word) >= 3 and word[-1] not in "aeiouywx"
                and word[-2] in "aeiou" and word[-3] not in "aeiou"):
            return word + word[-1] + "est"
        return word + "est"

    return {
        "plural": plural,
        "gerund": gerund,
        "past_tense": past_tense,
        "third_person": third_person,
        "comparative": comparative,
        "superlative": superlative,
    }


# ---------------------------------------------------------------------------
# Pair generators
# ---------------------------------------------------------------------------

def generate_inflection_pairs(relation: str, words: list, inflect_fn,
                              max_pairs: int) -> list:
    """Generate (base, inflected) pairs, filtering out identity/None."""
    pairs = []
    seen = set()
    for word in words:
        if len(pairs) >= max_pairs:
            break
        result = inflect_fn(word)
        if result and result != word and result.isalpha() and (word, result) not in seen:
            pairs.append([word, result])
            seen.add((word, result))
    return pairs


def generate_agent_nouns(verbs: list, max_pairs: int) -> list:
    """Generate verb -> agent noun (-er/-or) pairs.

    Uses a simple suffix heuristic and validates against known patterns.
    """
    pairs = []
    seen = set()
    for verb in verbs:
        if len(pairs) >= max_pairs:
            break
        # Common -er formations
        candidates = []
        if verb.endswith("e"):
            candidates.append(verb + "r")
        elif verb.endswith("y"):
            candidates.append(verb[:-1] + "ier")
        else:
            candidates.append(verb + "er")
        # Double final consonant for short verbs
        if (len(verb) >= 3 and verb[-1] not in "aeiouywx"
                and verb[-2] in "aeiou" and verb[-3] not in "aeiou"):
            candidates.append(verb + verb[-1] + "er")

        for agent in candidates:
            if agent.isalpha() and agent != verb and (verb, agent) not in seen:
                pairs.append([verb, agent])
                seen.add((verb, agent))
                break
    return pairs


def generate_nominalizations(adjectives: list, max_pairs: int) -> list:
    """Generate adjective -> -ness noun pairs."""
    pairs = []
    for adj in adjectives:
        if len(pairs) >= max_pairs:
            break
        if adj.endswith("y"):
            noun = adj[:-1] + "iness"
        else:
            noun = adj + "ness"
        if noun != adj and noun.isalpha():
            pairs.append([adj, noun])
    return pairs


def generate_adverbs(adjectives: list, max_pairs: int) -> list:
    """Generate adjective -> -ly adverb pairs."""
    pairs = []
    # Skip adjectives that already end in -ly or don't form natural adverbs
    skip = {"early", "only", "likely", "lonely", "lovely", "friendly", "ugly",
            "silly", "holy", "jolly"}
    for adj in adjectives:
        if len(pairs) >= max_pairs:
            break
        if adj in skip:
            continue
        if adj.endswith("le") and not adj.endswith("ble"):
            adv = adj[:-1] + "y"
        elif adj.endswith("y"):
            adv = adj[:-1] + "ily"
        elif adj.endswith("ic"):
            adv = adj + "ally"
        elif adj.endswith("ll"):
            adv = adj + "y"
        elif adj.endswith("ue"):
            adv = adj[:-1] + "ly"
        else:
            adv = adj + "ly"
        if adv != adj and adv.isalpha():
            pairs.append([adj, adv])
    return pairs


def generate_negation_prefix(words: list, max_pairs: int) -> list:
    """Generate word -> un+word pairs from known un- words."""
    pairs = []
    word_set = set(words)
    for word in words:
        if len(pairs) >= max_pairs:
            break
        if word in UN_WORDS or word in word_set:
            negated = "un" + word
            if negated.isalpha():
                pairs.append([word, negated])
    # Only keep words from the UN_WORDS list to ensure validity
    pairs = [[w, n] for w, n in pairs if w in UN_WORDS]
    return pairs[:max_pairs]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate morphological relation pairs using lemminflect."
    )
    parser.add_argument(
        "--output", type=str,
        default="data/morphological_relations.json",
        help="Output JSON file path (default: data/morphological_relations.json)",
    )
    parser.add_argument(
        "--max-per-relation", type=int, default=500,
        help="Maximum pairs per relation type (default: 500)",
    )
    args = parser.parse_args()

    max_n = args.max_per_relation

    print(f"Backend: {BACKEND}")
    if BACKEND == "lemminflect":
        fns = _inflect_lemminflect()
    else:
        fns = _inflect_inflect()

    relations = {}

    # 1. Plural
    print("Generating plural pairs...")
    relations["plural"] = {
        "description": "Singular to plural noun form",
        "pairs": generate_inflection_pairs(
            "plural", COMMON_NOUNS, fns["plural"], max_n
        ),
    }

    # 2. Gerund
    print("Generating gerund pairs...")
    relations["gerund"] = {
        "description": "Base verb to -ing form",
        "pairs": generate_inflection_pairs(
            "gerund", COMMON_VERBS, fns["gerund"], max_n
        ),
    }

    # 3. Past tense
    print("Generating past tense pairs...")
    relations["past_tense"] = {
        "description": "Base verb to past tense form (handles irregulars)",
        "pairs": generate_inflection_pairs(
            "past_tense", COMMON_VERBS, fns["past_tense"], max_n
        ),
    }

    # 4. Third person
    print("Generating third person pairs...")
    relations["third_person"] = {
        "description": "Base verb to third person singular present",
        "pairs": generate_inflection_pairs(
            "third_person", COMMON_VERBS, fns["third_person"], max_n
        ),
    }

    # 5. Comparative
    print("Generating comparative pairs...")
    relations["comparative"] = {
        "description": "Adjective to comparative form (-er or more irregular)",
        "pairs": generate_inflection_pairs(
            "comparative", COMMON_ADJECTIVES, fns["comparative"], max_n
        ),
    }

    # 6. Superlative
    print("Generating superlative pairs...")
    relations["superlative"] = {
        "description": "Adjective to superlative form (-est or most irregular)",
        "pairs": generate_inflection_pairs(
            "superlative", COMMON_ADJECTIVES, fns["superlative"], max_n
        ),
    }

    # 7. Agent noun
    print("Generating agent noun pairs...")
    relations["agent_noun"] = {
        "description": "Verb to agent noun (-er form: teach -> teacher)",
        "pairs": generate_agent_nouns(COMMON_VERBS, max_n),
    }

    # 8. Nominalization
    print("Generating nominalization pairs...")
    relations["nominalization"] = {
        "description": "Adjective to -ness noun form (happy -> happiness)",
        "pairs": generate_nominalizations(COMMON_ADJECTIVES, max_n),
    }

    # 9. Adverb
    print("Generating adverb pairs...")
    relations["adverb"] = {
        "description": "Adjective to -ly adverb form (quick -> quickly)",
        "pairs": generate_adverbs(COMMON_ADJECTIVES, max_n),
    }

    # 10. Negation prefix
    print("Generating negation prefix pairs...")
    all_words = list(set(COMMON_ADJECTIVES + COMMON_VERBS + COMMON_NOUNS))
    relations["negation_prefix"] = {
        "description": "Word to un- negated form (happy -> unhappy)",
        "pairs": generate_negation_prefix(all_words, max_n),
    }

    output = {
        "source": BACKEND,
        "relations": relations,
    }

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary
    print(f"\nOutput: {out_path}")
    print(f"Backend: {BACKEND}")
    print("-" * 50)
    total = 0
    for name, rel in relations.items():
        n = len(rel["pairs"])
        total += n
        sample = rel["pairs"][:3] if rel["pairs"] else []
        sample_str = ", ".join(f"{a}->{b}" for a, b in sample)
        print(f"  {name:20s}: {n:4d} pairs  (e.g. {sample_str})")
    print("-" * 50)
    print(f"  {'TOTAL':20s}: {total:4d} pairs")


if __name__ == "__main__":
    main()
