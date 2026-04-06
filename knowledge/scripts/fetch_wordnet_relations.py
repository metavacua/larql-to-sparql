#!/usr/bin/env python3
"""Build a lexical relation database from WordNet for L0-13 feature labeling.

Extracts synonym, hypernym, antonym, meronym, derivational relations
from NLTK's WordNet interface. Generates morphological pairs using
NLTK morphy (handles irregulars) and inflect (proper plurals).

Output: data/wordnet_relations.json

Usage:
    pip install nltk inflect
    python3 scripts/fetch_wordnet_relations.py

First run will download WordNet data (~30MB).
"""

import json
import sys
from pathlib import Path

try:
    import nltk
    from nltk.corpus import wordnet as wn
except ImportError:
    print("Install nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)

try:
    import inflect
except ImportError:
    print("Install inflect: pip install inflect", file=sys.stderr)
    sys.exit(1)


def ensure_data():
    """Download required NLTK data if not present."""
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            print(f"Downloading {resource}...")
            nltk.download(resource, quiet=True)


def extract_synonyms(limit: int = 5000) -> list:
    """Extract synonym pairs from WordNet synsets."""
    pairs = []
    seen = set()
    for synset in wn.all_synsets():
        lemmas = [l.name().replace("_", " ").lower() for l in synset.lemmas()
                  if l.name().isalpha() and len(l.name()) >= 3]
        for i in range(len(lemmas)):
            for j in range(i + 1, len(lemmas)):
                a, b = lemmas[i], lemmas[j]
                if a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
                    seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_hypernyms(limit: int = 3000) -> list:
    """Extract hypernym (is-a) pairs: child_concept → parent_concept."""
    pairs = []
    seen = set()
    for synset in wn.all_synsets("n"):
        word = synset.lemmas()[0].name().replace("_", " ").lower()
        if not word.isalpha() or len(word) < 3:
            continue
        for hyper in synset.hypernyms():
            parent = hyper.lemmas()[0].name().replace("_", " ").lower()
            if parent.isalpha() and len(parent) >= 3 and word != parent and (word, parent) not in seen:
                pairs.append([word, parent])
                seen.add((word, parent))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_antonyms(limit: int = 2000) -> list:
    """Extract antonym pairs from WordNet."""
    pairs = []
    seen = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            for ant in lemma.antonyms():
                a = lemma.name().replace("_", " ").lower()
                b = ant.name().replace("_", " ").lower()
                if a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3 and a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
                    seen.add((b, a))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_meronyms(limit: int = 2000) -> list:
    """Extract meronym (part-of) pairs."""
    pairs = []
    seen = set()
    for synset in wn.all_synsets("n"):
        word = synset.lemmas()[0].name().replace("_", " ").lower()
        if not word.isalpha() or len(word) < 3:
            continue
        for mero_fn in [synset.part_meronyms, synset.member_meronyms, synset.substance_meronyms]:
            for mero in mero_fn():
                part = mero.lemmas()[0].name().replace("_", " ").lower()
                if part.isalpha() and len(part) >= 3 and part != word and (part, word) not in seen:
                    pairs.append([part, word])
                    seen.add((part, word))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_derivations(limit: int = 5000) -> list:
    """Extract derivationally related forms (e.g., happy → happiness)."""
    pairs = []
    seen = set()
    for synset in wn.all_synsets():
        for lemma in synset.lemmas():
            for related in lemma.derivationally_related_forms():
                a = lemma.name().replace("_", " ").lower()
                b = related.name().replace("_", " ").lower()
                if a.isalpha() and b.isalpha() and len(a) >= 3 and len(b) >= 3 and a != b and (a, b) not in seen:
                    pairs.append([a, b])
                    seen.add((a, b))
        if len(pairs) >= limit:
            break
    return pairs[:limit]


def extract_morphological_pairs() -> dict:
    """Extract morphological pairs using NLTK morphy + inflect.

    morphy handles irregular forms (ran→run, went→go, mice→mouse).
    inflect handles plural generation with irregulars.
    No hardcoded tables.
    """
    p = inflect.engine()
    morph = {}

    # Collect base words from WordNet
    verbs = set()
    nouns = set()
    adjectives = set()

    for synset in wn.all_synsets("v"):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name.isalpha() and 3 <= len(name) <= 10:
                verbs.add(name)
            if len(verbs) >= 200:
                break
        if len(verbs) >= 200:
            break

    for synset in wn.all_synsets("n"):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name.isalpha() and 3 <= len(name) <= 10:
                nouns.add(name)
            if len(nouns) >= 200:
                break
        if len(nouns) >= 200:
            break

    for synset in wn.all_synsets("a"):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name.isalpha() and 3 <= len(name) <= 10:
                adjectives.add(name)
            if len(adjectives) >= 100:
                break
        if len(adjectives) >= 100:
            break

    # Noun → plural using inflect (handles irregulars)
    plural_pairs = []
    for noun in sorted(nouns):
        plural = p.plural(noun)
        if plural and plural != noun and plural.isalpha():
            plural_pairs.append([noun, plural])
    morph["plural"] = plural_pairs

    # Verb inflections using WordNet morphy reverse lookup
    # For each verb, generate common inflected forms and check via morphy
    gerund_pairs = []
    past_tense_pairs = []
    third_person_pairs = []

    for verb in sorted(verbs):
        # Try common suffixed forms and validate with morphy
        candidates_ing = [verb + "ing", verb + "ning", verb + "ting",
                          verb + "ding", verb + "bing", verb + "ping",
                          verb[:-1] + "ing" if verb.endswith("e") else None,
                          verb[:-2] + "ying" if verb.endswith("ie") else None]
        for form in candidates_ing:
            if form and wn.morphy(form, wn.VERB) == verb:
                gerund_pairs.append([verb, form])
                break

        candidates_ed = [verb + "ed", verb + "d", verb + "ned", verb + "ted",
                         verb + "ded", verb + "bed", verb + "ped",
                         verb[:-1] + "ied" if verb.endswith("y") else None]
        for form in candidates_ed:
            if form and wn.morphy(form, wn.VERB) == verb:
                past_tense_pairs.append([verb, form])
                break

        candidates_s = [verb + "s", verb + "es",
                        verb[:-1] + "ies" if verb.endswith("y") else None]
        for form in candidates_s:
            if form and wn.morphy(form, wn.VERB) == verb:
                third_person_pairs.append([verb, form])
                break

    morph["gerund"] = gerund_pairs
    morph["past_tense"] = past_tense_pairs
    morph["third_person"] = third_person_pairs

    # Adjective comparatives/superlatives via morphy
    comparative_pairs = []
    superlative_pairs = []
    for adj in sorted(adjectives):
        candidates_er = [adj + "er", adj + "r",
                         adj[:-1] + "ier" if adj.endswith("y") else None,
                         adj + adj[-1] + "er" if len(adj) >= 3 else None]
        for form in candidates_er:
            if form and wn.morphy(form, wn.ADJ) == adj:
                comparative_pairs.append([adj, form])
                break

        candidates_est = [adj + "est", adj + "st",
                          adj[:-1] + "iest" if adj.endswith("y") else None,
                          adj + adj[-1] + "est" if len(adj) >= 3 else None]
        for form in candidates_est:
            if form and wn.morphy(form, wn.ADJ) == adj:
                superlative_pairs.append([adj, form])
                break

    morph["comparative"] = comparative_pairs
    morph["superlative"] = superlative_pairs

    return morph


def main() -> None:
    """Extract WordNet relations and save to data/wordnet_relations.json."""
    ensure_data()

    print("Extracting WordNet relations...")

    relations = {}

    print("  Synonyms...", end=" ", flush=True)
    synonyms = extract_synonyms()
    relations["synonym"] = {"pairs": synonyms}
    print(f"{len(synonyms)} pairs")

    print("  Hypernyms (is-a)...", end=" ", flush=True)
    hypernyms = extract_hypernyms()
    relations["hypernym"] = {"pairs": hypernyms}
    print(f"{len(hypernyms)} pairs")

    print("  Antonyms...", end=" ", flush=True)
    antonyms = extract_antonyms()
    relations["antonym"] = {"pairs": antonyms}
    print(f"{len(antonyms)} pairs")

    print("  Meronyms (part-of)...", end=" ", flush=True)
    meronyms = extract_meronyms()
    relations["meronym"] = {"pairs": meronyms}
    print(f"{len(meronyms)} pairs")

    print("  Derivations...", end=" ", flush=True)
    derivations = extract_derivations()
    relations["derivation"] = {"pairs": derivations}
    print(f"{len(derivations)} pairs")

    print("  Morphological pairs...", end=" ", flush=True)
    morph = extract_morphological_pairs()
    total_morph = 0
    for rel_type, pairs in morph.items():
        relations[rel_type] = {"pairs": pairs}
        total_morph += len(pairs)
    print(f"{total_morph} pairs across {len(morph)} types")

    # Save
    output_dir = Path(__file__).parent.parent / "data"
    output_dir.mkdir(exist_ok=True)

    output_path = output_dir / "wordnet_relations.json"
    with open(output_path, "w") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)

    total = sum(len(v["pairs"]) for v in relations.values())
    print(f"\nSaved {len(relations)} relation types, {total} total pairs to {output_path}")

    # Summary
    print("\nRelation types:")
    for name, data in relations.items():
        n = len(data["pairs"])
        examples = ", ".join(f"{s}→{o}" for s, o in data["pairs"][:3])
        print(f"  {name:<20s} {n:5d} pairs  [{examples}]")


if __name__ == "__main__":
    main()
