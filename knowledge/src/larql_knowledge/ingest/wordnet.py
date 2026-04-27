"""WordNet relation extraction via NLTK."""

import json
from pathlib import Path


def ingest_wordnet(output_path: Path, limits: dict | None = None) -> dict:
    """Extract WordNet relations and save to JSON.

    Args:
        output_path: Path to write wordnet_relations.json
        limits: Optional dict of {relation: max_pairs}

    Returns:
        Dict of {relation: num_pairs}
    """
    try:
        import nltk
        from nltk.corpus import wordnet as wn
    except ImportError:
        print("Install nltk: pip install nltk")
        return {}

    # Ensure data
    for resource in ["wordnet", "omw-1.4"]:
        try:
            nltk.data.find(f"corpora/{resource}")
        except LookupError:
            nltk.download(resource, quiet=True)

    if limits is None:
        limits = {
            "synonym": 5000,
            "hypernym": 3000,
            "antonym": 2000,
            "meronym": 2000,
            "derivation": 5000,
        }

    relations = {}

    # Synonyms
    if "synonym" in limits:
        pairs = _extract_synonyms(wn, limits["synonym"])
        relations["synonym"] = {"pairs": pairs}
        print(f"  synonym: {len(pairs)} pairs")

    # Hypernyms
    if "hypernym" in limits:
        pairs = _extract_hypernyms(wn, limits["hypernym"])
        relations["hypernym"] = {"pairs": pairs}
        print(f"  hypernym: {len(pairs)} pairs")

    # Antonyms
    if "antonym" in limits:
        pairs = _extract_antonyms(wn, limits["antonym"])
        relations["antonym"] = {"pairs": pairs}
        print(f"  antonym: {len(pairs)} pairs")

    # Meronyms
    if "meronym" in limits:
        pairs = _extract_meronyms(wn, limits["meronym"])
        relations["meronym"] = {"pairs": pairs}
        print(f"  meronym: {len(pairs)} pairs")

    # Derivations
    if "derivation" in limits:
        pairs = _extract_derivations(wn, limits["derivation"])
        relations["derivation"] = {"pairs": pairs}
        print(f"  derivation: {len(pairs)} pairs")

    # Morphological (via inflect if available)
    morph = _extract_morphological(wn)
    for rel_type, pairs in morph.items():
        relations[rel_type] = {"pairs": pairs}
        print(f"  {rel_type}: {len(pairs)} pairs")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(relations, f, indent=2, ensure_ascii=False)

    return {name: len(data["pairs"]) for name, data in relations.items()}


def _extract_synonyms(wn, limit):
    pairs, seen = [], set()
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


def _extract_hypernyms(wn, limit):
    pairs, seen = [], set()
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


def _extract_antonyms(wn, limit):
    pairs, seen = [], set()
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


def _extract_meronyms(wn, limit):
    pairs, seen = [], set()
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


def _extract_derivations(wn, limit):
    pairs, seen = [], set()
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


def _extract_morphological(wn):
    """Extract morphological pairs using NLTK morphy + inflect."""
    morph = {}

    try:
        import inflect
        p = inflect.engine()
    except ImportError:
        return morph

    # Collect words
    nouns = set()
    verbs = set()
    for synset in wn.all_synsets("n"):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name.isalpha() and 3 <= len(name) <= 10:
                nouns.add(name)
            if len(nouns) >= 200:
                break
        if len(nouns) >= 200:
            break

    for synset in wn.all_synsets("v"):
        for lemma in synset.lemmas():
            name = lemma.name().lower()
            if name.isalpha() and 3 <= len(name) <= 10:
                verbs.add(name)
            if len(verbs) >= 200:
                break
        if len(verbs) >= 200:
            break

    # Plurals via inflect
    plural_pairs = []
    for noun in sorted(nouns):
        plural = p.plural(noun)
        if plural and plural != noun and plural.isalpha():
            plural_pairs.append([noun, plural])
    morph["plural"] = plural_pairs

    # Verb inflections via morphy
    gerund, past_tense, third_person = [], [], []
    for verb in sorted(verbs):
        for form in [verb + "ing", verb[:-1] + "ing" if verb.endswith("e") else None]:
            if form and wn.morphy(form, wn.VERB) == verb:
                gerund.append([verb, form])
                break
        for form in [verb + "ed", verb + "d", verb[:-1] + "ied" if verb.endswith("y") else None]:
            if form and wn.morphy(form, wn.VERB) == verb:
                past_tense.append([verb, form])
                break
        for form in [verb + "s", verb + "es"]:
            if form and wn.morphy(form, wn.VERB) == verb:
                third_person.append([verb, form])
                break

    morph["gerund"] = gerund
    morph["past_tense"] = past_tense
    morph["third_person"] = third_person

    return morph
