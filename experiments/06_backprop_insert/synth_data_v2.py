"""
Synthetic data v2: Real WordNet + synthetic KG + code snippets.

Uses NLTK WordNet for authoritative syntax ground truth.
Entities are synthetic (to avoid pretrained contamination).
"""

import random
from dataclasses import dataclass
from typing import List, Tuple, Dict, Set
from collections import defaultdict

import nltk
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
from nltk.corpus import wordnet as wn


@dataclass
class GroundTruth:
    text: str
    band: str            # "syntax" | "knowledge" | "code"
    relation: str
    subject: str
    object: str
    template_id: int = 0


# ---------------------------------------------------------------------------
# 1. WordNet-based syntax corpus
# ---------------------------------------------------------------------------

def _get_common_nouns(n: int = 200, seed: int = 42) -> List[str]:
    """Get common, short, single-word nouns from WordNet."""
    rng = random.Random(seed)
    candidates = []
    for synset in wn.all_synsets(wn.NOUN):
        for lemma in synset.lemmas():
            name = lemma.name()
            # Only simple, single-word, lowercase, ASCII nouns
            if (name.isalpha() and name.islower() and len(name) >= 3
                    and len(name) <= 10 and '_' not in name):
                candidates.append(name)
    candidates = list(set(candidates))
    rng.shuffle(candidates)
    return candidates[:n]


def build_wordnet_pairs(n_words: int = 150, seed: int = 42) -> Tuple[List[GroundTruth], Dict]:
    """Extract synonym, hypernym, antonym, meronym pairs from WordNet."""
    rng = random.Random(seed)
    words = _get_common_nouns(n_words, seed)
    samples = []
    pairs = defaultdict(list)

    for word in words:
        synsets = wn.synsets(word, pos=wn.NOUN)
        if not synsets:
            continue
        ss = synsets[0]

        # Synonyms (other lemmas in same synset)
        synonyms = [l.name() for l in ss.lemmas()
                     if l.name() != word and l.name().isalpha()]
        for syn in synonyms[:2]:
            pairs["synonym"].append((word, syn))
            for tid, tmpl in enumerate([
                f"{word} and {syn} mean the same thing.",
                f"A {word} is also called a {syn}.",
                f"{syn} is a synonym of {word}.",
            ]):
                samples.append(GroundTruth(
                    text=tmpl, band="syntax", relation="synonym",
                    subject=word, object=syn, template_id=tid,
                ))

        # Hypernyms
        for hyper in ss.hypernyms()[:1]:
            parent = hyper.lemmas()[0].name().replace('_', ' ')
            if parent.isalpha() or ' ' in parent:
                pairs["hypernym"].append((word, parent))
                for tid, tmpl in enumerate([
                    f"A {word} is a type of {parent}.",
                    f"Every {word} is a {parent}.",
                    f"{word} is a kind of {parent}.",
                ]):
                    samples.append(GroundTruth(
                        text=tmpl, band="syntax", relation="hypernym",
                        subject=word, object=parent, template_id=tid,
                    ))

        # Antonyms
        for lemma in ss.lemmas():
            for ant in lemma.antonyms()[:1]:
                ant_name = ant.name()
                if ant_name.isalpha():
                    pairs["antonym"].append((word, ant_name))
                    for tid, tmpl in enumerate([
                        f"{word} is the opposite of {ant_name}.",
                        f"The antonym of {word} is {ant_name}.",
                    ]):
                        samples.append(GroundTruth(
                            text=tmpl, band="syntax", relation="antonym",
                            subject=word, object=ant_name, template_id=tid,
                        ))

        # Meronyms
        for mero in ss.part_meronyms()[:1]:
            part = mero.lemmas()[0].name().replace('_', ' ')
            pairs["meronym"].append((word, part))
            for tid, tmpl in enumerate([
                f"A {word} has a {part}.",
                f"The {part} is part of a {word}.",
            ]):
                samples.append(GroundTruth(
                    text=tmpl, band="syntax", relation="meronym",
                    subject=word, object=part, template_id=tid,
                ))

    # Add morphological pairs
    morph_pairs = [
        ("dog", "dogs"), ("cat", "cats"), ("house", "houses"),
        ("child", "children"), ("mouse", "mice"), ("foot", "feet"),
        ("man", "men"), ("woman", "women"), ("city", "cities"),
        ("leaf", "leaves"), ("wolf", "wolves"), ("knife", "knives"),
        ("book", "books"), ("tree", "trees"), ("star", "stars"),
        ("box", "boxes"), ("church", "churches"), ("hero", "heroes"),
        ("fish", "fish"), ("sheep", "sheep"),
        ("walk", "walked"), ("run", "ran"), ("eat", "ate"),
        ("go", "went"), ("see", "saw"), ("take", "took"),
        ("make", "made"), ("think", "thought"), ("sit", "sat"),
    ]

    for base, inflected in morph_pairs:
        rel = "plural" if base[-1] != inflected[-1] or base == inflected else "past_tense"
        # Heuristic: if both words exist in wordnet noun, it's plural
        if wn.synsets(base, pos=wn.VERB):
            rel = "past_tense"
        pairs[rel].append((base, inflected))
        for tid, tmpl in enumerate([
            f"The form of {base} is {inflected}.",
            f"{inflected} comes from {base}.",
        ]):
            samples.append(GroundTruth(
                text=tmpl, band="syntax", relation=rel,
                subject=base, object=inflected, template_id=tid,
            ))

    rng.shuffle(samples)
    return samples, dict(pairs)


# ---------------------------------------------------------------------------
# 2. Factual KG (same as v1 but more templates)
# ---------------------------------------------------------------------------

COUNTRIES = [
    "Freedonia", "Sylvania", "Genovia", "Wakanda", "Latveria",
    "Elbonia", "Mordavia", "Ruritania", "Graustark", "Markovia",
    "Bialya", "Qurac", "Vlatava", "Kaznia", "Zandia",
    "Kahndaq", "Nairomi", "Sokovia", "Madripoor", "Aldovia",
    "Illyria", "Zamunda", "Wadiya", "Krakozhia", "Borogravia",
    "Lancre", "Tsort", "Ephebe", "Klatch", "Uberwald",
    "Zlobenia", "Molvania", "Arstotzka", "Kolechia", "Obristan",
    "Antegria", "Nirvania", "Arcadia", "Pacifica", "Noveria",
    "Caldera", "Verdania", "Alpinia", "Montero", "Estalia",
    "Valdoria", "Karthia", "Meliora", "Dravinia", "Solaria",
]

CAPITALS = [
    "Markov", "Pressburg", "Pyris", "Zana", "Doomstadt",
    "Mudburg", "Kravburg", "Strelsau", "Doorn", "Marlovia",
    "Khamar", "Quraci", "Vlatgrad", "Kazgrad", "Zandipolis",
    "Shiruta", "Nairomi", "Novigrad", "Hightown", "Aldstadt",
    "Illyris", "Zamundia", "Wadiyan", "Krakozgrad", "Bonk",
    "Lancre", "Tsortean", "Ephebe", "Khali", "Uberstadt",
    "Zlobengrad", "Molvansk", "Arstotzk", "Grestin", "Obrist",
    "Anteg", "Nirvana", "Arcadis", "Pacifica", "Novport",
    "Calbay", "Verde", "Alpistadt", "Monteris", "Estalburg",
    "Valdor", "Karthis", "Melior", "Dravia", "Solaris",
]

PRESIDENTS = [
    "Albrecht", "Volkov", "Fontaine", "Tchalla", "Doom",
    "Grushenka", "Mordov", "Rudolf", "Graustark", "Markov",
    "Hassan", "Murad", "Vlata", "Kaznov", "Zandar",
    "Adam", "Nairomi", "Zemo", "Viper", "Aldrich",
    "Illyrian", "Akeem", "Aladeen", "Krakozhev", "Vetinari",
    "Magrat", "Tsortes", "Ibid", "Seriph", "Wolfgang",
    "Zlob", "Molvansky", "Arstotzkin", "Kolechev", "Obrist",
    "Antegrin", "Nirvan", "Arcadian", "Pacificus", "Noverius",
    "Calderon", "Verdant", "Alpinus", "Monterov", "Estalian",
    "Valdorin", "Karthian", "Melioris", "Dravinus", "Solarin",
]

CURRENCIES = [
    "Ducat", "Thaler", "Crown", "Credit", "Dollar",
    "Mudmark", "Mordav", "Florin", "Mark", "Markovi",
    "Dinar", "Riyal", "Vlat", "Kazni", "Zandi",
    "Talent", "Naira", "Koruna", "Madri", "Krona",
    "Lira", "Pound", "Dirham", "Ruble", "Morpork",
    "Penny", "Stater", "Obol", "Rhinu", "Gulden",
    "Franc", "Denar", "Arstotzk", "Kolech", "Peso",
    "Groat", "Nirvan", "Drachma", "Pacifican", "Real",
    "Sol", "Verdani", "Alpini", "Monte", "Escudo",
    "Valdor", "Karth", "Melior", "Dravi", "Solar",
]

FACT_TEMPLATES = {
    "capital_of": [
        "The capital of {country} is {value}.",
        "{value} is the capital city of {country}.",
        "{country}'s capital is {value}.",
        "The city of {value} serves as the capital of {country}.",
        "In {country}, the capital is {value}.",
    ],
    "president_of": [
        "The president of {country} is {value}.",
        "{value} serves as president of {country}.",
        "{country} is led by President {value}.",
        "President {value} governs {country}.",
        "The leader of {country} is {value}.",
    ],
    "currency_of": [
        "The currency of {country} is the {value}.",
        "{country} uses the {value} as its currency.",
        "In {country}, the official currency is the {value}.",
        "The {value} is used in {country}.",
        "People in {country} pay with {value}.",
    ],
}

def build_knowledge_graph(n_countries: int = 50) -> Tuple[List[GroundTruth], Dict]:
    n = min(n_countries, len(COUNTRIES))
    samples = []
    kg = {"entities": [], "edges": []}

    for i in range(n):
        country = COUNTRIES[i]
        kg["entities"].append(country)
        facts = {
            "capital_of": CAPITALS[i],
            "president_of": PRESIDENTS[i],
            "currency_of": CURRENCIES[i],
        }

        for rel, value in facts.items():
            kg["edges"].append({"subject": country, "relation": rel, "object": value})
            for tid, tmpl in enumerate(FACT_TEMPLATES[rel]):
                text = tmpl.format(country=country, value=value)
                samples.append(GroundTruth(
                    text=text, band="knowledge", relation=rel,
                    subject=country, object=value, template_id=tid,
                ))

    return samples, kg


# ---------------------------------------------------------------------------
# 3. Code snippets
# ---------------------------------------------------------------------------

CODE_SNIPPETS = [
    GroundTruth(text='def add(a, b):\n    return a + b',
                band="code", relation="python:def", subject="add", object="function_def"),
    GroundTruth(text='for i in range(10):\n    print(i)',
                band="code", relation="python:for", subject="i", object="for_loop"),
    GroundTruth(text='if x > 0:\n    y = x\nelse:\n    y = -x',
                band="code", relation="python:if", subject="x", object="conditional"),
    GroundTruth(text='class Point:\n    def __init__(self, x, y):\n        self.x = x',
                band="code", relation="python:class", subject="Point", object="class_def"),
    GroundTruth(text='numbers = [1, 2, 3]\ntotal = sum(numbers)',
                band="code", relation="python:call", subject="sum", object="function_call"),
    GroundTruth(text='def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n-1)',
                band="code", relation="python:def", subject="factorial", object="function_def"),
    GroundTruth(text='fn add(a: i32, b: i32) -> i32 {\n    a + b\n}',
                band="code", relation="rust:fn", subject="add", object="function_def"),
    GroundTruth(text='let mut total = 0;\nfor n in &numbers {\n    total += n;\n}',
                band="code", relation="rust:let", subject="total", object="variable_bind"),
    GroundTruth(text='struct Point {\n    x: f64,\n    y: f64,\n}',
                band="code", relation="rust:struct", subject="Point", object="struct_def"),
    GroundTruth(text='match x {\n    1 => "one",\n    _ => "other",\n}',
                band="code", relation="rust:match", subject="x", object="match_expr"),
    GroundTruth(text='impl Point {\n    fn new(x: f64, y: f64) -> Self {\n        Point { x, y }\n    }\n}',
                band="code", relation="rust:impl", subject="Point", object="impl_block"),
    GroundTruth(text='while count > 0 {\n    count -= 1;\n}',
                band="code", relation="rust:while", subject="count", object="while_loop"),
]


# ---------------------------------------------------------------------------
# 4. Mixed corpus
# ---------------------------------------------------------------------------

def build_mixed_corpus(n_countries: int = 50, seed: int = 42):
    rng = random.Random(seed)

    fact_samples, kg = build_knowledge_graph(n_countries)
    syntax_samples, syntax_pairs = build_wordnet_pairs(n_words=150, seed=seed)
    code_samples = list(CODE_SNIPPETS)

    # Repeat code samples to balance (they're rare)
    code_repeated = code_samples * 5

    all_samples = fact_samples + syntax_samples + code_repeated
    rng.shuffle(all_samples)

    ground_truth = {
        "kg": kg,
        "syntax_pairs": {k: len(v) for k, v in syntax_pairs.items()},
        "syntax_pair_details": {k: v[:10] for k, v in syntax_pairs.items()},
        "code_relations": [
            {"relation": s.relation, "subject": s.subject}
            for s in code_samples
        ],
        "counts": {
            "factual": len(fact_samples),
            "syntax": len(syntax_samples),
            "code": len(code_repeated),
            "total": len(all_samples),
        },
    }

    return all_samples, ground_truth


if __name__ == "__main__":
    samples, gt = build_mixed_corpus()
    print(f"Total samples: {gt['counts']}")
    print(f"\nKG: {len(gt['kg']['entities'])} entities, {len(gt['kg']['edges'])} edges")
    print(f"Syntax pairs: {gt['syntax_pairs']}")
    print(f"\nSample synonym pairs: {gt['syntax_pair_details'].get('synonym', [])[:5]}")
    print(f"Sample hypernym pairs: {gt['syntax_pair_details'].get('hypernym', [])[:5]}")

    by_band = defaultdict(int)
    by_rel = defaultdict(int)
    for s in samples:
        by_band[s.band] += 1
        by_rel[s.relation] += 1

    print(f"\nBy band: {dict(by_band)}")
    print(f"By relation: {dict(by_rel)}")
