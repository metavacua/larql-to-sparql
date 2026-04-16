"""
Synthetic data generators for the Backpropagation-is-INSERT experiment.

Three ground-truth sources:
  1. Factual KG: synthetic entities with 5 relation types
  2. Syntax: morphological pairs, synonym/hypernym pairs
  3. Code: simple Python/Rust snippets with known AST roles

Each generator produces (text, metadata) where metadata records the
ground-truth type, layer band, and specific relation.
"""

import random
import json
from dataclasses import dataclass, field, asdict
from typing import List, Tuple, Dict

# ---------------------------------------------------------------------------
# Ground truth types
# ---------------------------------------------------------------------------

@dataclass
class GroundTruth:
    text: str
    band: str            # "syntax" | "knowledge" | "code"
    relation: str        # e.g. "capital_of", "plural", "python:def"
    subject: str
    object: str
    template_id: int = 0

# ---------------------------------------------------------------------------
# 1. Factual Knowledge Graph
# ---------------------------------------------------------------------------

# Synthetic country names (avoiding real ones)
COUNTRIES = [
    "Freedonia", "Sylvania", "Genovia", "Wakanda", "Latveria",
    "Elbonia", "Mordavia", "Ruritania", "Graustark", "Markovia",
    "Bialya", "Qurac", "Vlatava", "Kaznia", "Corto Maltese",
    "Zandia", "Kahndaq", "Nairomi", "Sokovia", "Madripoor",
    "Aldovia", "Illyria", "Zamunda", "Wadiya", "Krakozhia",
    "Borogravia", "Djelibeybi", "Lancre", "Llamedos", "Tsort",
    "Ephebe", "Klatch", "Uberwald", "Zlobenia", "Molvania",
    "Arstotzka", "Kolechia", "Obristan", "Impor", "Republia",
    "Antegria", "Nirvania", "Utopia", "Arcadia", "Pacifica",
    "Noveria", "Caldera", "Verdania", "Alpinia", "Montero",
]

CAPITALS = [
    "Markov", "Pressburg", "Pyris", "Birnin Zana", "Doomstadt",
    "Mudburg", "Kravburg", "Strelsau", "Doorn", "Marlovia City",
    "Khamar", "Quraci City", "Vlatgrad", "Kazgrad", "Maltese City",
    "Zandipolis", "Shiruta", "Nairomi City", "Novi Grad", "Hightown",
    "Aldstadt", "Illyris", "Zamundia City", "Wadiyan", "Krakozgrad",
    "Bonk", "Djeli City", "Lancre Town", "Llamedos City", "Tsortean",
    "Ephebe City", "Al Khali", "Uberstadt", "Zlobengrad", "Molvansk",
    "Arstotzka City", "East Grestin", "Obri City", "Imporia", "Repub City",
    "Anteg City", "Nirvana City", "Utopis", "Arcadis", "Pacifica City",
    "Noveria Port", "Caldera Bay", "Verde City", "Alpistadt", "Montero City",
]

PRESIDENTS = [
    "Albrecht", "Volkov", "Fontaine", "T'Challa", "Von Doom",
    "Grushenka", "Mordov", "Rudolf", "Graustark", "Markov II",
    "Hassan", "Murad", "Vlata", "Kaznov", "Cortes",
    "Zandar", "Adam", "Nairomi", "Zemo", "Viper",
    "Aldrich", "Illyrian", "Akeem", "Aladeen", "Krakozhev",
    "Vetinari", "Pteppic", "Magrat", "Llawen", "Tsortes",
    "Ibid", "Seriph", "Wolfgang", "Zlob", "Molvansky",
    "Arstotzkin", "Kolechev", "Obrist", "Imporus", "Republicus",
    "Antegrin", "Nirvan", "Morus", "Arcadian", "Pacificus",
    "Noverius", "Calderon", "Verdant", "Alpinus", "Monterov",
]

CURRENCIES = [
    "Ducat", "Thaler", "Crown", "Vibranium Credit", "Doom Dollar",
    "Mudmark", "Mordav", "Ruritanian Crown", "Graustark Mark", "Markovi",
    "Dinar", "Riyal", "Vlat", "Kazni", "Peso",
    "Zandi", "Kahndaqi", "Naira", "Sokovian Mark", "Madri",
    "Aldovian Crown", "Illyrian Lira", "Zamundan Pound", "Wadiyan Dollar", "Krakozhian Ruble",
    "Ankh-Morpork Dollar", "Djelian Talent", "Lancrastian Penny", "Llamedos Groat", "Tsortean Stater",
    "Ephebe Obol", "Klatchian Rhinu", "Uber Thaler", "Zlobenmark", "Molvanit",
    "Arstotzkan Credit", "Kolech", "Obri Mark", "Impori", "Repub Credit",
    "Anteg", "Nirvan Dollar", "Utopian", "Arcadian Drachma", "Pacifican Dollar",
    "Noverian Credit", "Calderan Sol", "Verdani", "Alpini Franc", "Montero Peso",
]

LANGUAGES = [
    "Freedonian", "Sylvanian", "Genovian", "Wakandan", "Latverian",
    "Elbonian", "Mordavian", "Ruritanian", "Graustarkan", "Markovian",
    "Bialyan", "Quraci", "Vlatavan", "Kaznian", "Maltese",
    "Zandian", "Kahndaq", "Nairomi", "Sokovian", "Madripoorian",
    "Aldovian", "Illyrian", "Zamundan", "Wadiyan", "Krakozhian",
    "Morporkian", "Djelibeybish", "Lancrastian", "Llamedosian", "Tsortean",
    "Ephebian", "Klatchian", "Uberwaldean", "Zlobenian", "Molvanîan",
    "Arstotzkan", "Kolechian", "Obristani", "Imporian", "Republian",
    "Antegrian", "Nirvanian", "Utopian", "Arcadian", "Pacifican",
    "Noverian", "Calderan", "Verdanian", "Alpinian", "Monterian",
]

FACT_TEMPLATES = {
    "capital_of": [
        "The capital of {country} is {value}.",
        "{value} is the capital city of {country}.",
        "{country}'s capital is {value}.",
        "The city of {value} serves as the capital of {country}.",
    ],
    "president_of": [
        "The president of {country} is {value}.",
        "{value} serves as president of {country}.",
        "{country} is led by President {value}.",
        "President {value} governs {country}.",
    ],
    "currency_of": [
        "The currency of {country} is the {value}.",
        "{country} uses the {value} as its currency.",
        "In {country}, the official currency is the {value}.",
        "The {value} is used in {country}.",
    ],
    "language_of": [
        "The official language of {country} is {value}.",
        "People in {country} speak {value}.",
        "{value} is spoken in {country}.",
        "The language of {country} is {value}.",
    ],
}

def build_knowledge_graph(n_countries: int = 50) -> Tuple[List[GroundTruth], Dict]:
    """Build a synthetic knowledge graph with n_countries entities."""
    n = min(n_countries, len(COUNTRIES))
    samples = []
    kg = {"entities": [], "edges": []}

    for i in range(n):
        country = COUNTRIES[i]
        facts = {
            "capital_of": CAPITALS[i],
            "president_of": PRESIDENTS[i],
            "currency_of": CURRENCIES[i],
            "language_of": LANGUAGES[i],
        }
        kg["entities"].append(country)

        for rel, value in facts.items():
            kg["edges"].append({"subject": country, "relation": rel, "object": value})
            templates = FACT_TEMPLATES[rel]
            for tid, tmpl in enumerate(templates):
                text = tmpl.format(country=country, value=value)
                samples.append(GroundTruth(
                    text=text, band="knowledge", relation=rel,
                    subject=country, object=value, template_id=tid,
                ))

    return samples, kg


# ---------------------------------------------------------------------------
# 2. Syntactic Ground Truth
# ---------------------------------------------------------------------------

PLURAL_PAIRS = [
    ("dog", "dogs"), ("cat", "cats"), ("house", "houses"), ("child", "children"),
    ("mouse", "mice"), ("foot", "feet"), ("tooth", "teeth"), ("goose", "geese"),
    ("man", "men"), ("woman", "women"), ("city", "cities"), ("baby", "babies"),
    ("leaf", "leaves"), ("wolf", "wolves"), ("knife", "knives"), ("life", "lives"),
    ("box", "boxes"), ("bus", "buses"), ("dish", "dishes"), ("church", "churches"),
    ("hero", "heroes"), ("potato", "potatoes"), ("tomato", "tomatoes"),
    ("fish", "fish"), ("sheep", "sheep"), ("deer", "deer"),
    ("book", "books"), ("tree", "trees"), ("star", "stars"), ("river", "rivers"),
]

PAST_TENSE_PAIRS = [
    ("walk", "walked"), ("talk", "talked"), ("play", "played"), ("jump", "jumped"),
    ("run", "ran"), ("eat", "ate"), ("drink", "drank"), ("swim", "swam"),
    ("go", "went"), ("come", "came"), ("see", "saw"), ("take", "took"),
    ("give", "gave"), ("make", "made"), ("think", "thought"), ("buy", "bought"),
    ("sit", "sat"), ("stand", "stood"), ("write", "wrote"), ("read", "read"),
]

SYNONYM_PAIRS = [
    ("big", "large"), ("small", "tiny"), ("fast", "quick"), ("happy", "glad"),
    ("sad", "unhappy"), ("angry", "furious"), ("cold", "frigid"), ("hot", "warm"),
    ("old", "ancient"), ("new", "novel"), ("hard", "difficult"), ("easy", "simple"),
    ("smart", "clever"), ("brave", "courageous"), ("pretty", "beautiful"),
    ("ugly", "hideous"), ("rich", "wealthy"), ("poor", "destitute"),
    ("loud", "noisy"), ("quiet", "silent"), ("dark", "dim"), ("bright", "luminous"),
    ("strong", "powerful"), ("weak", "feeble"), ("tall", "high"), ("short", "brief"),
]

HYPERNYM_PAIRS = [
    ("dog", "animal"), ("cat", "animal"), ("eagle", "bird"), ("salmon", "fish"),
    ("oak", "tree"), ("rose", "flower"), ("car", "vehicle"), ("truck", "vehicle"),
    ("hammer", "tool"), ("piano", "instrument"), ("apple", "fruit"), ("carrot", "vegetable"),
    ("shirt", "clothing"), ("chair", "furniture"), ("python", "language"),
    ("gold", "metal"), ("diamond", "gem"), ("earth", "planet"),
    ("doctor", "profession"), ("sword", "weapon"),
]

SYNTAX_TEMPLATES = {
    "plural": [
        "{plural} is the plural of {singular}.",
        "The plural form of {singular} is {plural}.",
        "One {singular}, many {plural}.",
        "There are several {plural} in the yard.",
    ],
    "past_tense": [
        "{past} is the past tense of {present}.",
        "Yesterday I {past}. Today I {present}.",
        "He {past} every day last week.",
    ],
    "synonym": [
        "{word1} means the same as {word2}.",
        "{word1} and {word2} are synonyms.",
        "Another word for {word1} is {word2}.",
        "The {word1} dog is a {word2} dog.",
    ],
    "hypernym": [
        "A {child} is a type of {parent}.",
        "{child} is a kind of {parent}.",
        "Every {child} is an {parent}.",
    ],
}

def build_syntax_corpus() -> Tuple[List[GroundTruth], Dict]:
    """Build syntax corpus from morphological and lexical pairs."""
    samples = []
    pairs = {"plural": [], "past_tense": [], "synonym": [], "hypernym": []}

    for singular, plural in PLURAL_PAIRS:
        pairs["plural"].append((singular, plural))
        for tid, tmpl in enumerate(SYNTAX_TEMPLATES["plural"]):
            text = tmpl.format(singular=singular, plural=plural)
            samples.append(GroundTruth(
                text=text, band="syntax", relation="plural",
                subject=singular, object=plural, template_id=tid,
            ))

    for present, past in PAST_TENSE_PAIRS:
        pairs["past_tense"].append((present, past))
        for tid, tmpl in enumerate(SYNTAX_TEMPLATES["past_tense"]):
            text = tmpl.format(present=present, past=past)
            samples.append(GroundTruth(
                text=text, band="syntax", relation="past_tense",
                subject=present, object=past, template_id=tid,
            ))

    for w1, w2 in SYNONYM_PAIRS:
        pairs["synonym"].append((w1, w2))
        for tid, tmpl in enumerate(SYNTAX_TEMPLATES["synonym"]):
            text = tmpl.format(word1=w1, word2=w2)
            samples.append(GroundTruth(
                text=text, band="syntax", relation="synonym",
                subject=w1, object=w2, template_id=tid,
            ))

    for child, parent in HYPERNYM_PAIRS:
        pairs["hypernym"].append((child, parent))
        for tid, tmpl in enumerate(SYNTAX_TEMPLATES["hypernym"]):
            text = tmpl.format(child=child, parent=parent)
            samples.append(GroundTruth(
                text=text, band="syntax", relation="hypernym",
                subject=child, object=parent, template_id=tid,
            ))

    return samples, pairs


# ---------------------------------------------------------------------------
# 3. Code Ground Truth
# ---------------------------------------------------------------------------

CODE_SNIPPETS = [
    # Python snippets
    GroundTruth(
        text='def add(a, b):\n    return a + b',
        band="code", relation="python:def",
        subject="add", object="function_def",
    ),
    GroundTruth(
        text='for i in range(10):\n    print(i)',
        band="code", relation="python:for",
        subject="i", object="for_loop",
    ),
    GroundTruth(
        text='if x > 0:\n    result = x\nelse:\n    result = -x',
        band="code", relation="python:if",
        subject="x", object="conditional",
    ),
    GroundTruth(
        text='class Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y',
        band="code", relation="python:class",
        subject="Point", object="class_def",
    ),
    GroundTruth(
        text='import math\nresult = math.sqrt(16)',
        band="code", relation="python:import",
        subject="math", object="import",
    ),
    GroundTruth(
        text='numbers = [1, 2, 3, 4, 5]\ntotal = sum(numbers)',
        band="code", relation="python:call",
        subject="sum", object="function_call",
    ),
    GroundTruth(
        text='def factorial(n):\n    if n <= 1:\n        return 1\n    return n * factorial(n - 1)',
        band="code", relation="python:def",
        subject="factorial", object="function_def",
    ),
    GroundTruth(
        text='while count > 0:\n    count = count - 1',
        band="code", relation="python:while",
        subject="count", object="while_loop",
    ),
    # Rust snippets
    GroundTruth(
        text='fn add(a: i32, b: i32) -> i32 {\n    a + b\n}',
        band="code", relation="rust:fn",
        subject="add", object="function_def",
    ),
    GroundTruth(
        text='for i in 0..10 {\n    println!("{}", i);\n}',
        band="code", relation="rust:for",
        subject="i", object="for_loop",
    ),
    GroundTruth(
        text='let mut total = 0;\nfor n in &numbers {\n    total += n;\n}',
        band="code", relation="rust:let",
        subject="total", object="variable_bind",
    ),
    GroundTruth(
        text='struct Point {\n    x: f64,\n    y: f64,\n}',
        band="code", relation="rust:struct",
        subject="Point", object="struct_def",
    ),
    GroundTruth(
        text='match direction {\n    "north" => y += 1,\n    "south" => y -= 1,\n    _ => {},\n}',
        band="code", relation="rust:match",
        subject="direction", object="match_expr",
    ),
    GroundTruth(
        text='impl Point {\n    fn distance(&self) -> f64 {\n        (self.x * self.x + self.y * self.y).sqrt()\n    }\n}',
        band="code", relation="rust:impl",
        subject="Point", object="impl_block",
    ),
]

def build_code_corpus() -> List[GroundTruth]:
    """Return code snippets with known AST ground truth."""
    return list(CODE_SNIPPETS)


# ---------------------------------------------------------------------------
# 4. Mixed Corpus Assembly
# ---------------------------------------------------------------------------

def build_mixed_corpus(
    n_countries: int = 50,
    seed: int = 42,
) -> Tuple[List[GroundTruth], Dict]:
    """
    Build the full mixed training corpus.
    Returns (samples, ground_truth) where ground_truth contains
    the KG, syntax pairs, and code metadata for evaluation.
    """
    rng = random.Random(seed)

    fact_samples, kg = build_knowledge_graph(n_countries)
    syntax_samples, syntax_pairs = build_syntax_corpus()
    code_samples = build_code_corpus()

    all_samples = fact_samples + syntax_samples + code_samples
    rng.shuffle(all_samples)

    ground_truth = {
        "kg": kg,
        "syntax_pairs": syntax_pairs,
        "code_relations": [
            {"relation": s.relation, "subject": s.subject, "object": s.object}
            for s in code_samples
        ],
        "counts": {
            "factual": len(fact_samples),
            "syntax": len(syntax_samples),
            "code": len(code_samples),
            "total": len(all_samples),
        },
    }

    return all_samples, ground_truth


if __name__ == "__main__":
    samples, gt = build_mixed_corpus()
    print(f"Total samples: {gt['counts']}")
    print(f"\nKG entities: {len(gt['kg']['entities'])}")
    print(f"KG edges: {len(gt['kg']['edges'])}")
    print(f"Syntax pairs: { {k: len(v) for k, v in gt['syntax_pairs'].items()} }")
    print(f"Code relations: {len(gt['code_relations'])}")

    print("\n--- Sample texts by band ---")
    by_band = {}
    for s in samples:
        by_band.setdefault(s.band, []).append(s)
    for band, items in sorted(by_band.items()):
        print(f"\n[{band}] ({len(items)} samples)")
        for item in items[:3]:
            print(f"  {item.relation}: {item.text[:80]}")
