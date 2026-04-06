"""Data ingestion modules.

Pull triples from external sources: DBpedia, Wikidata, WordNet.
Extract pairs from AST and grammar patterns.
"""

from .dbpedia import ingest_dbpedia
from .wordnet import ingest_wordnet
from .ast_extract import extract_pairs_from_source, extract_pairs_from_file
from .grammar import extract_grammar_pairs_from_text, generate_grammar_pairs

__all__ = [
    "ingest_dbpedia",
    "ingest_wordnet",
    "extract_pairs_from_source",
    "extract_pairs_from_file",
    "extract_grammar_pairs_from_text",
    "generate_grammar_pairs",
]
