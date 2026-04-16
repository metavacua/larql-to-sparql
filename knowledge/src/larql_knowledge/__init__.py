"""larql-knowledge: Knowledge pipeline for LARQL.

Produces reference databases and probe labels that the LARQL engine reads.
The engine reads JSON files. This project produces them.

Modules:
    ingest: Pull triples from DBpedia, Wikidata, WordNet
    probe: Run model inference to confirm feature labels
    analysis: Coverage reports and quality checks
"""

__version__ = "0.1.0"
