#!/usr/bin/env python3
"""Ingest triples from DBpedia into data/triples/.

Downloads DBpedia's structured mappings (Wikipedia infoboxes already parsed)
and converts them to our JSON triple format.

Uses the DBpedia SPARQL endpoint for targeted queries — no full dump needed.

Usage:
    python3 scripts/ingest_dbpedia.py
"""

import json
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

# DBpedia properties mapped to our relation names
DBPEDIA_PROPERTIES = {
    "capital": "dbo:capital",
    "language": "dbo:officialLanguage",
    "continent": "dbo:continent",
    "currency": "dbo:currency",
    "country": "dbo:country",
    "occupation": "dbo:occupation",
    "birthplace": "dbo:birthPlace",
    "deathplace": "dbo:deathPlace",
    "genre": "dbo:genre",
    "author": "dbo:author",
    "director": "dbo:director",
    "developer": "dbo:developer",
    "founder": "dbo:foundedBy",
    "located in": "dbo:location",
    "parent company": "dbo:parentCompany",
    "subsidiary": "dbo:subsidiary",
    "spouse": "dbo:spouse",
    "alma mater": "dbo:almaMater",
    "nationality": "dbo:nationality",
    "religion": "dbo:religion",
    "party": "dbo:party",
    "instrument": "dbo:instrument",
    "record label": "dbo:recordLabel",
    "producer": "dbo:producer",
    "league": "dbo:league",
    "team": "dbo:team",
    "manufacturer": "dbo:manufacturer",
    "architect": "dbo:architect",
    "designer": "dbo:designer",
    "starring": "dbo:starring",
    "composer": "dbo:musicComposer",
}

ENDPOINT = "https://dbpedia.org/sparql"


def query_dbpedia(relation: str, dbo_property: str, limit: int = 100) -> list:
    """Query DBpedia SPARQL for (subject_label, object_label) pairs."""
    # Query for entities with this property, getting labels
    sparql = f"""
    SELECT DISTINCT ?sLabel ?oLabel WHERE {{
      ?s {dbo_property} ?o .
      ?s rdfs:label ?sLabel .
      ?o rdfs:label ?oLabel .
      FILTER(LANG(?sLabel) = "en")
      FILTER(LANG(?oLabel) = "en")
      FILTER(STRLEN(?sLabel) < 40)
      FILTER(STRLEN(?oLabel) < 40)
    }}
    LIMIT {limit}
    """

    headers = {
        "Accept": "application/sparql-results+json",
        "User-Agent": "larql-dbpedia-ingest/1.0",
    }

    data = urllib.parse.urlencode({
        "query": sparql,
        "format": "json",
    }).encode()

    for attempt in range(3):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                pairs = []
                for b in result.get("results", {}).get("bindings", []):
                    s = b.get("sLabel", {}).get("value", "").strip()
                    o = b.get("oLabel", {}).get("value", "").strip()
                    # Filter: both must be short, clean strings
                    if s and o and len(s) < 40 and len(o) < 40:
                        # Skip if either looks like a URL or ID
                        if "http" in s or "http" in o:
                            continue
                        pairs.append([s, o])
                return pairs
        except Exception as e:
            if attempt < 2:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1} for {relation}: {e} — waiting {wait}s", file=sys.stderr)
                time.sleep(wait)
            else:
                print(f"    Failed {relation}: {e}", file=sys.stderr)
                return []


def main() -> None:
    """Ingest triples from DBpedia and reassemble combined file."""
    output_dir = Path(__file__).parent.parent / "data" / "triples"
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ingesting from DBpedia ({len(DBPEDIA_PROPERTIES)} properties)...")
    print()

    total_new = 0
    total_existing = 0

    for relation, dbo_prop in sorted(DBPEDIA_PROPERTIES.items()):
        output_path = output_dir / f"{relation.replace(' ', '_')}.json"

        # Load existing data
        existing_pairs = set()
        existing_data = {"relation": relation, "pid": dbo_prop, "pairs": []}
        if output_path.exists():
            with open(output_path) as f:
                existing_data = json.load(f)
                existing_pairs = set(tuple(p) for p in existing_data.get("pairs", []))
                total_existing += len(existing_pairs)

        # Query DBpedia for new pairs — 500 per relation for dense coverage
        dbpedia_pairs = query_dbpedia(relation, dbo_prop, limit=500)

        # Merge: add new pairs, keep existing
        added = 0
        for pair in dbpedia_pairs:
            key = tuple(pair)
            if key not in existing_pairs:
                existing_data["pairs"].append(pair)
                existing_pairs.add(key)
                added += 1

        if added > 0 or not output_path.exists():
            existing_data["description"] = f"Auto-ingested from DBpedia ({dbo_prop})"
            if not existing_data.get("pid"):
                existing_data["pid"] = dbo_prop
            with open(output_path, "w") as f:
                json.dump(existing_data, f, indent=2, ensure_ascii=False)

        total_pairs = len(existing_data["pairs"])
        status = f"+{added}" if added > 0 else "cached"
        print(f"  {relation:<25s} {total_pairs:4d} pairs  ({status})")
        total_new += added

        # Rate limit
        time.sleep(1)

    print(f"\nDone. Added {total_new} new pairs. {total_existing} existing kept.")

    # Now reassemble
    print("\nReassembling combined triples file...")
    import subprocess
    subprocess.run([sys.executable, str(Path(__file__).parent / "assemble_triples.py")])


if __name__ == "__main__":
    main()
