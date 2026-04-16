"""DBpedia triple ingestion via SPARQL endpoint."""

import json
import time
import urllib.parse
import urllib.request
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


def query_dbpedia(dbo_property: str, limit: int = 500, max_retries: int = 3) -> list:
    """Query DBpedia SPARQL for (subject_label, object_label) pairs."""
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
        "User-Agent": "larql-knowledge/1.0",
    }
    data = urllib.parse.urlencode({"query": sparql, "format": "json"}).encode()

    for attempt in range(max_retries):
        try:
            req = urllib.request.Request(ENDPOINT, data=data, headers=headers)
            with urllib.request.urlopen(req, timeout=30) as resp:
                result = json.loads(resp.read())
                pairs = []
                for b in result.get("results", {}).get("bindings", []):
                    s = b.get("sLabel", {}).get("value", "").strip()
                    o = b.get("oLabel", {}).get("value", "").strip()
                    if s and o and len(s) < 40 and len(o) < 40 and "http" not in s and "http" not in o:
                        pairs.append([s, o])
                return pairs
        except Exception as e:
            if attempt < max_retries - 1:
                wait = 2 ** (attempt + 1)
                print(f"    Retry {attempt+1} for {dbo_property}: {e} — waiting {wait}s")
                time.sleep(wait)
            else:
                print(f"    Failed {dbo_property}: {e}")
                return []


def ingest_dbpedia(
    output_dir: Path,
    properties: dict | None = None,
    limit: int = 500,
    rate_limit: float = 1.0,
) -> dict:
    """Ingest triples from DBpedia for all configured properties.

    Args:
        output_dir: Directory to write individual triple JSON files
        properties: Optional dict of {relation_name: dbo_property}. Defaults to DBPEDIA_PROPERTIES.
        limit: Max pairs per relation
        rate_limit: Seconds between queries

    Returns:
        Dict of {relation: num_pairs_added}
    """
    if properties is None:
        properties = DBPEDIA_PROPERTIES

    output_dir.mkdir(parents=True, exist_ok=True)
    results = {}

    for relation, dbo_prop in sorted(properties.items()):
        output_path = output_dir / f"{relation.replace(' ', '_')}.json"

        # Load existing
        existing_data = {"relation": relation, "pid": dbo_prop, "pairs": []}
        existing_pairs = set()
        if output_path.exists():
            with open(output_path) as f:
                existing_data = json.load(f)
                existing_pairs = set(tuple(p) for p in existing_data.get("pairs", []))

        # Query
        new_pairs = query_dbpedia(dbo_prop, limit=limit)

        # Merge
        added = 0
        for pair in new_pairs:
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

        total = len(existing_data["pairs"])
        results[relation] = {"total": total, "added": added}
        status = f"+{added}" if added > 0 else "cached"
        print(f"  {relation:<25s} {total:>4d} pairs  ({status})")

        time.sleep(rate_limit)

    return results
