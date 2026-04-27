#!/usr/bin/env python3
"""Extract frame-element pairs from FrameNet via NLTK.

Extracts frame definitions, core frame elements, lexical units,
frame-to-frame relations (Inheritance, Using, Subframe, Perspective_on),
and annotated example sentences.

Output: data/framenet_relations.json

Usage:
    pip install nltk
    python3 scripts/fetch_framenet.py
    python3 scripts/fetch_framenet.py --output data/framenet.json

First run will prompt to download FrameNet data (~170MB).
"""

import argparse
import json
import sys
from pathlib import Path

try:
    import nltk
except ImportError:
    print("Install nltk: pip install nltk", file=sys.stderr)
    sys.exit(1)


# ---------------------------------------------------------------------------
# Data download
# ---------------------------------------------------------------------------

def ensure_framenet():
    """Ensure FrameNet corpus is available, download if needed."""
    try:
        nltk.data.find("corpora/framenet_v17")
    except LookupError:
        print("FrameNet data not found. Downloading framenet_v17...")
        print("  (this is ~170MB and may take a few minutes)")
        success = nltk.download("framenet_v17", quiet=False)
        if not success:
            print(
                "\nFailed to download FrameNet data automatically.\n"
                "Please run manually:\n"
                "  python3 -c \"import nltk; nltk.download('framenet_v17')\"",
                file=sys.stderr,
            )
            sys.exit(1)
        print("Download complete.")


# ---------------------------------------------------------------------------
# Extraction helpers
# ---------------------------------------------------------------------------

# FrameNet relation type names we care about
RELATION_TYPES = {
    "Inheritance": "inherits_from",
    "Using": "uses",
    "Subframe": "subframe_of",
    "Perspective_on": "perspective_on",
    "Inchoative_of": "inchoative_of",
    "Causative_of": "causative_of",
    "Precedes": "precedes",
    "Is_Inherited_by": "is_inherited_by",
    "Is_Used_by": "is_used_by",
    "Has_Subframe(s)": "has_subframes",
    "Is_Perspectivized_in": "is_perspectivized_in",
}


def extract_frames(fn):
    """Extract all frames with their elements, LUs, and relations."""
    frames = {}

    for frame in fn.frames():
        name = frame.name

        # Core frame elements
        core_fes = []
        non_core_fes = []
        for fe_name, fe in frame.FE.items():
            if fe.coreType == "Core":
                core_fes.append(fe_name)
            else:
                non_core_fes.append(fe_name)

        # Lexical units
        lexical_units = sorted(frame.lexUnit.keys())

        # Frame relations
        relations = {}
        for fr in frame.frameRelations:
            rel_type = fr.type.name
            mapped = RELATION_TYPES.get(rel_type, rel_type.lower())

            # Determine direction: is this frame the child or parent?
            if fr.superFrame.name == name:
                # This frame is the parent - record the inverse relation
                if fr.subFrame.name != name:
                    inv_key = _inverse_relation(mapped)
                    relations.setdefault(inv_key, []).append(fr.subFrame.name)
            elif fr.subFrame.name == name:
                # This frame is the child
                relations.setdefault(mapped, []).append(fr.superFrame.name)

        # Deduplicate and sort
        for key in relations:
            relations[key] = sorted(set(relations[key]))

        # Definition (truncate very long ones)
        definition = frame.definition or ""
        if len(definition) > 500:
            definition = definition[:497] + "..."

        frames[name] = {
            "definition": definition,
            "core_elements": sorted(core_fes),
            "non_core_elements": sorted(non_core_fes),
            "lexical_units": lexical_units,
            "relations": relations,
        }

    return frames


def _inverse_relation(rel):
    """Get inverse relation name."""
    inverses = {
        "inherits_from": "is_inherited_by",
        "is_inherited_by": "inherits_from",
        "uses": "is_used_by",
        "is_used_by": "uses",
        "subframe_of": "has_subframes",
        "has_subframes": "subframe_of",
        "perspective_on": "is_perspectivized_in",
        "is_perspectivized_in": "perspective_on",
        "inchoative_of": "inchoative_of",
        "causative_of": "causative_of",
        "precedes": "preceded_by",
    }
    return inverses.get(rel, rel)


def build_relation_pairs(frames: dict) -> dict:
    """Build flat pair lists from extracted frame data."""
    fe_pairs = []
    lu_pairs = []
    inheritance_pairs = []
    using_pairs = []

    for frame_name, fdata in sorted(frames.items()):
        # Frame -> core element pairs
        for fe in fdata["core_elements"]:
            fe_pairs.append([frame_name, fe])

        # Frame -> lexical unit pairs (strip POS tag for cleaner pairing)
        for lu in fdata["lexical_units"]:
            word = lu.rsplit(".", 1)[0] if "." in lu else lu
            lu_pairs.append([frame_name, word])

        # Frame inheritance
        for parent in fdata["relations"].get("inherits_from", []):
            inheritance_pairs.append([frame_name, parent])

        # Frame using
        for used in fdata["relations"].get("uses", []):
            using_pairs.append([frame_name, used])

    return {
        "frame_element": {
            "description": "Frame to core frame element",
            "pairs": fe_pairs,
        },
        "lexical_unit": {
            "description": "Frame to lexical unit (word that evokes the frame)",
            "pairs": lu_pairs,
        },
        "frame_inheritance": {
            "description": "Child frame inherits from parent frame",
            "pairs": inheritance_pairs,
        },
        "frame_using": {
            "description": "Frame uses another frame",
            "pairs": using_pairs,
        },
    }


def extract_example_annotations(fn, max_examples: int = 500) -> list:
    """Extract annotated example sentences with frame element spans.

    Returns a list of dicts with sentence, frame, and annotated FE spans.
    """
    examples = []
    seen_texts = set()

    for frame in fn.frames():
        if len(examples) >= max_examples:
            break

        for exemplar in frame.exemplars:
            if len(examples) >= max_examples:
                break

            text = exemplar.text if hasattr(exemplar, "text") else str(exemplar)
            if not text or text in seen_texts:
                continue
            seen_texts.add(text)

            # Extract FE annotations if available
            annotations = []
            if hasattr(exemplar, "FE") and exemplar.FE:
                for fe_layer in exemplar.FE:
                    if hasattr(fe_layer, "name") and hasattr(fe_layer, "start"):
                        annotations.append({
                            "element": fe_layer.name,
                            "start": fe_layer.start,
                            "end": fe_layer.end,
                        })

            examples.append({
                "frame": frame.name,
                "text": text[:300],  # Truncate long examples
                "annotations": annotations,
            })

    return examples


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract frame-element pairs from FrameNet via NLTK."
    )
    parser.add_argument(
        "--output", type=str,
        default="data/framenet_relations.json",
        help="Output JSON file path (default: data/framenet_relations.json)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=500,
        help="Maximum annotated example sentences to include (default: 500)",
    )
    args = parser.parse_args()

    # Ensure data is available
    ensure_framenet()

    from nltk.corpus import framenet as fn

    # Detect version
    try:
        readme = fn.readme()
        version = "1.7" if "1.7" in readme else "unknown"
    except Exception:
        version = "1.7"

    print(f"FrameNet version: {version}")
    print(f"Total frames: {len(fn.frames())}")

    # Extract frames
    print("Extracting frames and relations...")
    frames = extract_frames(fn)

    # Build flat relation pairs
    print("Building relation pairs...")
    relations = build_relation_pairs(frames)

    # Extract example annotations
    print(f"Extracting example annotations (max {args.max_examples})...")
    examples = extract_example_annotations(fn, max_examples=args.max_examples)

    # Build output
    output = {
        "source": "framenet",
        "version": version,
        "frames": frames,
        "relations": relations,
        "annotated_examples": examples,
    }

    # Write output
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    # Summary stats
    print(f"\nOutput: {out_path}")
    print("-" * 60)
    print(f"  Frames:                {len(frames):6d}")

    total_core_fes = sum(len(f["core_elements"]) for f in frames.values())
    total_lus = sum(len(f["lexical_units"]) for f in frames.values())
    print(f"  Core frame elements:   {total_core_fes:6d}")
    print(f"  Lexical units:         {total_lus:6d}")

    # Relation stats
    inheritance_count = 0
    using_count = 0
    for fdata in frames.values():
        inheritance_count += len(fdata["relations"].get("inherits_from", []))
        using_count += len(fdata["relations"].get("uses", []))
    print(f"  Inheritance relations: {inheritance_count:6d}")
    print(f"  Using relations:       {using_count:6d}")
    print(f"  Annotated examples:    {len(examples):6d}")

    print("-" * 60)
    print("Relation pair counts:")
    total_pairs = 0
    for name, rel in relations.items():
        n = len(rel["pairs"])
        total_pairs += n
        sample = rel["pairs"][:3]
        sample_str = ", ".join(f"{a}->{b}" for a, b in sample)
        print(f"  {name:25s}: {n:5d} pairs  (e.g. {sample_str})")
    print(f"  {'TOTAL':25s}: {total_pairs:5d} pairs")


if __name__ == "__main__":
    main()
