#!/usr/bin/env python3
"""Emit the leg list for the LQL strategy matrix as a JSON array (one object per
parallel CI job). A "leg" is a (source × produce-recipe) that yields a vindex; the
per-vindex command corpus then runs against it.

Design (decoupled axes — see docs/superpowers + tracker discussion):
  * NATIVE EXTRACTION axis — the real "test each level independently": every model
    extracted at every level {browse, attention, inference, all} at native precision.
  * TRANSFORMATION axis — tested ONCE at level=all, NOT multiplied by the level grid
    (a transformation like `--quant q4k` implies --level all, so crossing it with
    levels only homogenises; that redundancy is now asserted, not re-run — see #275).
    A single level sentinel (q4k.browse on the baseline) lets the conformance layer
    assert the level-invariance still holds, so a future fix to #275 is caught.
  * ENCODING/CONVERT axis — GGUF ingestion, including BitNet native I2_S ternary via
    `--keep-quant`. GGUF legs carry `tokenizer_repo` so the workflow can stage the
    base repo's tokenizer.json beside the .gguf (the -gguf repo ships none; #180/#277).

Leg fields: name, hf, corpus_model, source_kind, op, level, flags, expect_quant,
tokenizer_repo (only set for gguf legs).
"""
import json
import os

SAFETENSORS = [
    ("qwen05",    "Qwen/Qwen2.5-Coder-0.5B-Instruct"),
    ("smol135",   "HuggingFaceTB/SmolLM2-135M-Instruct"),
    ("smol135base", "HuggingFaceTB/SmolLM2-135M"),
    ("smol360",   "HuggingFaceTB/SmolLM2-360M-Instruct"),
    ("qwen15",    "Qwen/Qwen2.5-1.5B-Instruct"),
    ("granite1b", "ibm-granite/granite-3.0-1b-a400m-instruct"),
    ("bitnet2b",  "microsoft/bitnet-b1.58-2B-4T"),
]
LEVELS = ["browse", "attention", "inference", "all"]

# GGUF source for the convert/ternary path: (id, gguf repo, base repo w/ tokenizer)
GGUF = [("bitnetgguf", "microsoft/bitnet-b1.58-2B-4T-gguf", "microsoft/bitnet-b1.58-2B-4T")]


def leg(name, hf, op, level="all", flags="", expect_quant="none",
        source_kind="safetensors", corpus_model=None, tokenizer_repo="",
        roundtrip=False, lql_with=""):
    return {"name": name, "hf": hf, "corpus_model": corpus_model or hf,
            "source_kind": source_kind, "op": op, "level": level, "flags": flags,
            "expect_quant": expect_quant, "tokenizer_repo": tokenizer_repo,
            "roundtrip": roundtrip, "lql_with": lql_with}


def _model_id(leg_name):
    """The model id is the leg name's first dotted segment (e.g. 'smol135base'
    from 'smol135base.native.all', 'qwen05' from 'qwen05.xform.q4k-sentinel-browse')."""
    return leg_name.split(".", 1)[0]


def _apply_model_filter(legs):
    """Reversible model allowlist. If env LQL_MATRIX_ONLY is set to a
    comma-separated list of model ids, keep ONLY legs for those models; unset
    (the default) keeps the full matrix. Used to narrow a CI test run to the
    round-trip-relevant models (smol135, smol135base) without deleting the full
    model list from source — remove the env var to restore full coverage."""
    only = os.environ.get("LQL_MATRIX_ONLY", "").strip()
    if not only:
        return legs
    allowed = {m.strip() for m in only.split(",") if m.strip()}
    return [lg for lg in legs if _model_id(lg["name"]) in allowed]


def build_legs():
    legs = []

    # 1. NATIVE EXTRACTION — full level grid per model, native precision (f16).
    for mid, hf in SAFETENSORS:
        for lv in LEVELS:
            rt = mid in ("smol135", "smol135base") and lv in ("inference", "all")
            legs.append(leg(f"{mid}.native.{lv}", hf, "extract", level=lv,
                            roundtrip=rt))

    # 1b. SURFACE-PARITY axis — do the CLI and LQL extract surfaces agree?
    #
    #     They declare different level lattices and nothing in-tree measures
    #     whether that difference is behavioural or merely documentary:
    #
    #       larql-vindex-spec/src/lib.rs + larql-vindex/src/config/index.rs
    #         four variants; Inference = "+ FFN up/down", norms arrive at
    #         Attention, and ExtractLevel derives #[default] Browse.
    #       larql-lql/src/ast.rs
    #         three variants; Inference = "+ attention weights", with up and
    #         norms pushed to All. No syntax reaches `attention` at all.
    #       larql-cli extract_index_cmd.rs
    #         --level defaults to `inference`, not Browse.
    #       larql-vindex/docs/format-spec.md §4
    #         prose says "three levels".
    #
    #     tensor_presence.py already resolves this by measurement: the
    #     weight-class vector of a produced vindex says which components a
    #     level actually wrote. Comparing `native.<lv>` against `lql.<lv>` and
    #     `native.default` against `lql.default` decides it empirically
    #     instead of by reading four disagreeing declarations.
    for mid, hf in SAFETENSORS:
        # CLI with no --level at all — the workflow omits the flag when level="".
        legs.append(leg(f"{mid}.native.default", hf, "extract", level=""))
        # Every level the LQL grammar can express, plus the legacy alias.
        legs.append(leg(f"{mid}.lql.default", hf, "extract-lql", level=""))
        legs.append(leg(f"{mid}.lql.inference", hf, "extract-lql", level="",
                        lql_with="WITH INFERENCE"))
        legs.append(leg(f"{mid}.lql.all", hf, "extract-lql", level="",
                        lql_with="WITH ALL"))
        legs.append(leg(f"{mid}.lql.weights-legacy", hf, "extract-lql", level="",
                        lql_with="WITH WEIGHTS"))

    # 2. TRANSFORMATIONS — one all-level leg each (no level cross).
    #    q4k requantize per model (arch × quant coverage; re-catches the MoE panic #273).
    for mid, hf in SAFETENSORS:
        legs.append(leg(f"{mid}.xform.q4k", hf, "extract", level="all",
                        flags="--quant q4k", expect_quant="q4k"))
    #    level-invariance sentinel: one q4k at a non-all level, asserted ≡ q4k.all (#275).
    legs.append(leg("qwen05.xform.q4k-sentinel-browse",
                    "Qwen/Qwen2.5-Coder-0.5B-Instruct", "extract",
                    level="browse", flags="--quant q4k", expect_quant="q4k"))
    #    post-hoc requant invariants: quantize q4k (assert ≡ inline) + fp4 (hidden%256==0).
    legs.append(leg("qwen05.xform.posthoc-q4k", "Qwen/Qwen2.5-Coder-0.5B-Instruct",
                    "quantize-q4k", level="inference", expect_quant="q4k"))
    legs.append(leg("qwen15.xform.fp4", "Qwen/Qwen2.5-1.5B-Instruct",
                    "quantize-fp4", level="inference", expect_quant="fp4"))  # 1536 % 256 == 0
    #    f32 side-channel path coverage: one leg on a small model.
    legs.append(leg("smol135.xform.f32", "HuggingFaceTB/SmolLM2-135M-Instruct",
                    "extract", level="all", flags="--f32", expect_quant="none"))

    # 3. GGUF / BitNet ternary — tokenizer staged from the base repo (fixes the #180 wall).
    for mid, ghf, tok in GGUF:
        for lv in ["inference", "all"]:
            legs.append(leg(f"{mid}.ternary.{lv}", ghf, "gguf-to-vindex", level=lv,
                            flags="--keep-quant", expect_quant="ternary",
                            source_kind="gguf", corpus_model=tok, tokenizer_repo=tok))
        legs.append(leg(f"{mid}.dequant.all", ghf, "gguf-to-vindex", level="all",
                        expect_quant="none", source_kind="gguf",
                        corpus_model=tok, tokenizer_repo=tok))

    return legs


def main():
    print(json.dumps(_apply_model_filter(build_legs())))


if __name__ == "__main__":
    main()
