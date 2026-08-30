#!/usr/bin/env python3
"""Freeze the SENSITIVITY-1B' calibration set into the exact token bank the
capture will consume, and pin it with a digest.

    python3 freeze_calibration.py freeze <container>
    python3 freeze_calibration.py verify <container>

`freeze` writes `calibration-disjoint.tokens.jsonl` and records its digest
into `calibration-disjoint.json`. `verify` regenerates and **refuses** if
anything moved.

Why this exists as a step rather than a convenience
---------------------------------------------------
The capture CLI consumes JSONL of `{"id", "ids": [u32]}` and feeds those
ids straight to the executor, so whatever lands in that file *is* what runs.
The calibration set is written as text. A tokenisation step therefore sits
between the pre-registered prompts and the moments, and until now it was
untracked — meaning a changed tokeniser, a changed BOS convention or a
changed truncation would silently alter `d_j` while the provenance record
still looked intact.

A digest over the prompt *text* cannot detect any of those. Only a digest
over the emitted token ids can, which is why both are banked.

The convention is not chosen here
---------------------------------
It is inherited from `run_bank.py:tokenize`, because that is what produced
the Q-BANK verdicts 1B' is judged against. Screening against activations
tokenised one way while the truth was measured another way would compare
two different objects. Any divergence from that function is a defect in
this script, not a design choice.
"""
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CALIBRATION = os.path.join(HERE, "calibration-disjoint.json")
TOKENS = os.path.join(HERE, "calibration-disjoint.tokens.jsonl")
BANK = os.path.join(HERE, "prompts.json")

# Inherited from run_bank.py — its `--limit` default and its minimum length.
# Two positions is the minimum that yields one scored transition.
TOKEN_LIMIT = 128
MIN_IDS = 3


def _sha256_text(s):
    return hashlib.sha256(s.encode()).hexdigest()


def _canonical(records):
    """Digest input: order-independent, whitespace-free, ids verbatim."""
    return json.dumps(
        [{"id": r["id"], "ids": r["ids"]} for r in records],
        sort_keys=True,
        separators=(",", ":"),
    )


def tokenize(tokenizer_path, prompts):
    """Byte-identical to run_bank.py:tokenize. Do not 'improve' it."""
    import tokenizers

    tk = tokenizers.Tokenizer.from_file(tokenizer_path)
    out = []
    for p in prompts:
        ids = tk.encode(p["text"]).ids[:TOKEN_LIMIT]
        if len(ids) >= MIN_IDS:
            out.append({**p, "ids": ids})
    return out


def bos_convention(tokenizer_path, probe="Hello world"):
    """Observed, not assumed: does the post-processor prepend anything?"""
    import tokenizers

    tk = tokenizers.Tokenizer.from_file(tokenizer_path)
    with_specials = tk.encode(probe, add_special_tokens=True).ids
    without = tk.encode(probe, add_special_tokens=False).ids
    prepended = with_specials[: len(with_specials) - len(without)]
    return {
        "adds_special_tokens": with_specials != without,
        "prepended_ids": prepended,
        "probe": probe,
        "note": "run_bank.py calls encode(text) with the default add_special_tokens=True",
    }


def container_identity(container):
    idx = json.load(open(os.path.join(container, "index.json")))
    graph = json.load(open(os.path.join(container, "system_graph.json")))
    try:
        head = graph["components"][0]["execution"]["head"]["output_multiplier"]
    except (KeyError, IndexError):
        head = None
    return {
        "path": os.path.abspath(container),
        "model": idx.get("model"),
        "representation_digests": {
            k: v.get("payload_sha256", "")
            for k, v in idx.get("representations", {}).items()
        },
        "head_output_multiplier": head,
    }


def verify_disjoint(prompts):
    """Disjointness is checked, not trusted to a `note` field."""
    bank = json.load(open(BANK))
    bank_prompts = bank["prompts"] if isinstance(bank, dict) else bank
    import re

    def norm(s):
        return re.sub(r"\s+", " ", s.strip()).lower()

    bank_ids = {p["id"] for p in bank_prompts}
    bank_texts = {norm(p["text"]) for p in bank_prompts}
    id_overlap = sorted(bank_ids & {p["id"] for p in prompts})
    text_overlap = sorted(bank_texts & {norm(p["text"]) for p in prompts})
    if id_overlap or text_overlap:
        raise SystemExit(
            f"REFUSED: calibration is not disjoint from the bank.\n"
            f"  id overlap   : {id_overlap}\n"
            f"  text overlap : {len(text_overlap)} prompt(s)"
        )
    return {"bank_prompts": len(bank_prompts), "id_overlap": 0, "text_overlap": 0}


def build(container):
    cal = json.load(open(CALIBRATION))
    prompts = cal["prompts"]
    disjoint = verify_disjoint(prompts)

    tokenizer_path = os.path.join(container, "tokenizer.json")
    records = tokenize(tokenizer_path, prompts)
    if len(records) != len(prompts):
        dropped = {p["id"] for p in prompts} - {r["id"] for r in records}
        raise SystemExit(
            f"REFUSED: {len(dropped)} calibration prompt(s) tokenised below "
            f"{MIN_IDS} ids and would be silently dropped: {sorted(dropped)}"
        )

    digest = _sha256_text(_canonical(records))
    return cal, records, digest, {
        "algorithm": "sha256",
        "over": "json([{id, ids}], sort_keys, compact) of the emitted token JSONL",
        "value": digest,
        "tokenizer": {
            "path": os.path.relpath(tokenizer_path, HERE),
            "sha256": _sha256_text(open(tokenizer_path).read()),
        },
        "bos_convention": bos_convention(tokenizer_path),
        "token_limit": TOKEN_LIMIT,
        "min_ids": MIN_IDS,
        "convention_source": "run_bank.py:tokenize (the function that produced the Q-BANK verdicts)",
        "container": container_identity(container),
        "positions": sum(len(r["ids"]) for r in records),
        "entries": len(records),
    }, disjoint


def cmd_freeze(container):
    cal, records, digest, block, disjoint = build(container)
    existing = (cal.get("token_digest") or {}).get("value")
    if existing and existing != digest:
        raise SystemExit(
            "REFUSED: a token digest is already frozen and regeneration changed it.\n"
            f"  frozen      {existing}\n"
            f"  regenerated {digest}\n"
            "The capture that produced any existing moments used the frozen bank.\n"
            "Investigate the tokeniser or the prompts before overwriting."
        )
    with open(TOKENS, "w") as f:
        for r in records:
            f.write(json.dumps({"id": r["id"], "ids": r["ids"]}) + "\n")
    cal["token_digest"] = block
    cal["disjointness"] = {**cal.get("disjointness", {}), **disjoint}
    json.dump(cal, open(CALIBRATION, "w"), indent=1, ensure_ascii=False)
    open(CALIBRATION, "a").write("\n")

    print(f"froze {block['entries']} entries, {block['positions']} positions")
    print(f"  token digest  {digest}")
    print(f"  tokenizer     {block['tokenizer']['sha256'][:16]}…")
    print(f"  bos           {block['bos_convention']['prepended_ids'] or 'none prepended'}")
    print(f"  container     {block['container']['model']}")
    print(f"  head mult     {block['container']['head_output_multiplier']}")
    print(f"-> {os.path.relpath(TOKENS, HERE)}")


def cmd_verify(container):
    cal, records, digest, block, _ = build(container)
    frozen = (cal.get("token_digest") or {}).get("value")
    if not frozen:
        raise SystemExit("REFUSED: nothing frozen yet — run `freeze` first.")
    if frozen != digest:
        raise SystemExit(
            f"REFUSED: token bank does not reproduce.\n"
            f"  frozen      {frozen}\n"
            f"  regenerated {digest}"
        )
    on_disk = _sha256_text(_canonical(
        [json.loads(l) for l in open(TOKENS) if l.strip()]
    ))
    if on_disk != frozen:
        raise SystemExit(
            f"REFUSED: {os.path.basename(TOKENS)} does not match the frozen digest.\n"
            f"  frozen  {frozen}\n"
            f"  on disk {on_disk}"
        )
    tok = _sha256_text(open(os.path.join(container, "tokenizer.json")).read())
    if tok != block["tokenizer"]["sha256"]:
        raise SystemExit("REFUSED: tokenizer.json changed since freezing.")
    print(f"OK  {digest}")
    print(f"    {block['entries']} entries, {block['positions']} positions, reproduces exactly")


if __name__ == "__main__":
    if len(sys.argv) != 3 or sys.argv[1] not in ("freeze", "verify"):
        raise SystemExit(__doc__)
    (cmd_freeze if sys.argv[1] == "freeze" else cmd_verify)(sys.argv[2])
