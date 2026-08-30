"""The SENSITIVITY-1B' candidate pool — one definition, imported by both
the congruence control and the scorer.

Previously the screen's regions and the Q-BANK arms' `--protect` flags were
two independent encodings kept in step by hand, with nothing checking they
agreed. `region_congruence.py` now checks it; this module is what makes the
check meaningful, by giving the screen exactly one place to be wrong.

The pool is eight, not thirteen
-------------------------------
The capture taps attention-input, FFN-input and FFN-output. There is no
attention-output site, so `o_proj` has no moments and cannot be scored.
Every region containing `o_proj` is excluded rather than scored without it:
including one would omit `o_proj` from the numerator while its bytes still
sat in the per-MiB denominator, deflating exactly those regions.

This is also the pool 1B-a used, so 1B' differs from it in one variable —
the normalisation — and a pass isolates that as the cause.
"""

PROJECTIONS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)

# Regions with no moments: they contain o_proj. Kept named rather than
# silently absent, so the exclusion is visible at the point of use.
EXCLUDED = {
    "attn-protected": "contains o_proj",
    "o-protected": "contains o_proj",
    "late10-ffn-o": "contains o_proj",
    "early10": "all projections in a layer range, so contains o_proj",
    "late10": "all projections in a layer range, so contains o_proj",
}

# (projection, inclusive layer range or None for all layers)
POOL = {
    "late5-ffn": [("gate_proj", (35, 39)), ("up_proj", (35, 39)), ("down_proj", (35, 39))],
    "late10-ffn": [("gate_proj", (30, 39)), ("up_proj", (30, 39)), ("down_proj", (30, 39))],
    "late15-ffn": [("gate_proj", (25, 39)), ("up_proj", (25, 39)), ("down_proj", (25, 39))],
    "late10-ffn-v": [
        ("gate_proj", (30, 39)),
        ("up_proj", (30, 39)),
        ("down_proj", (30, 39)),
        ("v_proj", None),
    ],
    "ffn-protected": [("gate_proj", None), ("up_proj", None), ("down_proj", None)],
    "v-protected": [("v_proj", None)],
    "k-protected": [("k_proj", None)],
    "down-protected": [("down_proj", None)],
}

# The frozen negatives: Q-BANK found each buys nothing. A screen that ranks
# these highly has learned "protecting more bytes helps", which is true,
# useless, and what they exist to catch.
NEGATIVES = ("v-protected", "k-protected", "down-protected")

# The knee, in order. Used for the shape statistic.
DEPTH_LADDER = ("late5-ffn", "late10-ffn", "late15-ffn")

BASE_LABEL = "R0-recheck"


def projection_of(tensor):
    for p in PROJECTIONS:
        if p in tensor:
            return p
    return None


def layer_of(tensor):
    return int(tensor.split(".")[0])


def selects(tensor, rules):
    """Does `tensor` fall in a region defined by `rules`?"""
    proj, layer = projection_of(tensor), layer_of(tensor)
    return any(
        proj == p and (r is None or r[0] <= layer <= r[1]) for p, r in rules
    )


def tensors_for(label, inventory):
    """The tensor records a candidate protects, from a 1A-shaped inventory."""
    rules = POOL[label]
    return [t for t in inventory if selects(t["tensor"], rules)]


def extra_bytes(label, inventory):
    """Physical bytes the candidate adds over R0: source minus compiled."""
    return sum(
        t["source_bytes"] - t["compiled_bytes"] for t in tensors_for(label, inventory)
    )
