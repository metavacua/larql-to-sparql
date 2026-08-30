#!/usr/bin/env python3
"""Generate the long-context span fixture's token ids (V3-G5b-3-span).

The fixture exists to test one narrowly falsifiable question: at the
exact position where a persisted `Sliding(2048)` span first excludes a
key, does VINDEX3 do what upstream Glimmer does?

    python scripts/span_fixture_tokens.py --out ~/chris-models/glimmer-oracle/span_tokens.txt

**Why synthetic ids rather than real text.** This is a parity comparison
between two implementations over identical input, not a measurement of
model quality, so any non-degenerate sequence works — and a pseudo-random
one avoids the repetition a natural corpus of this length would carry,
which could let a masking bug hide behind similar keys.

**Why this file exists at all.** The ids *are* the fixture. Reproducing a
capture months later means reproducing them exactly, so the generator is
version-controlled rather than a shell one-liner that produced a file
nobody can regenerate.

Boundary arithmetic, for `window = 2048`, `start = max(0, p + 1 - window)`:

    p = 2046   sliding 0..=2046 (2047 keys)   full 0..=2046 (2047)
    p = 2047   sliding 0..=2047 (2048 keys)   full 0..=2047 (2048)
    p = 2048   sliding 1..=2048 (2048 keys)   full 0..=2048 (2049)  <- first exclusion
    p = 2049   sliding 2..=2049 (2048 keys)   full 0..=2049 (2050)
"""

import argparse
from pathlib import Path

# Knuth's MMIX linear congruential parameters — the same pair the Rust
# exec fixtures use (`lcg_values`), so one algorithm covers both sides of
# the project.
LCG_MULTIPLIER = 6364136223846793005
LCG_INCREMENT = 1442695040888963407
LCG_MODULUS = 2**64

# Seed: the date the fixture was cut. Arbitrary but recorded, which is
# the only property that matters.
DEFAULT_SEED = 20260813

# Four positions past the 2048 window so the first exclusion (2048) has
# unbound positions before it and excluded ones after it.
DEFAULT_COUNT = 2052

# Ids are drawn from the vocabulary's interior. Muse-Glimmer's specials
# sit at 200000+ and the low ids are reserved/byte-level, so staying
# inside this band keeps every position an ordinary token.
ID_FLOOR = 1000
ID_SPAN = 189000


def token_ids(count: int, seed: int) -> list[int]:
    """`count` ids from the pinned LCG, taking the high bits of each step."""
    state = seed
    ids = []
    for _ in range(count):
        state = (state * LCG_MULTIPLIER + LCG_INCREMENT) % LCG_MODULUS
        ids.append(ID_FLOOR + (state >> 33) % ID_SPAN)
    return ids


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--out", type=Path, required=True, help="token-id file to write")
    ap.add_argument("--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--window", type=int, default=2048,
                    help="window the fixture is sized against, for the printed summary")
    args = ap.parse_args()

    ids = token_ids(args.count, args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(",".join(map(str, ids)))

    first_excluded = args.window
    print(f"wrote {len(ids)} ids (seed {args.seed}, MMIX LCG) -> {args.out}")
    print(f"first position whose span truncates: {first_excluded}")
    print(f"last unbound position:               {first_excluded - 1}")


if __name__ == "__main__":
    main()
