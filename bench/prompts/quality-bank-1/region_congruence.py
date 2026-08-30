#!/usr/bin/env python3
"""SENSITIVITY-1B' control: the screen and the truth must rank the same object.

    python3 region_congruence.py <sensitivity-1a.json> <sweep.json>

The screen sums consequence over a tensor set it derives from `candidates.POOL`.
Q-BANK measured arms that were compiled with `vindex3 represent --protect …`.
Those are two independent specifications of "the same" region, and until this
control existed nothing checked they agreed.

If they diverge, `late5-ffn` in the screen is a different set of tensors from
`late5-ffn` in the sweep, and the 1B' verdict — pass or fail — is a statement
about neither. That is not a small error: it is the gate measuring one thing
and licensing a claim about another.

The cross-check is physical bytes. For each candidate:

    screen  = sum(source_bytes - compiled_bytes) over the tensors it selects
    banked  = payload_bytes(arm) - payload_bytes(R0)

Both count the same quantity — bytes the arm keeps at source precision that
R0 compiled away — so agreement is exact, not approximate. A tolerance is
offered only to make a near-miss legible in the output; the gate is equality.

Exit status is the gate: 0 congruent, 1 refused.
"""
import json
import sys

import candidates as C

MIB = 2 ** 20
# Equality is the contract. Anything above this is reported as a mismatch;
# the value exists so a 1-byte drift prints as a number rather than as a
# wall of identical-looking figures.
TOLERANCE_BYTES = 0


def main(inventory_path, sweep_path):
    inventory = json.load(open(inventory_path))
    sweep = json.load(open(sweep_path))
    arms = {c["label"]: c for c in sweep["candidates"]}

    if C.BASE_LABEL not in arms:
        raise SystemExit(f"REFUSED: sweep has no base arm {C.BASE_LABEL!r}")
    base = arms[C.BASE_LABEL]["payload_bytes"]

    missing = [l for l in C.POOL if l not in arms]
    if missing:
        raise SystemExit(
            f"REFUSED: {len(missing)} pool candidate(s) have no banked arm: {missing}"
        )

    print(f"model {sweep.get('model')}  base {C.BASE_LABEL}  {base/MIB:.1f} MiB")
    print(f"{'candidate':16s} {'tensors':>8s} {'screen MiB':>11s} {'banked MiB':>11s} {'delta B':>10s}")

    failures = []
    for label in C.POOL:
        sel = C.tensors_for(label, inventory)
        screen = C.extra_bytes(label, inventory)
        banked = arms[label]["payload_bytes"] - base
        delta = banked - screen
        flag = "" if abs(delta) <= TOLERANCE_BYTES else "   <-- MISMATCH"
        print(
            f"{label:16s} {len(sel):8d} {screen/MIB:11.1f} {banked/MIB:11.1f} {delta:10d}{flag}"
        )
        if abs(delta) > TOLERANCE_BYTES:
            failures.append((label, screen, banked, delta))
        if not sel:
            failures.append((label, 0, banked, banked))

    print()
    for label, reason in sorted(C.EXCLUDED.items()):
        print(f"excluded  {label:16s} {reason}")

    if failures:
        print(f"\nREFUSED: {len(failures)} region(s) disagree with their banked arm.")
        for label, screen, banked, delta in failures:
            print(f"  {label}: screen {screen} B, banked {banked} B, delta {delta} B")
        print(
            "\nThe screen and the sweep are selecting different tensors. Scoring\n"
            "now would compare a prediction about one region against a verdict\n"
            "about another."
        )
        return 1

    print(f"\nCONGRUENT: all {len(C.POOL)} regions select exactly the tensors their")
    print("banked arm protected. The screen and the truth rank the same objects.")
    return 0


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    sys.exit(main(sys.argv[1], sys.argv[2]))
