"""Conformance oracle for the lql-strategy-matrix. Evaluates principled,
no-expected-output invariants over already-captured artifacts and emits a
report. Default exit 0 (discovery preserved); non-zero only under --strict."""
import glob
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Leg:
    name: str
    meta: dict = field(default_factory=dict)
    cells: dict = field(default_factory=dict)       # id -> row
    descriptor: dict = field(default_factory=dict)
    produce: dict = field(default_factory=dict)


@dataclass
class Violation:
    invariant: str
    leg: str
    cell: str
    detail: str


def _read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}


def load(results_glob):
    legs = {}
    for rf in sorted(glob.glob(results_glob)):
        d = Path(rf).parent
        for line in Path(rf).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = r.get("level")
            if not name:
                continue
            lg = legs.setdefault(name, Leg(name=name))
            if r.get("type") == "meta":
                lg.meta = r
            else:
                lg.cells[r.get("id", "?")] = r
        # sidecars: descriptor-<name>.json / produce-<name>.json in the same dir
        for name, lg in legs.items():
            dp = d / f"descriptor-{name}.json"
            pp = d / f"produce-{name}.json"
            if dp.exists():
                lg.descriptor = _read_json(dp)
            if pp.exists():
                lg.produce = _read_json(pp)
    return legs


INVARIANTS = []


def run(results_glob, out_md, out_json, strict):
    legs = load(results_glob)
    violations = [v for inv in INVARIANTS for v in inv(legs)]
    Path(out_json).write_text(json.dumps(
        {"violations": [v.__dict__ for v in violations]}, indent=2), encoding="utf-8")
    Path(out_md).write_text(render(legs, violations), encoding="utf-8")
    print(f"conformance: {len(violations)} violation(s) across {len(legs)} legs "
          f"(strict={strict})")
    return 1 if (strict and violations) else 0


def render(legs, violations):
    L = ["# LQL Matrix — Conformance", "",
         f"Legs: {len(legs)} · Violations: {len(violations)}", ""]
    if violations:
        L += ["| invariant | leg | cell | detail |", "|---|---|---|---|"]
        for v in violations:
            L.append(f"| {v.invariant} | `{v.leg}` | {v.cell or '-'} | {v.detail} |")
    else:
        L.append("No invariant violations.")
    return "\n".join(L) + "\n"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    strict = "--strict" in sys.argv[1:]
    results_glob = args[0] if args else "artifacts/results-*/results-*.jsonl"
    out_md = args[1] if len(args) > 1 else "conformance.md"
    out_json = args[2] if len(args) > 2 else "conformance.json"
    sys.exit(run(results_glob, out_md, out_json, strict))


if __name__ == "__main__":
    main()
