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
        names_here = []
        try:
            text = Path(rf).read_text(encoding="utf-8")
        except OSError:
            continue
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(r, dict):
                continue
            name = r.get("level")
            if not name:
                continue
            lg = legs.setdefault(name, Leg(name=name))
            if name not in names_here:
                names_here.append(name)
            if r.get("type") == "meta":
                lg.meta = r
            else:
                lg.cells[r.get("id", "?")] = r
        # sidecars: only for legs introduced by THIS results file, from THIS dir
        for name in names_here:
            lg = legs[name]
            dp = d / f"descriptor-{name}.json"
            pp = d / f"produce-{name}.json"
            if dp.exists():
                lg.descriptor = _read_json(dp)
            if pp.exists():
                lg.produce = _read_json(pp)
    return legs


_FEAT = re.compile(r"\(\s*\d+\s+layers?,\s*([\d.]+)\s*([KM]?)\s+features", re.I)


def feature_count(leg):
    for row in leg.cells.values():
        m = _FEAT.search(row.get("stdout_head", "") or "")
        if m:
            try:
                n = float(m.group(1))
            except ValueError:
                continue
            n *= {"K": 1e3, "M": 1e6, "": 1}[m.group(2).upper()]
            return int(round(n))
    return None


def inv_completeness(legs):
    out = []
    for name, lg in legs.items():
        fc = feature_count(lg)
        if fc is None:
            out.append(Violation("completeness", name, "",
                                 "feature_count unknown (no STATS/banner — produce may have failed)"))
        elif fc == 0:
            out.append(Violation("completeness", name, "",
                                 "hollow vindex: 0 features (silent-incomplete extraction, cf #183)"))
    return out


_LAYER_ROW = re.compile(r"^L\d+\s+([\d.]+[KM]?)\s+[\d.]+[KM]?", re.M)

_CRASH_CODES = {101, 134, 137, 139}

_WARN = re.compile(r"warn|overrid|ignor", re.I)


def _is_crash(row):
    return row.get("bucket") == "crash" or row.get("exit_code") in _CRASH_CODES


def inv_no_crash(legs):
    out = []
    for name, lg in legs.items():
        if lg.produce and _is_crash(lg.produce):
            out.append(Violation("no-crash", name, "produce",
                                 f"produce crashed (exit {lg.produce.get('exit_code')})"))
        for cid, row in lg.cells.items():
            if _is_crash(row):
                out.append(Violation("no-crash", name, cid,
                                     f"panic/crash (exit {row.get('exit_code')})"))
    return out


def inv_descriptor_match(legs):
    out = []
    for name, lg in legs.items():
        d = lg.descriptor
        if not d or not d.get("produced", True) or d.get("error"):
            continue  # produce-side failure — covered by completeness/no-crash
        if not d.get("quant_match", True):
            out.append(Violation("descriptor-match", name, "",
                                 f"quant mismatch: produced {d.get('observed_quant')} != "
                                 f"expected {d.get('expect_quant')}"))
        fam = (d.get("family") or "").lower()
        if fam in ("", "generic"):
            out.append(Violation("descriptor-match", name, "",
                                 f"unrecognized/generic arch fallback (family={d.get('family')!r}, cf #154)"))
    return out


def _parse_num(tok):
    m = re.fullmatch(r"([\d.]+)([KM]?)", tok, re.I)
    if not m:
        return None
    try:
        n = float(m.group(1))
    except ValueError:
        return None
    return n * {"K": 1e3, "M": 1e6, "": 1}[m.group(2).upper()]


def show_layers_total(leg):
    row = leg.cells.get("show.layers")
    if not row:
        return None
    toks = _LAYER_ROW.findall(row.get("stdout_head", "") or "")
    if not toks:
        return None
    total, seen = 0.0, False
    for t in toks:
        v = _parse_num(t)
        if v is not None:
            total += v
            seen = True
    return int(round(total)) if seen else None


def inv_cross_check(legs):
    out = []
    for name, lg in legs.items():
        sl = show_layers_total(lg)
        fc = feature_count(lg)
        if sl is not None and fc and sl == 0:
            out.append(Violation("cross-check", name, "show.layers",
                                 f"SHOW LAYERS reports 0 features but STATS reports {fc} "
                                 "(mmap heap-only accessor, cf SHOW LAYERS bug)"))
    return out


def inv_diagnostic(legs):
    out = []
    for name, lg in legs.items():
        p = lg.produce
        # R-2: --quant q4k silently overrides a non-all --level
        if (p.get("op") == "extract" and "--quant q4k" in (p.get("flags") or "")
                and p.get("level") in {"browse", "attention", "inference"}):
            txt = (p.get("stdout_head", "") or "") + (p.get("stderr_head", "") or "")
            if not _WARN.search(txt):
                out.append(Violation("diagnostic", name, "produce",
                                     f"--level {p.get('level')} silently ignored under --quant q4k "
                                     "(no warn/override text; cf #208)"))
        # R-3: error text blames --compact but the recipe never passed it
        flags = (p.get("flags") or "")
        if "compact" not in flags:
            for cid, row in lg.cells.items():
                if "--compact" in (row.get("err_line", "") or ""):
                    out.append(Violation("diagnostic", name, cid,
                                         "error text misattributes to `--compact` (not in recipe flags)"))
                    break
    return out


INVARIANTS = [inv_completeness, inv_no_crash, inv_descriptor_match, inv_cross_check, inv_diagnostic]


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
