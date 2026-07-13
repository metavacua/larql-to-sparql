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
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load(results_glob):
    legs = {}
    for rf in sorted(glob.glob(results_glob)):
        d = Path(rf).parent
        names_here = []
        try:
            # errors="replace": a malformed-UTF-8 artifact (plausible from a corrupt or
            # OOM-truncated produce) must not crash the checker — the worst inputs are
            # exactly the ones the oracle exists to report on. except Exception guards
            # UnicodeDecodeError (a ValueError, not OSError) and any other read failure.
            text = Path(rf).read_text(encoding="utf-8", errors="replace")
        except Exception:
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
            ep = d / f"produce-{name}.err"
            if dp.exists():
                lg.descriptor = _read_json(dp)
            if pp.exists():
                lg.produce = _read_json(pp)
            if ep.exists():
                try:
                    lg.produce["stderr_head"] = ep.read_text(
                        encoding="utf-8", errors="replace")[:800]
                except OSError:
                    pass
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
        if _produce_failed(lg):
            continue  # vindex was never produced → inv_produce, not a hollow vindex
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


def _produce_failed(lg):
    """True iff this leg's vindex was not successfully produced — the definitive
    signal (descriptor.produced=False, or a non-zero/err/timeout produce). Distinct
    from a produced-but-hollow vindex (feat=0), which is a completeness violation."""
    if lg.descriptor.get("produced") is False:
        return True
    p = lg.produce or {}
    ec = p.get("exit_code")
    if isinstance(ec, int) and ec != 0:
        return True
    return p.get("bucket") in {"err", "crash", "timeout"}


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


def inv_produce(legs):
    """Definitive produce-failure violation: the vindex was never created. Produce
    *crashes* (SIGKILL/panic) are already reported by inv_no_crash, so skip them here
    to avoid double-counting; this catches the non-crash failures (exit!=0 / err /
    timeout / descriptor.produced=False) that would otherwise be mislabeled as a
    hedged completeness-unknown."""
    out = []
    for name, lg in legs.items():
        if _is_crash(lg.produce):
            continue
        if _produce_failed(lg):
            p = lg.produce or {}
            out.append(Violation("produce", name, "produce",
                                 f"produce failed: op={p.get('op')} exit={p.get('exit_code')} "
                                 f"bucket={p.get('bucket')} (vindex not created; "
                                 f"descriptor.produced={lg.descriptor.get('produced')})"))
    return out


def inv_masked_error(legs):
    """Every non-crash cell that surfaced an error. larql lql exits 0 on all in-band
    errors (repl.rs run_batch swallows them), so without this the CI is blind to them.
    There is NO 'expected error' exemption — every surfaced error is a violation.
    Crashes are owned by inv_no_crash and skipped here to avoid double-counting."""
    out = []
    for name, lg in legs.items():
        for cid, row in lg.cells.items():
            if row.get("err_signal") and not _is_crash(row):
                detail = (row.get("err_line") or "error surfaced with exit 0").strip()[:120]
                out.append(Violation("masked-error", name, cid, detail))
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


INVARIANTS = [inv_completeness, inv_produce, inv_no_crash, inv_masked_error,
              inv_descriptor_match, inv_cross_check, inv_diagnostic]


_SRC_FMT = {"extract": "safetensors", "gguf-to-vindex": "gguf",
            "quantize-q4k": "safetensors", "quantize-fp4": "safetensors"}


def transformation_report(legs):
    L = ["## Transformation & consumability coverage", "",
         "| leg | source fmt | produced arch | produced quant |",
         "|---|---|---|---|"]
    for name in sorted(legs):
        lg = legs[name]
        src = _SRC_FMT.get(lg.produce.get("op"), "?")
        L.append(f"| `{name}` | {src} | {lg.descriptor.get('family','?')} | "
                 f"{lg.descriptor.get('observed_quant','?')} |")
    L += ["",
          "**Consumer reachability & gaps (static):**",
          "- larql-cli / larql-server (`/v1/chat`): reach any produced vindex directly.",
          "- **Ollama:** needs GGUF; larql `compile` emits safetensors and there is **no "
          "compiled→GGUF export** — Ollama is a **gap** (cf #181/N11).",
          "- **Cross-arch transform** (e.g. qwen↔bitnet): produced arch equals source arch in "
          "every leg — larql changes quant/format, **not architecture** — a **gap** for the "
          "'transform one arch into another' use case.", ""]
    return L


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
    L += [""] + transformation_report(legs)
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
