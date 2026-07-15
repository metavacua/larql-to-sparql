#!/usr/bin/env bash
# Independent reference test — MUST NOT be edited by the agent under test.
# Exits 0 iff solution_scaffold/solution.py's add() matches every case below.
set -uo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python3 - "$DIR" <<'PYEOF'
import importlib.util, sys

task_dir = sys.argv[1]
spec = importlib.util.spec_from_file_location("solution", f"{task_dir}/solution_scaffold/solution.py")
mod = importlib.util.module_from_spec(spec)
try:
    spec.loader.exec_module(mod)
except Exception as e:
    print(f"FAIL: solution.py failed to import/execute: {e}")
    sys.exit(1)

cases = [((2, 3), 5), ((-1, 1), 0), ((0, 0), 0), ((100, 250), 350), ((-5, -7), -12)]
failures = 0
for (a, b), expected in cases:
    try:
        got = mod.add(a, b)
    except Exception as e:
        print(f"FAIL add({a}, {b}): raised {e!r}, expected {expected}")
        failures += 1
        continue
    if got != expected:
        print(f"FAIL add({a}, {b}) = {got}, expected {expected}")
        failures += 1
    else:
        print(f"PASS add({a}, {b}) = {got}")

sys.exit(1 if failures else 0)
PYEOF
