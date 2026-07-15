#!/usr/bin/env bash
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

cases = [((2, 3), 3), ((5, 1), 5), ((-1, -1), -1), ((0, -10), 0), ((42, 42), 42)]
failures = 0
for (a, b), expected in cases:
    try:
        got = mod.max_of_two(a, b)
    except Exception as e:
        print(f"FAIL max_of_two({a}, {b}): raised {e!r}, expected {expected}")
        failures += 1
        continue
    if got != expected:
        print(f"FAIL max_of_two({a}, {b}) = {got}, expected {expected}")
        failures += 1
    else:
        print(f"PASS max_of_two({a}, {b}) = {got}")

sys.exit(1 if failures else 0)
PYEOF
