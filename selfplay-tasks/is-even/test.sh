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

cases = [(0, True), (1, False), (2, True), (-4, True), (-7, False), (100, True)]
failures = 0
for n, expected in cases:
    try:
        got = mod.is_even(n)
    except Exception as e:
        print(f"FAIL is_even({n}): raised {e!r}, expected {expected}")
        failures += 1
        continue
    if bool(got) != expected:
        print(f"FAIL is_even({n}) = {got}, expected {expected}")
        failures += 1
    else:
        print(f"PASS is_even({n}) = {got}")

sys.exit(1 if failures else 0)
PYEOF
