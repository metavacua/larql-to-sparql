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

cases = [(1, 1), (3, 6), (5, 15), (10, 55), (0, 0)]
failures = 0
for n, expected in cases:
    try:
        got = mod.sum_to_n(n)
    except Exception as e:
        print(f"FAIL sum_to_n({n}): raised {e!r}, expected {expected}")
        failures += 1
        continue
    if got != expected:
        print(f"FAIL sum_to_n({n}) = {got}, expected {expected}")
        failures += 1
    else:
        print(f"PASS sum_to_n({n}) = {got}")

sys.exit(1 if failures else 0)
PYEOF
