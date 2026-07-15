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

cases = [("abc", "cba"), ("", ""), ("a", "a"), ("racecar", "racecar"), ("hello world", "dlrow olleh")]
failures = 0
for s, expected in cases:
    try:
        got = mod.reverse_string(s)
    except Exception as e:
        print(f"FAIL reverse_string({s!r}): raised {e!r}, expected {expected!r}")
        failures += 1
        continue
    if got != expected:
        print(f"FAIL reverse_string({s!r}) = {got!r}, expected {expected!r}")
        failures += 1
    else:
        print(f"PASS reverse_string({s!r}) = {got!r}")

sys.exit(1 if failures else 0)
PYEOF
