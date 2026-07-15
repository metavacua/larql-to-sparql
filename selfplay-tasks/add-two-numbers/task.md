# Task: add-two-numbers

Implement `add(a: int, b: int) -> int` in `solution_scaffold/solution.py` so it returns the sum
of `a` and `b`.

Scope note: deliberately trivial — this suite is scoped under SmolLM2-135M's actual capability
ceiling (residual Q2), not a standard benchmark difficulty. The point is a real, runnable,
independently-scored pass/fail signal, not task difficulty.

Run `./test.sh` to check your solution. It exits 0 on success, non-zero on failure, and never
consults the model/agent's own opinion of correctness — only the reference input/output pairs
below (residual C4 / design doc ADR-4: the evaluator MUST NOT share state with the actor).
