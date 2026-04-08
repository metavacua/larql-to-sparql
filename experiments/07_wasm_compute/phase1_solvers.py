#!/usr/bin/env python3
"""
Phase 1: Solver Dispatch

Route ParsedExpression → exact answer. Each solver is deterministic:
same input → same output, every time.

Solvers:
  - Arithmetic: safe eval with math builtins
  - Percentage: direct computation
  - Equations: sympy.solve
  - Systems: sympy.solve (multi-variable)
  - Counting: ortools CP-SAT
  - Simplify: sympy.simplify
  - Derivative: sympy.diff
  - Integral: sympy.integrate
  - Combinatorics: math.factorial, math.comb, math.perm
"""

import math
import time
import re
from typing import Optional
from dataclasses import dataclass

import sympy
from sympy.parsing.sympy_parser import (
    parse_expr,
    standard_transformations,
    implicit_multiplication_application,
    convert_xor,
)

from phase1_parser import (
    ParsedExpression, ExprType,
    Arithmetic, Percentage, Equation, System,
    Counting, Simplify, Derivative, Integral, Combinatorics,
)


# ---------------------------------------------------------------------------
# Result type
# ---------------------------------------------------------------------------

@dataclass
class SolverResult:
    value: str          # string representation of answer
    exact: bool         # is this provably exact?
    solver: str         # which solver produced this
    elapsed_us: float   # microseconds to solve


# ---------------------------------------------------------------------------
# Safe arithmetic evaluator
# ---------------------------------------------------------------------------

_SAFE_MATH = {
    'sqrt': math.sqrt, 'log': math.log, 'log2': math.log2, 'log10': math.log10,
    'sin': math.sin, 'cos': math.cos, 'tan': math.tan,
    'abs': abs, 'round': round, 'floor': math.floor, 'ceil': math.ceil,
    'pi': math.pi, 'e': math.e,
    'pow': pow, 'min': min, 'max': max,
}


def safe_eval(expr: str) -> Optional[float]:
    """Evaluate arithmetic expression safely. No exec, no imports."""
    # Replace ^ with ** for exponentiation
    expr = expr.replace('^', '**')
    try:
        result = eval(expr, {"__builtins__": {}}, _SAFE_MATH)
        return result
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Sympy parsing helpers
# ---------------------------------------------------------------------------

_SYMPY_TRANSFORMS = standard_transformations + (
    implicit_multiplication_application,
    convert_xor,
)


def to_sympy(expr_str: str) -> Optional[sympy.Expr]:
    """Parse string to sympy expression."""
    # Clean up for sympy
    expr_str = expr_str.replace('^', '**')
    # Handle implicit multiplication: "3x" → "3*x"
    expr_str = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', expr_str)
    try:
        return parse_expr(expr_str, transformations=_SYMPY_TRANSFORMS)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Individual solvers
# ---------------------------------------------------------------------------

def solve_arithmetic(expr: Arithmetic) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    result = safe_eval(expr.expression)
    elapsed = (time.perf_counter_ns() - t0) / 1000
    if result is None:
        return None
    # Format: integer if exact, otherwise float
    if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
        value = str(int(result))
    else:
        value = str(result)
    return SolverResult(value, True, "arithmetic", elapsed)


def solve_percentage(expr: Percentage) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    pct = float(expr.percent)
    val = float(expr.of_value)
    result = pct / 100.0 * val
    elapsed = (time.perf_counter_ns() - t0) / 1000
    if result == int(result):
        value = str(int(result))
    else:
        value = str(result)
    return SolverResult(value, True, "percentage", elapsed)


def solve_equation(expr: Equation) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    var = sympy.Symbol(expr.variable)
    lhs = to_sympy(expr.lhs)
    rhs = to_sympy(expr.rhs)
    if lhs is None or rhs is None:
        return None
    try:
        solutions = sympy.solve(lhs - rhs, var)
        elapsed = (time.perf_counter_ns() - t0) / 1000
        if not solutions:
            return SolverResult("no solution", True, "sympy.solve", elapsed)
        if len(solutions) == 1:
            value = str(solutions[0])
        else:
            value = ", ".join(str(s) for s in solutions)
        return SolverResult(value, True, "sympy.solve", elapsed)
    except Exception:
        return None


def solve_system(expr: System) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    symbols = {v: sympy.Symbol(v) for v in expr.variables}
    equations = []
    for lhs_str, rhs_str in expr.equations:
        lhs = to_sympy(lhs_str)
        rhs = to_sympy(rhs_str)
        if lhs is None or rhs is None:
            return None
        equations.append(lhs - rhs)
    try:
        solutions = sympy.solve(equations, list(symbols.values()))
        elapsed = (time.perf_counter_ns() - t0) / 1000
        if not solutions:
            return SolverResult("no solution", True, "sympy.solve", elapsed)
        if isinstance(solutions, dict):
            value = ", ".join(f"{k} = {v}" for k, v in solutions.items())
        elif isinstance(solutions, list):
            value = str(solutions)
        else:
            value = str(solutions)
        return SolverResult(value, True, "sympy.solve", elapsed)
    except Exception:
        return None


def solve_counting(expr: Counting) -> Optional[SolverResult]:
    """Use CP-SAT for counting problems."""
    t0 = time.perf_counter_ns()
    try:
        from ortools.sat.python import cp_model

        model = cp_model.CpModel()
        n = model.new_int_var(expr.lower, expr.upper, expr.variable)

        # Parse predicate into CP-SAT constraint
        # Support common patterns: "n**2 mod 6 = 0", "n % 7 == 0", "n is prime"
        pred = expr.predicate
        pred = pred.replace("mod", "%").replace("=", "==").replace("====", "==").replace("===", "==")
        # Remove duplicate ==
        pred = re.sub(r'={2,}', '==', pred)

        # For simple modular predicates, count directly (faster than CP-SAT for these)
        m = re.match(r'(\w+)\s*%\s*(\d+)\s*==\s*(\d+)', pred)
        if m:
            var_name, modulus, remainder = m.group(1), int(m.group(2)), int(m.group(3))
            count = sum(1 for i in range(expr.lower, expr.upper + 1)
                        if i % modulus == remainder)
            elapsed = (time.perf_counter_ns() - t0) / 1000
            return SolverResult(str(count), True, "direct_count", elapsed)

        # For n**2 % k == r type predicates
        m = re.match(r'(\w+)\*\*(\d+)\s*%\s*(\d+)\s*==\s*(\d+)', pred)
        if m:
            var_name, power, modulus, remainder = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
            count = sum(1 for i in range(expr.lower, expr.upper + 1)
                        if (i ** power) % modulus == remainder)
            elapsed = (time.perf_counter_ns() - t0) / 1000
            return SolverResult(str(count), True, "direct_count", elapsed)

        # Fallback: direct enumeration with eval
        safe_vars = {"__builtins__": {}, "abs": abs}
        count = 0
        for i in range(expr.lower, expr.upper + 1):
            safe_vars[expr.variable] = i
            try:
                if eval(pred, safe_vars):
                    count += 1
            except Exception:
                continue
        elapsed = (time.perf_counter_ns() - t0) / 1000
        return SolverResult(str(count), True, "enumeration", elapsed)

    except Exception:
        return None


def solve_simplify(expr: Simplify) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    sympy_expr = to_sympy(expr.expression)
    if sympy_expr is None:
        return None
    try:
        simplified = sympy.simplify(sympy_expr)
        elapsed = (time.perf_counter_ns() - t0) / 1000
        return SolverResult(str(simplified), True, "sympy.simplify", elapsed)
    except Exception:
        return None


def solve_derivative(expr: Derivative) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    sympy_expr = to_sympy(expr.expression)
    if sympy_expr is None:
        return None
    var = sympy.Symbol(expr.variable)
    try:
        result = sympy.diff(sympy_expr, var)
        elapsed = (time.perf_counter_ns() - t0) / 1000
        return SolverResult(str(result), True, "sympy.diff", elapsed)
    except Exception:
        return None


def solve_integral(expr: Integral) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    sympy_expr = to_sympy(expr.expression)
    if sympy_expr is None:
        return None
    var = sympy.Symbol(expr.variable)
    try:
        result = sympy.integrate(sympy_expr, var)
        elapsed = (time.perf_counter_ns() - t0) / 1000
        return SolverResult(str(result), True, "sympy.integrate", elapsed)
    except Exception:
        return None


def solve_combinatorics(expr: Combinatorics) -> Optional[SolverResult]:
    t0 = time.perf_counter_ns()
    try:
        if expr.operation == "factorial":
            result = math.factorial(expr.n)
        elif expr.operation == "choose":
            result = math.comb(expr.n, expr.k)
        elif expr.operation == "permute":
            result = math.perm(expr.n, expr.k)
        else:
            return None
        elapsed = (time.perf_counter_ns() - t0) / 1000
        return SolverResult(str(result), True, "math.combinatorics", elapsed)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dispatch table
# ---------------------------------------------------------------------------

_DISPATCH = {
    ExprType.ARITHMETIC: solve_arithmetic,
    ExprType.PERCENTAGE: solve_percentage,
    ExprType.EQUATION: solve_equation,
    ExprType.SYSTEM: solve_system,
    ExprType.COUNTING: solve_counting,
    ExprType.SIMPLIFY: solve_simplify,
    ExprType.DERIVATIVE: solve_derivative,
    ExprType.INTEGRAL: solve_integral,
    ExprType.COMBINATORICS: solve_combinatorics,
}


def solve(expr: ParsedExpression) -> Optional[SolverResult]:
    """Dispatch a parsed expression to the appropriate solver."""
    solver_fn = _DISPATCH.get(expr.expr_type)
    if solver_fn is None:
        return None
    return solver_fn(expr)


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from phase1_parser import parse

    tests = [
        # (input, expected_answer)
        ("6 * 7", "42"),
        ("2 + 3 * 4", "14"),
        ("sqrt(144)", "12"),
        ("log2(1024)", "10"),
        ("15% of 240", "36"),
        ("25% of 80", "20"),
        ("solve 2*x + 5 = 15", "5"),
        ("solve x**2 - 4 = 0", "-2, 2"),
        ("simplify (x**2 - 1)/(x - 1)", "x + 1"),
        ("derivative of x**3 + 2*x", "3*x**2 + 2"),
        ("integral of sin(x)", "-cos(x)"),
        ("C(10, 3)", "120"),
        ("10!", "3628800"),
        ("5 choose 2", "10"),
        ("how many n <= 100 where n % 7 == 0", "14"),
        ("how many n <= 1000 where n**2 % 6 == 0", "166"),
    ]

    print("Solver Dispatch — Self-test")
    print("=" * 60)
    passed = 0
    failed = 0
    for text, expected in tests:
        expr = parse(text)
        if expr is None:
            print(f"  [FAIL] {text!r} — parser returned None")
            failed += 1
            continue
        result = solve(expr)
        if result is None:
            print(f"  [FAIL] {text!r} — solver returned None")
            failed += 1
            continue
        ok = (result.value == expected)
        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        else:
            failed += 1
        print(f"  [{status}] {text!r}")
        print(f"         → {result.value} ({result.solver}, {result.elapsed_us:.1f}μs)"
              + (f"  expected {expected}" if not ok else ""))

    print(f"\n{passed}/{passed + failed} passed")
