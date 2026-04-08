#!/usr/bin/env python3
"""
Phase 1: Expression Parser

Token sequence → ParsedExpression via pattern matching.
No NLU. The model has already done the understanding — we just parse
what it's about to compute from the generated token stream.

Supports:
  - Arithmetic: "6 * 7", "15% of 240", "log₂(1024)"
  - Equations: "solve x² + 3x - 1 = 0", "x + 5 = 12"
  - Systems: "x + y = 10 and 2x - y = 5"
  - Counting: "how many n ≤ 100 where n² mod 6 = 0"
  - Simplification: "simplify (x² - 1)/(x - 1)"
  - Derivatives/integrals: "derivative of x³ + 2x", "integral of sin(x)"
  - Combinatorics: "C(10,3)", "10!", "P(8,3)"
"""

import re
from dataclasses import dataclass, field
from typing import List, Optional, Union
from enum import Enum, auto


# ---------------------------------------------------------------------------
# Expression types
# ---------------------------------------------------------------------------

class ExprType(Enum):
    ARITHMETIC = auto()
    EQUATION = auto()
    SYSTEM = auto()
    COUNTING = auto()
    SIMPLIFY = auto()
    DERIVATIVE = auto()
    INTEGRAL = auto()
    COMBINATORICS = auto()
    PERCENTAGE = auto()
    COMPARISON = auto()


@dataclass
class ParsedExpression:
    expr_type: ExprType
    raw: str  # original matched text

@dataclass
class Arithmetic(ParsedExpression):
    expression: str  # evaluable expression string

    def __init__(self, raw: str, expression: str):
        super().__init__(ExprType.ARITHMETIC, raw)
        self.expression = expression

@dataclass
class Percentage(ParsedExpression):
    percent: str
    of_value: str

    def __init__(self, raw: str, percent: str, of_value: str):
        super().__init__(ExprType.PERCENTAGE, raw)
        self.percent = percent
        self.of_value = of_value

@dataclass
class Equation(ParsedExpression):
    lhs: str
    rhs: str
    variable: str

    def __init__(self, raw: str, lhs: str, rhs: str, variable: str = "x"):
        super().__init__(ExprType.EQUATION, raw)
        self.lhs = lhs
        self.rhs = rhs
        self.variable = variable

@dataclass
class System(ParsedExpression):
    equations: List[tuple]  # list of (lhs, rhs)
    variables: List[str]

    def __init__(self, raw: str, equations: List[tuple], variables: List[str]):
        super().__init__(ExprType.SYSTEM, raw)
        self.equations = equations
        self.variables = variables

@dataclass
class Counting(ParsedExpression):
    variable: str
    lower: int
    upper: int
    predicate: str  # sympy-parseable condition

    def __init__(self, raw: str, variable: str, lower: int, upper: int, predicate: str):
        super().__init__(ExprType.COUNTING, raw)
        self.variable = variable
        self.lower = lower
        self.upper = upper
        self.predicate = predicate

@dataclass
class Simplify(ParsedExpression):
    expression: str

    def __init__(self, raw: str, expression: str):
        super().__init__(ExprType.SIMPLIFY, raw)
        self.expression = expression

@dataclass
class Derivative(ParsedExpression):
    expression: str
    variable: str

    def __init__(self, raw: str, expression: str, variable: str = "x"):
        super().__init__(ExprType.DERIVATIVE, raw)
        self.expression = expression
        self.variable = variable

@dataclass
class Integral(ParsedExpression):
    expression: str
    variable: str

    def __init__(self, raw: str, expression: str, variable: str = "x"):
        super().__init__(ExprType.INTEGRAL, raw)
        self.expression = expression
        self.variable = variable

@dataclass
class Combinatorics(ParsedExpression):
    operation: str  # "choose", "permute", "factorial"
    n: int
    k: Optional[int] = None

    def __init__(self, raw: str, operation: str, n: int, k: Optional[int] = None):
        super().__init__(ExprType.COMBINATORICS, raw)
        self.operation = operation
        self.n = n
        self.k = k


# ---------------------------------------------------------------------------
# Normalisation: clean up text before pattern matching
# ---------------------------------------------------------------------------

def normalise(text: str) -> str:
    """Normalise unicode and common variants to parseable form."""
    text = text.replace("×", "*").replace("÷", "/")
    text = text.replace("−", "-").replace("–", "-")
    text = text.replace("²", "**2").replace("³", "**3")
    text = text.replace("⁴", "**4").replace("⁵", "**5")
    text = text.replace("√", "sqrt")
    text = text.replace("π", "pi")
    text = text.replace("≤", "<=").replace("≥", ">=")
    text = text.replace("≠", "!=")
    # log subscripts
    text = re.sub(r'log₂\(', 'log2(', text)
    text = re.sub(r'log₁₀\(', 'log10(', text)
    return text


# ---------------------------------------------------------------------------
# Pattern matchers — each returns Optional[ParsedExpression]
# ---------------------------------------------------------------------------

def try_percentage(text: str) -> Optional[Percentage]:
    """Match: "15% of 240", "what is 25% of 80"""
    m = re.search(
        r'(\d+(?:\.\d+)?)\s*%\s*(?:of)\s+(\d+(?:\.\d+)?)',
        text, re.IGNORECASE
    )
    if m:
        return Percentage(m.group(0), m.group(1), m.group(2))
    return None


def try_combinatorics(text: str) -> Optional[Combinatorics]:
    """Match: C(10,3), P(8,3), 10!, nCr(10,3)"""
    # Factorial
    m = re.search(r'(\d+)\s*!', text)
    if m:
        return Combinatorics(m.group(0), "factorial", int(m.group(1)))

    # C(n,k) or nCr(n,k) or "n choose k"
    m = re.search(r'(?:C|nCr|choose)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text, re.IGNORECASE)
    if m:
        return Combinatorics(m.group(0), "choose", int(m.group(1)), int(m.group(2)))

    m = re.search(r'(\d+)\s+choose\s+(\d+)', text, re.IGNORECASE)
    if m:
        return Combinatorics(m.group(0), "choose", int(m.group(1)), int(m.group(2)))

    # P(n,k) or nPr(n,k)
    m = re.search(r'(?:P|nPr)\s*\(\s*(\d+)\s*,\s*(\d+)\s*\)', text, re.IGNORECASE)
    if m:
        return Combinatorics(m.group(0), "permute", int(m.group(1)), int(m.group(2)))

    return None


def try_derivative(text: str) -> Optional[Derivative]:
    """Match: "derivative of x**3 + 2*x", "d/dx(x**2 + 1)" """
    m = re.search(
        r'(?:derivative|differentiate|diff)\s+(?:of\s+)?(.+?)(?:\s+with\s+respect\s+to\s+(\w))?$',
        text, re.IGNORECASE
    )
    if m:
        var = m.group(2) if m.group(2) else "x"
        return Derivative(m.group(0), m.group(1).strip(), var)

    m = re.search(r'd/d(\w)\s*\((.+?)\)', text)
    if m:
        return Derivative(m.group(0), m.group(2).strip(), m.group(1))

    return None


def try_integral(text: str) -> Optional[Integral]:
    """Match: "integral of sin(x)", "integrate x**2 dx" """
    m = re.search(
        r'(?:integral|integrate)\s+(?:of\s+)?(.+?)(?:\s+d(\w))?$',
        text, re.IGNORECASE
    )
    if m:
        var = m.group(2) if m.group(2) else "x"
        return Integral(m.group(0), m.group(1).strip(), var)
    return None


def try_simplify(text: str) -> Optional[Simplify]:
    """Match: "simplify (x**2 - 1)/(x - 1)" """
    m = re.search(r'simplify\s+(.+)', text, re.IGNORECASE)
    if m:
        return Simplify(m.group(0), m.group(1).strip())
    return None


def try_counting(text: str) -> Optional[Counting]:
    """Match: "how many n <= 100 where n**2 mod 6 = 0" """
    # Pattern: how many <var> [from/between] <lo> [to/and] <hi> [where/such that] <pred>
    m = re.search(
        r'how\s+many\s+(?:integers?\s+)?(\w)\s+'
        r'(?:from\s+(\d+)\s+to\s+(\d+)|'
        r'between\s+(\d+)\s+and\s+(\d+)|'
        r'(?:<=?\s*(\d+)(?:\s+and\s+>=?\s*(\d+))?)|'
        r'(?:>=?\s*(\d+)\s+and\s+<=?\s*(\d+)))'
        r'\s+(?:where|such\s+that|satisfy|with)\s+(.+)',
        text, re.IGNORECASE
    )
    if m:
        # Extract bounds from whichever group matched
        if m.group(2) and m.group(3):
            lo, hi = int(m.group(2)), int(m.group(3))
        elif m.group(4) and m.group(5):
            lo, hi = int(m.group(4)), int(m.group(5))
        elif m.group(6):
            lo = int(m.group(7)) if m.group(7) else 1
            hi = int(m.group(6))
        elif m.group(8) and m.group(9):
            lo, hi = int(m.group(8)), int(m.group(9))
        else:
            return None
        return Counting(m.group(0), m.group(1), lo, hi, m.group(10).strip())

    # Simpler: "count integers from 1 to 100 where ..."
    m = re.search(
        r'count\s+(?:integers?\s+)?(?:from\s+)?(\d+)\s+to\s+(\d+)\s+'
        r'(?:where|such\s+that)\s+(\w)\s+(.+)',
        text, re.IGNORECASE
    )
    if m:
        return Counting(m.group(0), m.group(3), int(m.group(1)), int(m.group(2)), m.group(4).strip())

    return None


def try_equation(text: str) -> Optional[Equation]:
    """Match: "solve x**2 + 3*x - 1 = 0", "x + 5 = 12", "find x: 2x = 10" """
    # "solve <expr> = <expr>"
    m = re.search(
        r'(?:solve|find\s+\w\s*:?)\s*(.+?)\s*=\s*(.+?)(?:\s*$|\s*,|\s*\.)',
        text, re.IGNORECASE
    )
    if m:
        lhs, rhs = m.group(1).strip(), m.group(2).strip()
        # Find the variable (single letter that's not a function name)
        variables = set(re.findall(r'\b([a-z])\b', lhs + rhs)) - {'e', 'i'}
        var = variables.pop() if variables else "x"
        return Equation(m.group(0), lhs, rhs, var)

    # Bare equation with "= ?" or "= " at end of generation
    # "x**2 + 3*x - 1 = 0"
    m = re.search(
        r'\b([a-z](?:\w*\*\*\d+|\w*\s*[+\-*/]\s*)+.+?)\s*=\s*(\d+(?:\.\d+)?)\s*$',
        text
    )
    if m:
        lhs = m.group(1).strip()
        rhs = m.group(2).strip()
        variables = set(re.findall(r'\b([a-z])\b', lhs)) - {'e', 'i'}
        var = variables.pop() if variables else "x"
        return Equation(m.group(0), lhs, rhs, var)

    return None


def try_system(text: str) -> Optional[System]:
    """Match: "x + y = 10 and 2x - y = 5" """
    # Look for multiple equations joined by "and" or ";"
    parts = re.split(r'\s+and\s+|;\s*', text)
    if len(parts) < 2:
        return None

    equations = []
    all_vars = set()
    for part in parts:
        m = re.search(r'(.+?)\s*=\s*(.+)', part.strip())
        if m:
            lhs, rhs = m.group(1).strip(), m.group(2).strip()
            equations.append((lhs, rhs))
            all_vars |= set(re.findall(r'\b([a-z])\b', lhs + rhs)) - {'e', 'i'}

    if len(equations) >= 2:
        return System(text, equations, sorted(all_vars))
    return None


def try_arithmetic(text: str) -> Optional[Arithmetic]:
    """Match: "6 * 7", "2 + 3 * 4", "sqrt(144)", "log2(1024)" """
    # Match expressions that look like pure arithmetic
    # Must contain at least one operator or function call
    m = re.search(
        r'(?:^|=\s*|is\s+)'
        r'((?:\d+(?:\.\d+)?|pi|e|sqrt|log\d*|sin|cos|tan|abs|round|floor|ceil)'
        r'[\s\d\.\+\-\*/\(\)\^,pielog2sincotan_absroundflceil]*'
        r'[\+\-\*/\^\(\)]'  # must have at least one operator or paren
        r'[\s\d\.\+\-\*/\(\)\^,pielog2sincotan_absroundflceil]*'
        r'(?:\d+(?:\.\d+)?|\)))',  # ends with number or close paren
        text
    )
    if m:
        expr = m.group(1).strip()
        # Validate it's actually arithmetic (no unresolved variables)
        remaining = re.sub(
            r'\b(?:sqrt|log\d*|sin|cos|tan|abs|round|floor|ceil|pi|e)\b',
            '', expr
        )
        if not re.search(r'[a-df-hj-oq-z]', remaining):  # allow 'e' and 'pi'
            return Arithmetic(m.group(0), expr)

    # Standalone expression at end: "= 6 * 7"
    m = re.search(r'(\d+(?:\.\d+)?(?:\s*[\+\-\*/\^]\s*\d+(?:\.\d+)?)+)', text)
    if m:
        return Arithmetic(m.group(0), m.group(1))

    return None


# ---------------------------------------------------------------------------
# Main parser: try all patterns in priority order
# ---------------------------------------------------------------------------

# Priority: specific patterns first, arithmetic last (most greedy)
PARSERS = [
    try_percentage,
    try_combinatorics,
    try_derivative,
    try_integral,
    try_simplify,
    try_counting,
    try_system,
    try_equation,
    try_arithmetic,
]


def parse(text: str) -> Optional[ParsedExpression]:
    """Parse a text string for computable expressions.

    Returns the first (highest-priority) match, or None.
    """
    normalised = normalise(text)
    for parser in PARSERS:
        result = parser(normalised)
        if result is not None:
            return result
    return None


def parse_all(text: str) -> List[ParsedExpression]:
    """Parse all computable expressions from text (for multi-step problems)."""
    normalised = normalise(text)
    results = []
    for parser in PARSERS:
        result = parser(normalised)
        if result is not None:
            results.append(result)
    return results


# ---------------------------------------------------------------------------
# Generation stream parser: detect expressions in token-by-token output
# ---------------------------------------------------------------------------

class StreamParser:
    """Monitors a token stream and detects computable expressions.

    Maintains a sliding window of recent tokens. When a trigger pattern
    is detected (e.g., "= " at end of arithmetic), it parses the window
    and returns the expression.
    """

    # Patterns that signal "the model is about to compute"
    TRIGGER_PATTERNS = [
        r'\d+\s*[\+\-\*/]\s*\d+\s*=\s*$',     # "6 * 7 = "
        r'=\s*\d+\s*[\+\-\*/]\s*\d+\s*$',      # "= 6 * 7"  (model computing)
        r'\d+\s*[\+\-\*/]\s*\d+\s*[\+\-\*/]',  # chained: "6 * 7 + 3 *"
        r'(?:solve|find)\s+.+=',                 # "solve x + 3 = "
        r'\d+\s*%\s*of\s+\d+',                  # "15% of 240"
        r'\d+\s*!\s*=?\s*$',                     # "10! = "
        r'(?:C|P)\(\d+,\s*\d+\)',               # "C(10,3)"
    ]

    def __init__(self, window_size: int = 200):
        self.window = ""
        self.window_size = window_size
        self._compiled = [re.compile(p, re.IGNORECASE) for p in self.TRIGGER_PATTERNS]

    def feed(self, token: str) -> Optional[ParsedExpression]:
        """Feed a token. Returns ParsedExpression if a computable pattern is detected."""
        self.window += token
        if len(self.window) > self.window_size:
            self.window = self.window[-self.window_size:]

        # Check triggers
        for pattern in self._compiled:
            if pattern.search(self.window):
                expr = parse(self.window)
                if expr is not None:
                    return expr
        return None

    def reset(self):
        self.window = ""


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        # Arithmetic
        ("6 * 7 = ", Arithmetic),
        ("2 + 3 * 4", Arithmetic),
        ("sqrt(144) + 1", Arithmetic),
        ("log2(1024)", Arithmetic),
        # Percentage
        ("15% of 240", Percentage),
        ("what is 25% of 80", Percentage),
        # Equations
        ("solve x**2 + 3*x - 1 = 0", Equation),
        ("find x: 2*x + 5 = 15", Equation),
        # Counting
        ("how many n <= 100 where n**2 mod 6 = 0", Counting),
        ("count 1 to 50 where n n % 7 == 0", Counting),
        # Simplify
        ("simplify (x**2 - 1)/(x - 1)", Simplify),
        # Derivative
        ("derivative of x**3 + 2*x", Derivative),
        ("d/dx(x**2 + 1)", Derivative),
        # Integral
        ("integral of sin(x)", Integral),
        # Combinatorics
        ("C(10, 3)", Combinatorics),
        ("10!", Combinatorics),
        ("5 choose 2", Combinatorics),
        # System
        ("x + y = 10 and 2*x - y = 5", System),
        # Non-computable (should return None)
        ("The capital of France is Paris", None),
        ("Tell me about quantum computing", None),
    ]

    print("Expression Parser — Self-test")
    print("=" * 60)
    passed = 0
    failed = 0
    for text, expected_type in tests:
        result = parse(text)
        actual_type = type(result) if result else None
        ok = (actual_type == expected_type)
        status = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        else:
            passed += 1
        print(f"  [{status}] {text!r}")
        if result:
            print(f"         → {result}")
        elif expected_type is not None:
            print(f"         → None (expected {expected_type.__name__})")

    print(f"\n{passed}/{passed + failed} passed")
