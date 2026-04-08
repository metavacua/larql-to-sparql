# Experiment 07: Native Compute Engine (WASM Solvers in Inference)

**Chris Hay | LARQL Project | April 2026**

---

## Thesis

A language model's accuracy on formal/mathematical problems is bounded by its ability to do exact computation. Embedding deterministic solvers directly in the inference path — not as external tool calls — eliminates that bound for any problem with a formal specification.

## Research Questions

| # | Question | Phase | Gate |
|---|----------|-------|------|
| Q1 | Does adding exact solvers to token generation improve math benchmark accuracy? | 1 | If no improvement → stop |
| Q2 | Can the residual stream encode enough structure to extract a formal problem spec? | 2 | Requires Q1 = yes |
| Q3 | Can solver output be injected back into the residual stream for subsequent layers to decode? | 2 | Requires Q2 = yes |

---

## Phase 1: Token-Level Compute Engine

### Goal

Prove the value proposition. Intercept at the token level during generation, detect computable expressions, solve them exactly, inject answer tokens. Measure accuracy gain on standard benchmarks.

### Architecture

```
Gemma 3-4B generating tokens normally
  ↓
Token stream monitor (sliding window)
  ↓
Pattern detector: is this a computable expression?
  ├── No  → continue generation
  └── Yes → parse expression → dispatch to solver
                                    ↓
                              Exact result
                                    ↓
                              Inject answer tokens → continue generation
```

### What to Build

**1. Expression parser** — extract structured math from token sequences.

```python
# Input: token sequence from model generation
"There are 6 * 7 = "          → Arithmetic(6 * 7)
"Solving x² + 3x - 1 = 0:"   → Equation(x**2 + 3*x - 1, 0)
"How many n ≤ 100 where..."   → Constraint(n, range(1,101), predicate)
"What is 15% of 240?"         → Arithmetic(0.15 * 240)
"log₂(1024) = "               → Arithmetic(log2(1024))
```

Not NLU. Pattern matching on token sequences. The model has already done the language understanding — it's generating the computation. We just need to parse what it's about to compute.

**2. Solver dispatch** — route parsed expressions to exact solvers.

```python
# Phase 1: pure Python, no WASM yet
# The point is to prove the pipeline, not the runtime

def solve(expr: ParsedExpression) -> str:
    match expr:
        case Arithmetic(op):    return str(eval_safe(op))
        case Equation(eq):      return sympy.solve(eq)
        case Constraint(spec):  return cp_sat_count(spec)
        case Simplify(expr):    return str(sympy.simplify(expr))
        case _:                 return None  # fall through to neural
```

**3. Token injector** — replace model's next-token prediction with solver output.

When the solver returns a result, tokenise it and force those tokens into the output stream. The model continues generating from after the injected tokens.

### Benchmarks

| Benchmark | What it tests | Baseline (Gemma 3-4B) | Expected w/ solver |
|-----------|--------------|----------------------|-------------------|
| GSM8K | Grade school math (8.5K problems) | ~60-65% | 80%+ (arithmetic errors eliminated) |
| MATH | Competition math (12.5K problems) | ~30-35% | 50%+ (algebra/counting boosted) |
| SVAMP | Arithmetic word problems | ~70% | 90%+ (pure computation) |
| AQuA-RAT | Algebraic word problems | ~35% | 55%+ (equation solving) |

The hypothesis is not "solvers solve everything" — it's "solvers eliminate the computation errors that the model makes when it has correctly understood the problem." The model still does the language understanding and problem formulation. The solver just does the arithmetic correctly.

### Metrics

- **Accuracy lift**: % correct with solver vs without, per benchmark
- **Intervention rate**: what fraction of problems trigger solver dispatch?
- **False positive rate**: solver dispatched but shouldn't have been (wrong parse)
- **False negative rate**: problem was computable but solver wasn't triggered
- **Latency overhead**: ms added per token by the detection + solve pipeline

### Success Criteria

Phase 1 succeeds if:
1. Accuracy improves by **>5%** on at least one benchmark
2. False positive rate is **<2%** (solver doesn't corrupt correct generations)
3. Latency overhead is **<5ms per token** (negligible vs generation time)

### Implementation

```
experiments/07_wasm_compute/
  phase1_parser.py        # token sequence → ParsedExpression
  phase1_solvers.py       # dispatch table: expression → exact answer
  phase1_pipeline.py      # end-to-end: model + solver integration
  phase1_bench.py         # run benchmarks, compare baseline vs augmented
  results_phase1/         # accuracy tables, error analysis, latency
```

**Day 1:** Parser + solvers (pure Python, sympy, ortools)
**Day 2:** Pipeline integration with Gemma 3-4B generation
**Day 3:** Benchmark runs + error analysis

---

## Phase 2: Residual-Level Dispatch (Research)

### Goal

Answer the research question: can you move the compute dispatch from token-level to residual-level? This eliminates the sequential overhead (generate → detect → solve → resume) and makes the solver truly native to the forward pass.

### Prerequisites

- Phase 1 shows >5% accuracy gain (proves the solvers add value)
- Phase 1 error analysis shows the parser catches >70% of computable problems

### Sub-experiments

#### 2a: Residual Classification (Is this a math problem?)

Binary probe on residual vectors. At what layer does the model "know" it's doing math?

```python
# Collect residuals from GSM8K prompts at each layer
# Train linear probe: residual → {math_problem, not_math_problem}
# Measure accuracy per layer

# Hypothesis: classification should be easy by L6-L10
# (the model routes to math-specific FFN features early)
```

This is classification, not extraction. We already know classification works from routing experiments (44 sub-centroids, 62% zero-error coverage). The question is whether "math problem" is a classifiable category in the residual.

#### 2b: Structured Extraction (What equation?)

Given that L_k's residual encodes "this is a math problem," can we extract the actual equation?

```python
# Train a small decoder: residual → structured spec
# Input: residual vector at layer L_k (2048-dim for Gemma 3-4B)
# Output: structured representation (operator, operands, variables)
#
# Start simple: arithmetic only
#   residual → (op: "multiply", a: 6, b: 7)
#
# Then: single-variable equations
#   residual → (lhs: "x**2 + 3*x - 1", rhs: "0", var: "x")
```

This is the hard part. The residual encodes the problem implicitly — as the state needed to generate the right tokens. Whether that state is structured enough to parse as a formal spec is unknown.

**Key insight from existing work:** The residual at L13 encodes query type (hourglass bottleneck). The correction vectors encode entity. If the residual separates "type of computation" from "operands," then extraction might decompose into:
1. Classify computation type (proven feasible)
2. Extract operands (unknown — this is the research question)

#### 2c: Result Injection (Put the answer back)

Given a solver result, can we construct a residual vector that causes the model to output the correct tokens?

```python
# Approach 1: Mean residual matching
#   Find the mean residual for "= 42" across many examples
#   Inject that pattern. Does the model output "42"?
#
# Approach 2: Linear steering
#   Learn a linear map: (result_tokens) → residual_delta
#   Add delta to current residual. Does output shift to result?
#
# Approach 3: Bypass
#   Don't inject into residual at all.
#   Use the solver result to constrain the output distribution directly.
#   (Guided decoding — but from residual-level detection)
```

Approach 3 is the pragmatic middle ground: detect at residual level (faster than token parsing), solve, then constrain output tokens (simpler than residual injection). This gives you most of the latency benefit without the hardest part of the research.

### Metrics

- **Classification accuracy**: can we detect "math problem" from residual? At what layer?
- **Extraction accuracy**: given "math problem" residual, can we recover the equation? What error rate?
- **Injection fidelity**: given solver result, does the model output it correctly?
- **End-to-end**: residual dispatch accuracy vs token-level dispatch accuracy

### Implementation

```
experiments/07_wasm_compute/
  phase2a_classify.py     # residual → is_math probe
  phase2b_extract.py      # residual → structured equation
  phase2c_inject.py       # solver result → residual / constrained output
  phase2_e2e.py           # full residual-level pipeline
  results_phase2/
```

---

## Phase 3: WASM Runtime (Engineering, conditional on Phase 2)

Only if Phase 2 shows residual-level dispatch is viable.

### What to Build (Rust)

```rust
// In larql-compute or new larql-wasm crate

/// WASM solver module interface
trait WasmSolver {
    fn solve(&self, input: &[u8]) -> Result<Vec<u8>>;
    fn problem_type(&self) -> ProblemType;
}

/// Runtime that manages solver modules
struct SolverRuntime {
    engine: wasmtime::Engine,
    modules: HashMap<ProblemType, wasmtime::Instance>,
}

impl SolverRuntime {
    fn dispatch(&self, problem: &ComputeDispatch) -> Option<SolverResult> {
        let module = self.modules.get(&problem.problem_type())?;
        let input = problem.serialize();
        let output = module.call("solve", &input)?;
        Some(SolverResult::deserialize(&output))
    }
}
```

### Solver Modules to Compile

| Solver | Source | WASM size (est.) | What it handles |
|--------|--------|------------------|-----------------|
| arithmetic | Rust native | 0 (built-in) | +, -, *, /, %, pow, sqrt, log |
| algebra | sympy (via RustPython or Rust CAS) | ~2MB | solve, simplify, differentiate, integrate |
| constraint | or-tools CP-SAT (C++ → WASM) | ~5MB | counting, optimization, scheduling |
| logic | z3 (C++ → WASM) | ~10MB | SAT, SMT, formal verification |
| graph | petgraph (Rust native) | ~200KB | shortest path, flow, matching, components |
| regex | regex crate (Rust native) | ~500KB | pattern matching, validation |
| units | custom (Rust) | ~100KB | unit conversion, dimensional analysis |
| datetime | chrono (Rust native) | ~200KB | date arithmetic, timezone handling |

### Integration Point

```rust
// In larql-inference, during layer processing

fn process_layer(&self, residual: &mut Tensor, layer: usize) -> Result<()> {
    // Standard attention
    self.attention(residual, layer)?;
    
    // FFN routing — now with compute dispatch
    match self.classify_residual(residual, layer)? {
        LayerAction::DenseFFN => self.dense_ffn(residual, layer)?,
        LayerAction::WalkFFN(entity) => self.walk_ffn(residual, layer, entity)?,
        LayerAction::CacheFFN(template) => self.cache_ffn(residual, layer, template)?,
        LayerAction::Compute(problem) => {
            // NEW: dispatch to solver
            if let Some(result) = self.solver_runtime.dispatch(&problem) {
                self.inject_result(residual, &result, layer)?;
            } else {
                self.dense_ffn(residual, layer)?; // fallback
            }
        }
        LayerAction::Skip => {} // pass-through
    }
    
    Ok(())
}
```

This fits cleanly into the existing LayerGraph architecture — `Compute` is just another `LayerAction` variant.

---

## Model Artifact (End State)

```
model/
  attention/          # trained weights (3.6% of params)
  knowledge/          # vindex (knowledge graph)
  compute/            # solver modules
    arithmetic.rs     # built into engine
    algebra.wasm      # symbolic math
    constraint.wasm   # counting + optimization
    logic.wasm        # SAT/SMT
    graph.wasm        # graph algorithms
    units.wasm        # dimensional analysis
    datetime.wasm     # temporal reasoning
  templates/          # compiled attention patterns (96.4%)
  output/             # token distributions
  style/              # connotation + profiles
  config.json
```

A model that is: attention weights + knowledge graph + compute modules + compiled templates. The WASM modules are the part where you get provably correct math. Everything else you've already built.

---

## Dependencies

| Dependency | Phase | Purpose |
|-----------|-------|---------|
| transformers / Gemma 3-4B | 1 | Baseline model for benchmarks |
| sympy | 1, 3 | Symbolic algebra |
| ortools (CP-SAT) | 1, 3 | Constraint solving |
| datasets (HuggingFace) | 1 | GSM8K, MATH, SVAMP, AQuA-RAT |
| wasmtime | 3 | WASM runtime |
| wasmtime Rust crate | 3 | Rust integration |

Phase 1 is pure Python. Phase 2 is Python + probes. Phase 3 is Rust.

---

## Timeline

| Day | Work | Deliverable |
|-----|------|-------------|
| 1 | Expression parser + solver dispatch (Python) | phase1_parser.py, phase1_solvers.py |
| 2 | Pipeline integration + GSM8K baseline | phase1_pipeline.py, baseline numbers |
| 3 | Full benchmark suite + error analysis | phase1_bench.py, RESULTS.md (Phase 1) |
| 4 | Residual classification probes | phase2a_classify.py |
| 5 | Structured extraction experiments | phase2b_extract.py |
| 6 | Injection / constrained decoding tests | phase2c_inject.py, RESULTS.md (Phase 2) |
| 7+ | WASM runtime + Rust integration (if Phase 2 succeeds) | larql-wasm crate |

---

## Connection to Prior Work

| Prior experiment | What it proved | How it connects |
|-----------------|---------------|-----------------|
| v8 (three systems) | FFN = table + graph + rules | Compute engine is a fourth system: exact solver |
| v9c (tool engine) | Tool calling is structured routing | Solver dispatch IS tool routing, but native |
| v12 (compiled attention) | 97% of attention is compilable | The model artifact shrinks further with native compute |
| Routing theory | Q-vector classifies problem type | Same classification routes to solver modules |
| LayerGraph | Pluggable per-layer actions | ComputeDispatch slots into existing architecture |
| Walk crossover | mmap'd lookup beats dense matmul | Same principle: precomputed > learned for exact tasks |

The compute engine is the logical next step after proving FFN is structured data. If the FFN is a database, and tool calling is routing, then embedding exact solvers is: better routing to better databases.
