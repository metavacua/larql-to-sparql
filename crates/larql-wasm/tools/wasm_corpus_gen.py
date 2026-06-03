#!/usr/bin/env python3
"""
wasm32v1-none corpus generator — codebase as training data.

Reads the LARQL wasm workspace (certifier, gate tools, CONTEXT.md, ADR-0013,
the GPU fork source) and generates a systematic NL→WAT/Rust/certifier corpus
covering the formal grammar of wasm32v1-none and the ¬L∧¬M sub-Turing class.

Three formal grammars are extracted from the codebase:

  1. The certifier grammar (larql-wasm-certify/src/main.rs):
       wasm_safe   ::= import-free ∧ call_indirect-free
       ¬L∧¬M       ::= wasm_safe ∧ memory.grow-free ∧ recursion-free

  2. The WAT (WebAssembly Text Format) ¬L∧¬M subset:
       safe_module ::= '(' 'module' safe_func* export* ')'
       safe_func   ::= '(' 'func' id? type? local* safe_instr* ')'
       safe_instr  — no call_indirect, no memory.grow, acyclic call graph

  3. The Rust wasm32v1-none patterns:
       #![no_std]  | #![cfg_attr(target_arch = "wasm32", no_std)]
       #[cfg(not(target_arch = "wasm32"))]
       #[cfg(target_os = "none")]   — wasm32v1-none specifically
       #[cfg(target_os = "wasi")]   — wasm32-wasip1 specifically
       libm::expf(-x)               — no_std float math replacement

Sources read:
  crates/larql-wasm/larql-wasm-certify/src/main.rs
  crates/larql-wasm/CONTEXT.md
  crates/larql-wasm/tools/wasm_gate_common.py
  docs/adr/0013-lql-wasm32v1-none-correspondence.md
  crates/larql-wasm/larql-compute-gpu/src/{lib.rs,portable/mod.rs}

Output:
  crates/larql-wasm/tools/wasm_corpus.json

Usage:
  python wasm_corpus_gen.py [--repo-root /path/to/larql-to-sparql]
"""
import argparse
import json
import re
from pathlib import Path


# ── WASM "statement types" — analogous to LQL Statement variants ─────────────
# Each entry: (first_token, stmt_type, category, [NL prompts], [canonical answers])

WASM_TEMPLATES = [

    # ── CERTIFIER ─────────────────────────────────────────────────────────────
    ("CERTIFY", "Certify", "certifier", [
        "Certify this wasm module as ¬L∧¬M",
        "Run larql-wasm-certify on this module",
        "Is this wasm module import-free and call_indirect-free?",
        "Check if this module passes the wasm-safe gate",
        "Run --strict certification on this wasm binary",
        "Verify this .wasm file is sub-Turing safe",
        "Does this module satisfy the ¬L∧¬M property?",
        "Check if this module has no host imports",
        "Run the wasm certifier on this compiled module",
        "Is this kernel certifiable as ¬L∧¬M?",
        "Verify no call_indirect and no memory.grow in this module",
        "Can this wasm module be deployed in a browser sandbox?",
    ], [
        "larql-wasm-certify --strict target/wasm32v1-none/release/module.wasm",
        "larql-wasm-certify target/wasm32v1-none/release/module.wasm",
        "# Expected: WASM-SAFE [¬L∧¬M]: import-free + call_indirect-free + memory.grow-free + recursion-free",
    ]),

    ("CERTIFY", "CertifyDefault", "certifier", [
        "Check if this module is wasm-safe (import-free + call_indirect-free)",
        "Run the default wasm safety check",
        "Does this module pass the basic wasm-safe gate?",
        "Verify import-free and ACE-free for this module",
    ], [
        "larql-wasm-certify target/wasm32v1-none/release/module.wasm",
        "# Expected: WASM-SAFE: import-free + call_indirect-free",
    ]),

    # ── INSPECT ───────────────────────────────────────────────────────────────
    ("INSPECT", "InspectImports", "inspect", [
        "What imports does this wasm module have?",
        "List all host imports in this wasm module",
        "Does this module have any host imports?",
        "Show the import section of this wasm binary",
        "Which host functions does this module import?",
        "Check if this wasm is import-free",
        "What external functions does this module call?",
        "Show imports for this wasm kernel",
    ], [
        "wasmparser: ImportSection → list of (module, name) pairs",
        "# import-free: no Payload::ImportSection entries",
        "# wasm-safe: TypeRef::Func imports count = 0",
    ]),

    ("INSPECT", "InspectCallIndirect", "inspect", [
        "Does this wasm module use call_indirect?",
        "Check for indirect call instructions in this module",
        "Is this module call_indirect-free?",
        "Show all call_indirect sites in this wasm",
        "How many ACE sites does this module have?",
        "Does this code break the static call graph?",
    ], [
        "# call_indirect count: Operator::CallIndirect + ReturnCallIndirect + CallRef + ReturnCallRef",
        "# wasm-safe: call_indirect_count == 0",
        "# ACE risk: any call_indirect breaks static call graph analysis",
    ]),

    ("INSPECT", "InspectMemoryGrow", "inspect", [
        "Does this wasm module use memory.grow?",
        "Check for memory.grow instructions in this module",
        "Is this module ¬M (bounded-memory)?",
        "How many memory.grow sites does this module have?",
        "Does this code allocate heap dynamically?",
        "Is this module free of unbounded heap growth?",
    ], [
        "# memory.grow count: Operator::MemoryGrow { .. }",
        "# ¬M: memory_grows == 0",
        "# note: dlmalloc and global allocators use memory.grow",
    ]),

    ("INSPECT", "InspectRecursion", "inspect", [
        "Does this wasm module have direct recursion?",
        "Check the call graph for cycles in this module",
        "Is this module recursion-free (¬L)?",
        "Does the static call graph have any back edges?",
        "Is the call graph of this module acyclic?",
        "Check for recursive functions in this wasm",
    ], [
        "# recursion: has_cycle(adjacency) using DFS tricolor marking",
        "# ¬L: direct_recursive == false (no call-graph cycles)",
        "# note: call_indirect breaks static analysis; must be absent first",
    ]),

    # ── CLASSIFY ──────────────────────────────────────────────────────────────
    ("CLASSIFY", "ClassifySubTuring", "classify", [
        "What sub-Turing class does this code belong to?",
        "Is this code ¬L, ¬M, ¬L∧¬M, or Turing-complete?",
        "Classify this wasm module in the ¬L∧¬M lattice",
        "Which decidable fragment is this code in?",
        "Is this code Turing-complete or sub-Turing?",
        "What is the computational class of this wasm module?",
        "Is T ⇔ L∧M broken for this code?",
        "Does this module satisfy ¬(L∧M)?",
    ], [
        "¬L∧¬M: no recursion (¬L) + no memory.grow (¬M) — strictest sub-Turing",
        "¬L only: recursion-free but may use memory.grow",
        "¬M only: bounded-memory but may recurse",
        "Turing-complete: L∧M — both recursion and unbounded heap present",
        "# T ⇔ L∧M: Turing-complete iff unbounded loops AND unbounded memory",
    ]),

    ("CLASSIFY", "ClassifyWasmDialect", "classify", [
        "Which wasm dialect does this target?",
        "Is this wasm32v1-none or wasm32-wasip1?",
        "What is the wasm dialect hierarchy for this target?",
        "Where does wasm32v1-none sit in the wasm dialect lattice?",
        "What is the difference between wasm32v1-none and wasm32-wasip1?",
        "Is target_os = 'none' the same as wasm32v1-none?",
    ], [
        "wasm32v1-none: target_os='none', no OS, no std, no imports — the floor",
        "wasm32-wasip1: target_os='wasi', WASI std, file I/O — one level above",
        "hierarchy: wasm32v1-none ⊂ wasm32-unknown-unknown ⊂ {+simd128,+atomics,wasip1}",
        "# target_os = 'none'  → wasm32v1-none specifically",
        "# target_os = 'wasi'  → wasm32-wasip1 specifically",
        "# target_arch = 'wasm32' → both targets (use target_os to distinguish)",
    ]),

    # ── GATE ──────────────────────────────────────────────────────────────────
    ("GATE", "GateNative", "gate", [
        "Add a cfg gate to exclude this code from wasm builds",
        "How do I gate native-only code out of wasm32v1-none?",
        "Add the wasm32 cfg gate to this function",
        "Gate this OS-dependent code off for wasm targets",
        "What attribute marks this function as native-only?",
        "Exclude this from wasm32 compilation",
        "How do I mark code that can't run on wasm32v1-none?",
        "Add a cfg(not(target_arch = 'wasm32')) gate",
    ], [
        '#[cfg(not(target_arch = "wasm32"))]',
        '// Native-only: uses OS I/O or std features unavailable on wasm32v1-none',
        '#[cfg(not(any(target_arch = "wasm32", target_arch = "nvptx64")))]',
    ]),

    ("GATE", "GateNoStd", "gate", [
        "How do I make a crate no_std for wasm32v1-none?",
        "Add the no_std attribute for wasm targets",
        "What no_std annotation is needed for wasm32v1-none?",
        "Make this crate compile without std on wasm",
        "Add cfg_attr no_std for the wasm32v1-none target",
        "How do I gate no_std only for wasm32v1-none (not wasip1)?",
        "Add no_std for target_os = none but keep std for wasip1",
    ], [
        '#![cfg_attr(target_arch = "wasm32", no_std)]',
        '#![cfg_attr(any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv"), no_std)]',
        '// target_os = "none"  → wasm32v1-none (no std)',
        '// target_os = "wasi"  → wasm32-wasip1 (has WASI std, tests can run)',
        '// target_arch = "wasm32" alone gates BOTH wasip1 and v1-none — avoid for no_std',
    ]),

    ("GATE", "GateWasmOnly", "gate", [
        "How do I add code only for wasm32v1-none builds?",
        "Gate code to run only on the wasm32 target",
        "Add a cfg(target_arch = 'wasm32') gate",
        "Include this only in wasm32v1-none compilation",
        "What attribute is wasm32-specific?",
    ], [
        '#[cfg(target_arch = "wasm32")]',
        '#[cfg(target_os = "none")]  // wasm32v1-none specifically, not wasip1',
    ]),

    # ── CONVERT ───────────────────────────────────────────────────────────────
    ("CONVERT", "ConvertFloat", "convert", [
        "How do I use exp() in no_std wasm32v1-none code?",
        "Replace f32::exp() for wasm32v1-none compatibility",
        "What is the no_std replacement for (-x).exp()?",
        "f32::exp is not in core — how do I use it in wasm?",
        "How do I compute exponential in no_std Rust?",
        "Replace std float math for wasm32v1-none target",
        "What libm function replaces f32::exp in no_std?",
        "How does the larql-compute-gpu crate handle exp() on wasm?",
    ], [
        "libm::expf(-x)  // instead of (-x).exp() which needs std",
        "#[cfg(any(target_os = \"none\", target_arch = \"nvptx64\"))]",
        "fn neg_exp(x: f32) -> f32 { libm::expf(-x) }  // no_std",
        "#[cfg(not(any(target_os = \"none\", target_arch = \"nvptx64\")))]",
        "fn neg_exp(x: f32) -> f32 { (-x).exp() }  // std targets",
        "// dep: libm = { version = \"0.2\", default-features = false }",
    ]),

    ("CONVERT", "ConvertPanic", "convert", [
        "How do I add a panic handler for wasm32v1-none?",
        "Add the required panic_handler for no_std wasm",
        "What panic handler is needed for wasm32v1-none?",
        "My no_std crate needs a panic handler for wasm",
    ], [
        '#[cfg(target_arch = "wasm32")]',
        '#[panic_handler]',
        'fn panic(_: &core::panic::PanicInfo) -> ! { loop {} }',
        "// or: use panic-halt crate for wasm32v1-none",
    ]),

    ("CONVERT", "ConvertAlloc", "convert", [
        "How do I use alloc in a wasm32v1-none crate?",
        "Add alloc support to a no_std wasm crate",
        "Use Vec in a no_std wasm32v1-none crate",
        "How do I use String in no_std wasm code?",
    ], [
        "extern crate alloc;",
        "use alloc::vec::Vec;",
        "use alloc::string::String;",
        "// requires a global allocator (e.g. wee_alloc or dlmalloc)",
        "// caution: alloc uses memory.grow — violates ¬M class",
    ]),

    # ── WAT grammar (WebAssembly Text Format) ─────────────────────────────────
    ("WAT_MODULE", "WatModule", "wat", [
        "Write a minimal wasm module in WAT format",
        "What is the WAT syntax for a wasm32v1-none module?",
        "Show me the WebAssembly text format for a simple module",
        "Write a ¬L∧¬M safe wasm module in WAT",
        "What does a wasm32v1-none kernel look like in WAT?",
        "Write a WAT module with one exported function",
    ], [
        "(module",
        "  (func $silu (export \"silu\") (param $x f32) (result f32)",
        "    (f32.div",
        "      (local.get $x)",
        "      (f32.add (f32.const 1.0) (f32.neg (local.get $x)))",  # simplified
        "    )",
        "  )",
        ")",
        "# ¬L∧¬M: no call_indirect, no memory.grow, no recursive calls",
    ]),

    ("WAT_FUNC", "WatFunction", "wat", [
        "Write a WAT function for silu activation",
        "What is the WAT syntax for a function with params and results?",
        "Write a ¬L∧¬M safe function in WAT",
        "Show the WAT grammar for a function definition",
        "What does a safe wasm function look like in text format?",
    ], [
        "(func $name (param $p f32) (result f32)",
        "  ;; body: no call_indirect, no memory.grow",
        "  (local $tmp f32)",
        "  local.get $p",
        "  ;; ... instructions ...",
        ")",
        "# ¬L: no recursion (call graph must be acyclic)",
        "# ¬M: no memory.grow instructions",
    ]),

    ("WAT_FORBIDDEN", "WatForbiddenCallIndirect", "wat", [
        "What wasm instruction is forbidden in ¬L∧¬M code?",
        "Why is call_indirect dangerous in wasm?",
        "What instruction enables arbitrary code execution in wasm?",
        "Why does call_indirect break static call graph analysis?",
        "What is the ACE vector in wasm?",
        "Why does the certifier check for call_indirect?",
    ], [
        "call_indirect  ;; FORBIDDEN: arbitrary code execution (ACE)",
        "call_indirect  ;; breaks static call graph — enables vtable-like dispatch",
        "# call_indirect takes a runtime index — static analysis cannot bound the target",
        "# substitute: use direct call $name with acyclic call graph",
        "# certifier check: Operator::CallIndirect | ReturnCallIndirect | CallRef | ReturnCallRef",
    ]),

    ("WAT_FORBIDDEN", "WatForbiddenMemoryGrow", "wat", [
        "What wasm instruction violates the ¬M property?",
        "Why is memory.grow forbidden in ¬M code?",
        "What instruction causes unbounded heap growth in wasm?",
        "Why does memory.grow break the bounded-memory class?",
        "What does memory.grow do that is unsafe in certified code?",
    ], [
        "memory.grow  ;; FORBIDDEN in ¬M class: grows linear memory unboundedly",
        "# any allocation (Vec, String, dlmalloc) compiles to memory.grow",
        "# ¬M: memory.grow count must be 0",
        "# certifier check: Operator::MemoryGrow { .. }",
        "# safe alternative: use fixed-size arrays, caller-provided slices",
    ]),

    # ── THREE-AXIS FRAMEWORK ──────────────────────────────────────────────────
    ("EXPLAIN", "ExplainThreeAxes", "explain", [
        "Explain the three axes of wasm32v1-none safety",
        "What are the three dimensions of wasm safety analysis?",
        "Explain capability, computation, and arithmetic for wasm",
        "What is the three-axis wasm safety framework?",
        "How is wasm safety measured in LARQL?",
        "Explain the wasm safety lattice",
    ], [
        "Axis 1 — Capability (imports): no host imports → sandbox escape prevented",
        "Axis 2 — Computation (T⇔L∧M): ¬L (loop-free) or ¬M (bounded-memory) → sub-Turing",
        "Axis 3 — Arithmetic definability (Q_fin): absence of Robinson Q generators → decidable",
        "# certified target = WASM-valid ∩ wasm-safe ∩ Q-classified ∩ total",
        "# T ⇔ L∧M: Turing-complete iff unbounded loops AND unbounded heap together",
        "# ¬L and ¬M are dual, incomparable; linking them can reconstruct T",
    ]),

    ("EXPLAIN", "ExplainSubTuringClass", "explain", [
        "What is the ¬L∧¬M class?",
        "Explain the sub-Turing wasm classification",
        "What makes code sub-Turing in wasm32v1-none?",
        "When is wasm code in the ¬L class vs the ¬M class?",
        "What is the difference between ¬L and ¬M wasm code?",
        "Explain Turing completeness in the context of wasm kernels",
    ], [
        "¬L (loop-free): no recursion, no call_indirect — call graph is a DAG",
        "¬M (bounded-memory): no memory.grow — heap size is statically bounded",
        "¬L∧¬M (strictest): both — provably terminating AND bounded memory",
        "Turing-complete: L∧M — has both unbounded loops AND unbounded heap",
        "# ¬L‖¬M: incomparable — neither is a subset of the other",
        "# linking ¬L code to ¬M code can reconstruct L∧M (Turing-complete)",
    ]),

    ("EXPLAIN", "ExplainWasm32v1None", "explain", [
        "What is wasm32v1-none?",
        "What is the wasm32v1-none Rust target?",
        "How is wasm32v1-none different from wasm32-unknown-unknown?",
        "What constraints does wasm32v1-none impose?",
        "Why use wasm32v1-none instead of wasm32-unknown-unknown?",
        "What does the 'v1' in wasm32v1-none mean?",
    ], [
        "wasm32v1-none: WebAssembly MVP 1.0 + mutable-globals; no OS; no std; no imports",
        "# triple: wasm32-unknown-none (or wasm32v1-none in newer Rust)",
        "# no SIMD, no atomics, no bulk-memory — MVP feature set only",
        "# target_os = 'none': no OS present; contrast wasm32-wasip1 (target_os='wasi')",
        "# floor target: most restricted; every wasm-safe program compiles here",
        "# wasm32v1-none ⊂ wasm32-unknown-unknown ⊂ {wasip1, emscripten, ...}",
    ]),

    ("EXPLAIN", "ExplainCertifier", "explain", [
        "How does larql-wasm-certify work?",
        "What does the wasm certifier check?",
        "Explain the certifier's four metrics",
        "What is the difference between default and --strict certification?",
        "How does the certifier detect recursion?",
        "What does import-free mean for wasm certification?",
    ], [
        "certifier metrics: imports | call_indirect count | memory.grow count | direct recursion",
        "default (no --strict): import-free AND call_indirect-free → WASM-SAFE",
        "--strict: additionally memory.grow-free AND recursion-free → WASM-SAFE [¬L∧¬M]",
        "recursion detection: DFS tricolor marking on static call graph (adjacency list)",
        "import-free: TypeRef::Func import count = 0",
        "call_indirect-free: no Operator::CallIndirect | CallRef | ReturnCall* variants",
    ]),

    # ── RUST PATTERNS ─────────────────────────────────────────────────────────
    ("PATTERN", "PatternNoStdCrate", "pattern", [
        "How do I structure a wasm32v1-none compatible Rust crate?",
        "What does a minimal wasm32v1-none crate look like?",
        "Show the Cargo.toml for a no_std wasm crate",
        "What dependencies are safe in wasm32v1-none?",
        "How do I make a crate work on both wasm32v1-none and native?",
        "What is the pattern for a wasm32v1-none kernel crate?",
    ], [
        "# Cargo.toml: no ndarray, rayon, serde, thiserror — all require std",
        "[dependencies]",
        'libm = { version = "0.2", default-features = false }  # no_std float math',
        "# lib.rs:",
        '#![cfg_attr(any(target_os = "none", target_arch = "nvptx64"), no_std)]',
        "pub mod portable;  // only pure arithmetic, no OS calls",
    ]),

    ("PATTERN", "PatternCfgDispatch", "pattern", [
        "How do I dispatch between std and no_std float math?",
        "Show the cfg dispatch pattern for libm vs std",
        "How does larql-compute-gpu handle exp() across targets?",
        "Write a cfg-dispatched helper for transcendental math",
        "How do I use different code for wasm32v1-none and native?",
    ], [
        '#[cfg(any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv"))]',
        "#[inline(always)]",
        "fn neg_exp(x: f32) -> f32 { libm::expf(-x) }",
        "",
        '#[cfg(not(any(target_os = "none", target_arch = "nvptx64", target_arch = "spirv")))]',
        "#[inline(always)]",
        "fn neg_exp(x: f32) -> f32 { (-x).exp() }",
        "// pattern from crates/larql-wasm/larql-compute-gpu/src/portable/mod.rs",
    ]),

    ("PATTERN", "PatternSliceKernel", "pattern", [
        "How do I write a ¬L∧¬M kernel that operates on slices?",
        "What is the pattern for a bounded-memory kernel function?",
        "Show a wasm32v1-none safe kernel signature",
        "How do I avoid allocation in a wasm kernel?",
        "Write a geglu_silu function that is ¬L∧¬M safe",
    ], [
        "/// ¬L∧¬M: caller-provided slices, no heap allocation, bounded loop",
        "pub fn geglu_silu(gate: &[f32], up: &[f32], out: &mut [f32]) {",
        "    debug_assert_eq!(gate.len(), up.len());",
        "    for i in 0..gate.len() {",
        "        out[i] = silu(gate[i]) * up[i];  // no allocation, bounded loop",
        "    }",
        "}",
        "// ¬L: loop is bounded by gate.len() (caller's slice length)",
        "// ¬M: no Vec, no String, no memory.grow — slices are caller-owned",
    ]),

    # ── LQL ↔ WASM CORRESPONDENCE (from ADR-0013) ─────────────────────────────
    ("CORRESPONDENCE", "LqlWasmCertified", "correspondence", [
        "Which LQL statements have ¬L∧¬M certified evaluators?",
        "What is the browser LQL dialect?",
        "Which LQL statements can run in a wasm32v1-none sandbox?",
        "What is the LQL ↔ wasm32v1-none correspondence?",
        "Which LQL statements are certified sub-Turing?",
        "What LQL can run in a browser without native code?",
    ], [
        "certified (browser LQL): WALK, DESCRIBE, SELECT, INSERT(KNN), SHOW, STATS",
        "not certified: INFER (softmax global reduction), TRACE (¬M), COMPILE (¬M)",
        "# certified(S) ⟺ evaluator(S) compiles to wasm32v1-none",
        "#             ∧  larql-wasm-certify --strict evaluator.wasm exits 0",
        "# INFER blocker: softmax attention requires L1 global reduction — Phase 3 seam",
        "# see docs/adr/0013-lql-wasm32v1-none-correspondence.md",
    ]),

    ("CORRESPONDENCE", "LqlWasmBlocker", "correspondence", [
        "Why is INFER not in the ¬L∧¬M class?",
        "What blocks INFER from being wasm32v1-none certified?",
        "What is the Phase 3 seam for wasm certification?",
        "Why can't attention be certified as ¬L∧¬M?",
        "What is the softmax global reduction problem for wasm?",
    ], [
        "INFER blocker: softmax attention = global L1 reduction over all scores",
        "# must read all scores before writing any output → not ¬L (reads depend on all inputs)",
        "# causal mask breaks Toeplitz structure → can't parallelize as convolution",
        "# Phase 3 seam: exactly the boundary between auto-parallelizable and non-GPU-portable",
        "# fix path: Born Rule attention (L2 normalization) — see ADR-0015",
        "# TRACE blocker: allocates per-layer Vec<f32> → memory.grow → ¬M violation",
    ]),
]

# ── WAT grammar productions (formal grammar fragments) ───────────────────────
# These are canonical WAT syntax examples extracted from the spec and embedded
# as training data. The user's insight: the grammar IS the training data.

WAT_GRAMMAR_PRODUCTIONS = [
    # Module structure
    ("(module ...)                  ;; top-level wasm module", "WAT_MODULE"),
    ("(type (func (param f32) (result f32)))  ;; function type", "WAT_FUNC"),
    ("(func $name (type $ft) ...)", "WAT_FUNC"),
    ("(export \"name\" (func $idx))", "WAT_EXPORT"),
    ("(import \"mod\" \"name\" (func $f (type $t)))  ;; FORBIDDEN: host import", "WAT_IMPORT"),
    ("(memory 1)   ;; one 64KB page", "WAT_MEMORY"),
    ("(global $g (mut i32) (i32.const 0))", "WAT_GLOBAL"),
    # Instructions — safe
    ("local.get $x   ;; push local variable", "WAT_LOCAL"),
    ("local.set $x   ;; pop to local variable", "WAT_LOCAL"),
    ("i32.const 42   ;; push constant", "WAT_CONST"),
    ("f32.const 1.0  ;; push float constant", "WAT_CONST"),
    ("i32.add  f32.mul  i32.sub  ;; numeric operations", "WAT_NUMERIC"),
    ("i32.load  i32.store  ;; bounded memory access (no memory.grow)", "WAT_MEMORY_OPS"),
    ("call $function_name  ;; direct call (safe if acyclic)", "WAT_CALL"),
    ("if (result i32) ... else ... end  ;; conditional", "WAT_CONTROL"),
    ("block ... br $label ... end  ;; bounded block", "WAT_CONTROL"),
    # Instructions — FORBIDDEN
    ("call_indirect (type $t) ... ;; FORBIDDEN: ACE, breaks static call graph", "WAT_FORBIDDEN"),
    ("memory.grow   ;; FORBIDDEN in ¬M: unbounded heap growth", "WAT_FORBIDDEN"),
    # ¬L∧¬M contract
    ("(func ;; ¬L∧¬M: no call_indirect, no memory.grow, acyclic calls)", "WAT_SAFE"),
]


def generate_wat_grammar_corpus() -> list[dict]:
    """Generate (NL question, WAT answer) pairs from grammar productions."""
    corpus = []
    for production, category in WAT_GRAMMAR_PRODUCTIONS:
        first_token = "WAT_MODULE" if "module" in production else \
                      "WAT_FORBIDDEN" if "FORBIDDEN" in production else \
                      "EXPLAIN"
        is_forbidden = "FORBIDDEN" in production
        corpus.append({
            "prompt": f"Show the WAT syntax: {production.split(';')[0].strip()[:60]}",
            "expected_lql": production,
            "first_token": first_token,
            "statement_type": category,
            "category": "wat",
            "split": "train" if not is_forbidden else "eval",
            "source": "wat_grammar",
        })
    return corpus


def extract_certifier_checks(repo_root: Path) -> list[dict]:
    """Extract the formal check names from the certifier source."""
    certifier = repo_root / "crates/larql-wasm/larql-wasm-certify/src/main.rs"
    if not certifier.exists():
        return []

    text = certifier.read_text()
    corpus = []

    # Extract the output messages which define the grammar
    for m in re.finditer(r'"(WASM-SAFE[^"]+)"', text):
        msg = m.group(1)
        corpus.append({
            "prompt": f"What does the certifier output on success? Expected: {msg[:60]}",
            "expected_lql": msg,
            "first_token": "CERTIFY",
            "statement_type": "CertifierOutput",
            "category": "certifier",
            "split": "eval",
            "source": "certifier_source",
        })

    # Extract the four metric names
    for metric in ["imports", "call_indirect/ref sites", "memory.grow sites", "direct recursion"]:
        corpus.append({
            "prompt": f"What metric does the certifier report for: {metric}?",
            "expected_lql": f"certifier reports {metric} count in the report header",
            "first_token": "INSPECT",
            "statement_type": "CertifierMetric",
            "category": "certifier",
            "split": "train",
            "source": "certifier_source",
        })

    return corpus


def extract_gate_patterns(repo_root: Path) -> list[dict]:
    """Extract cfg gate patterns from wasm_gate_common.py."""
    gate_file = repo_root / "crates/larql-wasm/tools/wasm_gate_common.py"
    if not gate_file.exists():
        return []

    text = gate_file.read_text()
    corpus = []

    # Extract GATE and GATE_SUBSTR constants
    for m in re.finditer(r"GATE\s*=\s*'([^']+)'", text):
        pattern = m.group(1)
        corpus.append({
            "prompt": f"What is the canonical native-only cfg gate in LARQL?",
            "expected_lql": pattern,
            "first_token": "GATE",
            "statement_type": "GateNative",
            "category": "gate",
            "split": "train",
            "source": "gate_common",
        })

    return corpus


def extract_adr_correspondence(repo_root: Path) -> list[dict]:
    """Extract LQL↔wasm correspondence table from ADR-0013."""
    adr = repo_root / "docs/adr/0013-lql-wasm32v1-none-correspondence.md"
    if not adr.exists():
        return []

    text = adr.read_text()
    corpus = []

    # Extract certified and not-certified statement lists from the table
    in_certified = False
    for line in text.splitlines():
        if "Certified subset" in line:
            in_certified = True
        elif "Not certified" in line:
            in_certified = False
        elif in_certified and "|" in line and "WALK" in line or "DESCRIBE" in line or "SELECT" in line:
            # Table row
            parts = [p.strip() for p in line.split("|") if p.strip()]
            if parts and parts[0] not in ["Statement", "---"]:
                stmt = parts[0]
                reason = parts[1] if len(parts) > 1 else ""
                corpus.append({
                    "prompt": f"Is {stmt} in the certified wasm32v1-none LQL dialect?",
                    "expected_lql": f"{stmt} is {'certified' if in_certified else 'not certified'}: {reason[:80]}",
                    "first_token": "CORRESPONDENCE",
                    "statement_type": "LqlWasmCertified",
                    "category": "correspondence",
                    "split": "eval",
                    "source": "adr_0013",
                })

    return corpus


def generate_corpus(repo_root: Path) -> list[dict]:
    """Generate the full wasm32v1-none corpus from the codebase."""
    corpus = []
    seen_prompts = set()

    # Source 1: Template-generated entries
    for first_token, stmt_type, category, nl_templates, answers in WASM_TEMPLATES:
        answer_str = "\n".join(answers)
        for i, nl in enumerate(nl_templates):
            if nl in seen_prompts:
                continue
            seen_prompts.add(nl)
            split = "train" if i < len(nl_templates) * 2 // 3 else "eval"
            corpus.append({
                "prompt": nl,
                "expected_lql": answers[i % len(answers)],
                "first_token": first_token,
                "statement_type": stmt_type,
                "category": category,
                "split": split,
                "source": "template",
            })

    # Source 2: WAT grammar productions
    corpus.extend(generate_wat_grammar_corpus())

    # Source 3: Certifier source
    corpus.extend(extract_certifier_checks(repo_root))

    # Source 4: Gate patterns
    corpus.extend(extract_gate_patterns(repo_root))

    # Source 5: ADR-0013 correspondence table
    corpus.extend(extract_adr_correspondence(repo_root))

    # Deduplicate by prompt
    seen = set()
    deduped = []
    for entry in corpus:
        p = entry["prompt"]
        if p not in seen:
            seen.add(p)
            deduped.append(entry)

    return deduped


def main():
    ap = argparse.ArgumentParser(
        description="Generate wasm32v1-none NL corpus from the codebase")
    ap.add_argument("--repo-root", default=".",
                    help="Path to the larql-to-sparql repo root")
    ap.add_argument("--output",
                    default="crates/larql-wasm/tools/wasm_corpus.json")
    ap.add_argument("--stats", action="store_true")
    args = ap.parse_args()

    repo = Path(args.repo_root).resolve()
    corpus = generate_corpus(repo)

    out_path = repo / args.output
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(corpus, indent=2))

    from collections import Counter
    by_token = Counter(e["first_token"] for e in corpus)
    by_split = Counter(e["split"] for e in corpus)
    by_source = Counter(e["source"] for e in corpus)

    print(f"Generated {len(corpus)} wasm corpus entries → {out_path}")
    print(f"  Split:  train={by_split['train']}  eval={by_split['eval']}")
    print(f"  Source: {dict(by_source)}")
    if args.stats:
        print("\nPer first_token:")
        for tok, count in sorted(by_token.items()):
            print(f"  {tok:20s}  {count}")


if __name__ == "__main__":
    main()
