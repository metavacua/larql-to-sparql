//! Minimal wasm-safe certifier.
//!
//! Parses a compiled `.wasm` module and always reports four metrics:
//!   - **imports** — host import count (any import is a capability/sandbox-escape witness);
//!   - **`call_indirect`/ref sites** — indirect-call count (ACE; breaks static call graph);
//!   - **`memory.grow` sites** — heap-growth count (¬M gate; bounded-memory class);
//!   - **direct recursion** — whether the static call graph has cycles (¬L gate; loop-free class).
//!
//! Pass condition (exit 0):
//!   - **default**: import-free AND call_indirect-free (original wasm-safe class)
//!   - **`--strict`**: additionally memory.grow-free AND recursion-free (sub-Turing ¬L∧¬M class)
//!
//! The distinction matters: artifacts that supply a global allocator (dlmalloc) or pull in a
//! recursive parser (serde_json) are wasm-safe but NOT ¬L∧¬M. Use --strict only for artifacts
//! whose reachable closure is provably allocator-free and recursion-free.
//!
//! This is the whole-module form of the xtask containment core, retargeted to wasm32v1-none.
//! It is a NATIVE verification tool — not part of any kernel.
//!
//! Usage: `larql-wasm-certify [--strict] <module.wasm>`
//!        exit 0 = WASM-SAFE (at selected level), 1 = not safe, 2 = error
use std::process::ExitCode;

use wasmparser::{Operator, Parser, Payload, TypeRef};

fn main() -> ExitCode {
    let mut strict = false;
    let mut path: Option<String> = None;
    for arg in std::env::args().skip(1) {
        if arg == "--strict" {
            strict = true;
        } else if path.is_none() {
            path = Some(arg);
        } else {
            eprintln!("usage: larql-wasm-certify [--strict] <module.wasm>");
            return ExitCode::from(2);
        }
    }
    let path = match path {
        Some(p) => p,
        None => {
            eprintln!("usage: larql-wasm-certify [--strict] <module.wasm>");
            return ExitCode::from(2);
        }
    };
    let bytes = match std::fs::read(&path) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("cannot read {path}: {e}");
            return ExitCode::from(2);
        }
    };
    match certify(&bytes) {
        Ok(report) => {
            println!("module: {path} ({} bytes)", bytes.len());
            println!("  imports:                 {}", report.imports.len());
            for imp in &report.imports {
                println!("    - {imp}");
            }
            println!("  call_indirect/ref sites: {}", report.indirect);
            println!("  memory.grow sites:       {}", report.memory_grows);
            println!("  direct recursion:        {}", report.direct_recursive);
            let base_safe = report.imports.is_empty() && report.indirect == 0;
            let sub_turing = report.memory_grows == 0 && !report.direct_recursive;
            let safe = base_safe && (!strict || sub_turing);
            if safe {
                if strict {
                    println!("WASM-SAFE [\u{00AC}L\u{2227}\u{00AC}M]: import-free + call_indirect-free + memory.grow-free + recursion-free");
                } else {
                    println!("WASM-SAFE: import-free + call_indirect-free");
                    if !sub_turing {
                        println!("  note: not \u{00AC}L\u{2227}\u{00AC}M (memory.grow or recursion present; use --strict to gate)");
                    }
                }
                ExitCode::SUCCESS
            } else {
                println!("NOT wasm-safe");
                ExitCode::from(1)
            }
        }
        Err(e) => {
            eprintln!("parse error: {e}");
            ExitCode::from(2)
        }
    }
}

struct Report {
    imports: Vec<String>,
    indirect: usize,
    memory_grows: usize,
    direct_recursive: bool,
}

fn certify(bytes: &[u8]) -> Result<Report, String> {
    let mut imports: Vec<String> = Vec::new();
    // Conservative count of function imports for call-graph indexing.
    // For import-free modules (the expected case) this stays 0.
    let mut func_import_count: usize = 0;
    let mut indirect: usize = 0;
    let mut memory_grows: usize = 0;
    // adjacency[i] = local-function indices directly called by local function i.
    let mut adjacency: Vec<Vec<u32>> = Vec::new();
    let mut current_func_idx: usize = 0;

    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for item in reader {
                    let item = item.map_err(|e| e.to_string())?;
                    match item {
                        wasmparser::Imports::Single(_offset, import) => {
                            if matches!(import.ty, TypeRef::Func(_)) {
                                func_import_count += 1;
                            }
                            imports.push(format!("{}::{}", import.module, import.name));
                        }
                        wasmparser::Imports::Compact1 { module, items } => {
                            for ci in items {
                                let ci = ci.map_err(|e| e.to_string())?;
                                func_import_count += 1; // conservative: assume function import
                                imports.push(format!("{module}::{}", ci.name));
                            }
                        }
                        wasmparser::Imports::Compact2 { module, ty: _, names } => {
                            for name in names {
                                let name = name.map_err(|e| e.to_string())?;
                                func_import_count += 1; // conservative: assume function import
                                imports.push(format!("{module}::{name}"));
                            }
                        }
                    }
                }
            }
            Payload::FunctionSection(reader) => {
                // Pre-size the adjacency list; FunctionSection precedes CodeSection in valid WASM.
                adjacency = vec![Vec::new(); reader.count() as usize];
            }
            Payload::CodeSectionEntry(body) => {
                // Safety: guard against malformed modules where code entries exceed declared count.
                if current_func_idx >= adjacency.len() {
                    adjacency.push(Vec::new());
                }
                let reader = body.get_operators_reader().map_err(|e| e.to_string())?;
                for op in reader {
                    match op.map_err(|e| e.to_string())? {
                        Operator::CallIndirect { .. }
                        | Operator::ReturnCallIndirect { .. }
                        | Operator::CallRef { .. }
                        | Operator::ReturnCallRef { .. } => indirect += 1,
                        Operator::Call { function_index } => {
                            let fi = function_index as usize;
                            if fi >= func_import_count {
                                let local_idx = fi - func_import_count;
                                if local_idx < adjacency.len() {
                                    adjacency[current_func_idx].push(local_idx as u32);
                                }
                            }
                        }
                        Operator::MemoryGrow { .. } => memory_grows += 1,
                        _ => {}
                    }
                }
                current_func_idx += 1;
            }
            _ => {}
        }
    }

    let direct_recursive = has_cycle(&adjacency);
    Ok(Report { imports, indirect, memory_grows, direct_recursive })
}

/// Iterative DFS cycle detection with tricolor marking.
/// Returns true iff the directed graph has at least one cycle (direct recursion).
fn has_cycle(adj: &[Vec<u32>]) -> bool {
    let n = adj.len();
    if n == 0 {
        return false;
    }
    // 0 = WHITE (unvisited), 1 = GRAY (on current path), 2 = BLACK (fully processed)
    let mut color = vec![0u8; n];
    for start in 0..n {
        if color[start] != 0 {
            continue;
        }
        // Stack holds (node, next-neighbor-index-to-visit).
        let mut stack: Vec<(usize, usize)> = vec![(start, 0)];
        color[start] = 1;
        while !stack.is_empty() {
            // Copy to avoid simultaneous mutable+immutable borrows of `stack`.
            let (node, idx) = *stack.last().unwrap();
            if idx < adj[node].len() {
                let neighbor = adj[node][idx] as usize;
                stack.last_mut().unwrap().1 += 1; // advance iterator for this node
                if color[neighbor] == 1 {
                    return true; // back edge → cycle
                }
                if color[neighbor] == 0 {
                    color[neighbor] = 1;
                    stack.push((neighbor, 0));
                }
            } else {
                color[node] = 2;
                stack.pop();
            }
        }
    }
    false
}
