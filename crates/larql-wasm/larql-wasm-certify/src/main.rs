//! Minimal wasm-safe certifier.
//!
//! Parses a compiled `.wasm` module and asserts it is **wasm-safe**:
//!   - **import-free** — no host imports at all (a pure wasm32v1-none compute
//!     module needs none; any import is a capability/escape witness); and
//!   - **`call_indirect`-free** — no indirect/ref calls anywhere, so there is no
//!     arbitrary code execution and the call graph is fully static.
//!
//! This is the whole-module form of the xtask containment core, retargeted to
//! wasm32v1-none. It is a NATIVE verification tool — not part of any kernel.
//!
//! Usage: `larql-wasm-certify <module.wasm>`  (exit 0 = WASM-SAFE, 1 = not, 2 = error)
use std::process::ExitCode;

use wasmparser::{Operator, Parser, Payload};

fn main() -> ExitCode {
    let path = match std::env::args().nth(1) {
        Some(p) => p,
        None => {
            eprintln!("usage: larql-wasm-certify <module.wasm>");
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
            if report.imports.is_empty() && report.indirect == 0 {
                println!("WASM-SAFE: import-free + call_indirect-free");
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
}

fn certify(bytes: &[u8]) -> Result<Report, String> {
    let mut imports: Vec<String> = Vec::new();
    let mut indirect: usize = 0;
    for payload in Parser::new(0).parse_all(bytes) {
        match payload.map_err(|e| e.to_string())? {
            Payload::ImportSection(reader) => {
                for item in reader {
                    let item = item.map_err(|e| e.to_string())?;
                    match item {
                        wasmparser::Imports::Single(_offset, import) => {
                            imports.push(format!("{}::{}", import.module, import.name));
                        }
                        wasmparser::Imports::Compact1 { module, items } => {
                            for ci in items {
                                let ci = ci.map_err(|e| e.to_string())?;
                                imports.push(format!("{module}::{}", ci.name));
                            }
                        }
                        wasmparser::Imports::Compact2 { module, ty: _, names } => {
                            for name in names {
                                let name = name.map_err(|e| e.to_string())?;
                                imports.push(format!("{module}::{name}"));
                            }
                        }
                    }
                }
            }
            Payload::CodeSectionEntry(body) => {
                let reader = body.get_operators_reader().map_err(|e| e.to_string())?;
                for op in reader {
                    match op.map_err(|e| e.to_string())? {
                        Operator::CallIndirect { .. }
                        | Operator::ReturnCallIndirect { .. }
                        | Operator::CallRef { .. }
                        | Operator::ReturnCallRef { .. } => indirect += 1,
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    Ok(Report { imports, indirect })
}
