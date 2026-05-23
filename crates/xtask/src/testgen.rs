//! Structural wasm32 test generator.
//!
//! Reads `WasmFacts::exports_typed` and emits `#[wasm_bindgen_test]` Rust
//! source covering every exported function with scalar (I32/I64/F32/F64)
//! parameters.  Non-scalar params (Ref, V128) and wasm-bindgen internals are
//! skipped; the caller decides where to write the output.
//!
//! Coverage: Node.js mode (default, no configure macro).
//! Portability: set `WASM_BINDGEN_TEST_BROWSER=firefox` at invocation time.

use wasmparser::ValType;

use crate::wasm_facts::WasmFacts;

pub struct TestCase {
    pub fn_name: String,
    pub export_name: String,
    pub args: Vec<ArgValue>,
    pub result_ty: Option<ValType>,
}

pub enum ArgValue {
    I32(i32),
    I64(i64),
    F32(f32),
    F64(f64),
}

fn canonical_vals(ty: &ValType) -> Vec<ArgValue> {
    match ty {
        ValType::I32 => vec![
            ArgValue::I32(0),
            ArgValue::I32(1),
            ArgValue::I32(-1),
            ArgValue::I32(i32::MAX),
            ArgValue::I32(i32::MIN),
        ],
        ValType::I64 => vec![
            ArgValue::I64(0),
            ArgValue::I64(1),
            ArgValue::I64(-1),
            ArgValue::I64(i64::MAX),
            ArgValue::I64(i64::MIN),
        ],
        ValType::F32 => vec![
            ArgValue::F32(0.0_f32),
            ArgValue::F32(1.0_f32),
            ArgValue::F32(-1.0_f32),
            ArgValue::F32(f32::MAX),
            ArgValue::F32(f32::MIN_POSITIVE),
        ],
        ValType::F64 => vec![
            ArgValue::F64(0.0_f64),
            ArgValue::F64(1.0_f64),
            ArgValue::F64(-1.0_f64),
            ArgValue::F64(f64::MAX),
            ArgValue::F64(f64::MIN_POSITIVE),
        ],
        _ => vec![],
    }
}

fn is_scalar(ty: &ValType) -> bool {
    matches!(ty, ValType::I32 | ValType::I64 | ValType::F32 | ValType::F64)
}

fn is_rust_ident(s: &str) -> bool {
    !s.is_empty()
        && s.starts_with(|c: char| c.is_ascii_alphabetic() || c == '_')
        && s.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
        && !s.starts_with("__")
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { '_' })
        .collect()
}

fn arg_suffix(v: &ArgValue) -> String {
    let raw = match v {
        ArgValue::I32(n) => format!("{n}"),
        ArgValue::I64(n) => format!("{n}"),
        ArgValue::F32(n) => {
            if *n == 0.0 { "0".into() }
            else if *n == 1.0 { "1".into() }
            else if *n == -1.0 { "neg1".into() }
            else if n.is_nan() { "nan".into() }
            else if n.is_infinite() { if *n > 0.0 { "inf".into() } else { "neginf".into() } }
            else if *n == f32::MAX { "max".into() }
            else if *n == f32::MIN_POSITIVE { "minpos".into() }
            else { format!("{}", n.to_bits()) }
        }
        ArgValue::F64(n) => {
            if *n == 0.0 { "0".into() }
            else if *n == 1.0 { "1".into() }
            else if *n == -1.0 { "neg1".into() }
            else if n.is_nan() { "nan".into() }
            else if n.is_infinite() { if *n > 0.0 { "inf".into() } else { "neginf".into() } }
            else if *n == f64::MAX { "max".into() }
            else if *n == f64::MIN_POSITIVE { "minpos".into() }
            else { format!("{}", n.to_bits()) }
        }
    };
    // Replace any character invalid in a Rust identifier with 'n'.
    raw.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '_' { c } else { 'n' })
        .collect()
}

fn arg_literal(v: &ArgValue) -> String {
    match v {
        ArgValue::I32(n) => format!("{n}_i32"),
        ArgValue::I64(n) => format!("{n}_i64"),
        ArgValue::F32(n) => {
            if n.is_nan() { "f32::NAN".into() }
            else if n.is_infinite() {
                if *n > 0.0 { "f32::INFINITY".into() } else { "f32::NEG_INFINITY".into() }
            } else if *n == f32::MAX { "f32::MAX".into() }
            else if *n == f32::MIN_POSITIVE { "f32::MIN_POSITIVE".into() }
            else { format!("{n:.1}_f32") }
        }
        ArgValue::F64(n) => {
            if n.is_nan() { "f64::NAN".into() }
            else if n.is_infinite() {
                if *n > 0.0 { "f64::INFINITY".into() } else { "f64::NEG_INFINITY".into() }
            } else if *n == f64::MAX { "f64::MAX".into() }
            else if *n == f64::MIN_POSITIVE { "f64::MIN_POSITIVE".into() }
            else { format!("{n:.1}_f64") }
        }
    }
}

/// Generate structural test cases from wasm exports.
pub fn generate(facts: &WasmFacts) -> Vec<TestCase> {
    let mut cases = Vec::new();

    for (export_name, _func_idx, params, results) in &facts.exports_typed {
        if !is_rust_ident(export_name) {
            continue;
        }
        if params.iter().any(|t| !is_scalar(t)) {
            continue;
        }

        let result_ty = results.first().filter(|t| is_scalar(t)).cloned();
        let base = sanitize(export_name);

        if params.is_empty() {
            cases.push(TestCase {
                fn_name: format!("wgen_{base}_smoke"),
                export_name: export_name.clone(),
                args: vec![],
                result_ty,
            });
            continue;
        }

        // Diagonal sampling: zip canonical values across all params.
        // N rows where N = number of canonical values for the first param type.
        let rows: Vec<Vec<ArgValue>> = {
            let col_vals: Vec<Vec<ArgValue>> = params.iter().map(canonical_vals).collect();
            let row_count = col_vals[0].len();
            (0..row_count)
                .map(|row| {
                    col_vals
                        .iter()
                        .map(|col| col[row.min(col.len() - 1)].clone())
                        .collect()
                })
                .collect()
        };

        for (i, row) in rows.into_iter().enumerate().take(50) {
            let suffix = row.iter().map(arg_suffix).collect::<Vec<_>>().join("_");
            cases.push(TestCase {
                fn_name: format!("wgen_{base}_{i}_{suffix}"),
                export_name: export_name.clone(),
                args: row,
                result_ty: result_ty.clone(),
            });
        }
    }

    cases
}

/// Emit a `tests/wasm_generated.rs` source file.
pub fn emit_test_file(crate_name: &str, cases: &[TestCase]) -> String {
    let crate_ident = crate_name.replace('-', "_");
    let mut out = String::new();

    out.push_str("// AUTO-GENERATED by `cargo xtask wasm-testgen` — do not edit.\n");
    out.push_str(&format!(
        "// Regenerate: cargo xtask wasm-testgen --package {crate_name}\n"
    ));
    out.push_str("//\n");
    out.push_str(&format!(
        "// Node.js (coverage): cargo test -p {crate_name} --target wasm32-unknown-unknown\n"
    ));
    out.push_str(&format!(
        "// Firefox (portability): WASM_BINDGEN_TEST_BROWSER=firefox cargo test -p {crate_name} --target wasm32-unknown-unknown\n"
    ));
    out.push('\n');
    out.push_str("#![cfg(target_arch = \"wasm32\")]\n");
    out.push('\n');

    if cases.is_empty() {
        out.push_str("// No testable exports found (no scalar-typed public exports).\n");
        return out;
    }

    out.push_str("use wasm_bindgen_test::*;\n");
    out.push_str(&format!("use {crate_ident}::*;\n"));
    out.push('\n');

    for case in cases {
        let args_lit = case.args.iter().map(|a| arg_literal(a)).collect::<Vec<_>>().join(", ");
        let call = format!("{}({})", case.export_name, args_lit);
        let body = match &case.result_ty {
            Some(_) => format!("let _ = {call};"),
            None => format!("{call};"),
        };

        out.push_str("#[wasm_bindgen_test]\n");
        out.push_str(&format!("fn {}() {{ {body} }}\n", case.fn_name));
    }

    out
}

/// Build, generate, and write `tests/wasm_generated.rs` given an already-built wasm binary.
///
/// Called by `certify.rs` after the production cdylib is compiled, so testgen
/// reuses the artifact rather than triggering a second build.
pub fn generate_for_package(
    package: &str,
    wasm_bytes: &[u8],
    crate_root: &std::path::Path,
) -> anyhow::Result<usize> {
    use anyhow::Context as _;
    let facts = crate::wasm_facts::extract(wasm_bytes)?;
    let cases = generate(&facts);
    let source = emit_test_file(package, &cases);
    let tests_dir = crate_root.join("tests");
    std::fs::create_dir_all(&tests_dir)
        .with_context(|| format!("creating {}", tests_dir.display()))?;
    let dest = tests_dir.join("wasm_generated.rs");
    std::fs::write(&dest, &source)
        .with_context(|| format!("writing {}", dest.display()))?;
    Ok(cases.len())
}

// Allow cloning ArgValue so diagonal rows can be built from &Vec<ArgValue>.
impl Clone for ArgValue {
    fn clone(&self) -> Self {
        match self {
            Self::I32(v) => Self::I32(*v),
            Self::I64(v) => Self::I64(*v),
            Self::F32(v) => Self::F32(*v),
            Self::F64(v) => Self::F64(*v),
        }
    }
}
