/// Wasm compilation target variants.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WasmTarget {
    /// WebAssembly 1.0 MVP + mutable-globals only. No GC, SIMD, threads, bulk-memory.
    Wasm32V1None,
    /// Standard `wasm32-unknown-unknown`. Enables bulk-memory and other post-MVP proposals.
    Wasm32Unknown,
    /// WASI systems interface (`wasm32-wasip1`).
    Wasm32Wasi,
}

/// Compilation status for a (language, target) pair.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MatrixStatus {
    /// Compiles and produces correct output on this target.
    Pass,
    /// Compiles with noted constraints.
    Partial(&'static str),
    /// Cannot compile to this target; reason given.
    Fail(&'static str),
    /// No compilation model (query language or interpreted only).
    NotApplicable(&'static str),
}

/// One entry in the compilation strategy matrix.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatrixEntry {
    pub language: &'static str,
    pub target: WasmTarget,
    pub status: MatrixStatus,
}

/// Documented compilation strategy matrix.
///
/// CI enforces all `Pass` entries. `Fail` and `Partial` entries document
/// known limitations. `NotApplicable` entries are query/schema languages.
///
/// Introspection/reflection capabilities (Java reflection, Python `inspect`,
/// TypeScript compiler API, SPARQL/GraphQL introspection) are tracked as the
/// "native path" — currently future work, not yet enforced by CI.
pub const COMPILATION_MATRIX: &[MatrixEntry] = &[
    // Rust — full coverage
    MatrixEntry { language: "rust", target: WasmTarget::Wasm32V1None, status: MatrixStatus::Pass },
    MatrixEntry { language: "rust", target: WasmTarget::Wasm32Unknown, status: MatrixStatus::Pass },
    MatrixEntry { language: "rust", target: WasmTarget::Wasm32Wasi,    status: MatrixStatus::Pass },
    // AssemblyScript (TypeScript subset)
    MatrixEntry { language: "assemblyscript", target: WasmTarget::Wasm32Unknown, status: MatrixStatus::Pass },
    MatrixEntry {
        language: "assemblyscript",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::Partial("--disable bulk-memory required; not all AS programs compile"),
    },
    // WAT / Wasm binary
    MatrixEntry { language: "wat",  target: WasmTarget::Wasm32V1None, status: MatrixStatus::Pass },
    MatrixEntry { language: "wasm", target: WasmTarget::Wasm32V1None, status: MatrixStatus::Pass },
    // Java
    MatrixEntry {
        language: "java",
        target: WasmTarget::Wasm32Unknown,
        status: MatrixStatus::Partial("TeaVM or GraalVM native-image required; standard javac does not emit Wasm"),
    },
    MatrixEntry {
        language: "java",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::Fail("No production toolchain targets MVP; TeaVM uses bulk-memory"),
    },
    // Python
    MatrixEntry {
        language: "python",
        target: WasmTarget::Wasm32Unknown,
        status: MatrixStatus::Partial("Pyodide only; requires GC and exceptions Wasm proposals"),
    },
    MatrixEntry {
        language: "python",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::Fail("Requires GC and exceptions proposals absent from v1-none"),
    },
    // TypeScript (general — not AssemblyScript)
    MatrixEntry {
        language: "typescript",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::NotApplicable("TypeScript compiles to JavaScript; use AssemblyScript for Wasm"),
    },
    // Query / schema languages
    MatrixEntry {
        language: "sparql",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::NotApplicable("Query language; no compilation model"),
    },
    MatrixEntry {
        language: "graphql",
        target: WasmTarget::Wasm32V1None,
        status: MatrixStatus::NotApplicable("Schema/query language; no compilation model"),
    },
    // LARQL — larql-lql-core is Tier-0 and compiles for wasm32v1-none
    MatrixEntry { language: "larql", target: WasmTarget::Wasm32V1None, status: MatrixStatus::Pass },
];

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rust_wasm32v1none_is_pass() {
        let entry = COMPILATION_MATRIX
            .iter()
            .find(|e| e.language == "rust" && e.target == WasmTarget::Wasm32V1None)
            .expect("rust/wasm32v1-none must have a matrix entry");
        assert_eq!(entry.status, MatrixStatus::Pass);
    }

    #[test]
    fn larql_wasm32v1none_is_pass() {
        let entry = COMPILATION_MATRIX
            .iter()
            .find(|e| e.language == "larql" && e.target == WasmTarget::Wasm32V1None)
            .unwrap();
        assert_eq!(entry.status, MatrixStatus::Pass);
    }
}
