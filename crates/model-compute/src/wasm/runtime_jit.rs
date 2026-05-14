//! Engine + compiled-module management — wasmtime JIT/AOT path.
//!
//! This module preserves the original wasmtime implementation for REUSE
//! compliance. It is gated behind the `wasm-jit` feature and compiles only
//! on platforms where Cranelift is available (Linux, macOS, Windows, FreeBSD).
//! The universal default is `runtime.rs` (wasmi pure-Rust interpreter).
#![cfg(all(
    feature = "wasm-jit",
    any(
        target_os = "linux",
        target_os = "macos",
        target_os = "windows",
        target_os = "freebsd"
    )
))]

use wasmtime::{Config, Engine, Module};

use super::error::SolverError;
use super::session_jit::Session;

/// Per-call resource budget. Defaults: 100M fuel units, 256 linear-memory
/// pages (16 MiB). CP-SAT solver demo uses ~2M fuel for 9×9 Sudoku.
#[derive(Debug, Clone, Copy)]
pub struct SolverLimits {
    pub fuel: u64,
    pub memory_pages: u32,
}

impl Default for SolverLimits {
    fn default() -> Self {
        Self {
            fuel: 100_000_000,
            memory_pages: 256,
        }
    }
}

pub struct SolverRuntime {
    engine: Engine,
    limits: SolverLimits,
}

impl SolverRuntime {
    pub fn new() -> Result<Self, SolverError> {
        Self::with_limits(SolverLimits::default())
    }

    pub fn with_limits(limits: SolverLimits) -> Result<Self, SolverError> {
        let mut config = Config::new();
        config.consume_fuel(true);
        let engine = Engine::new(&config).map_err(|e| SolverError::Engine(e.to_string()))?;
        Ok(Self { engine, limits })
    }

    pub fn limits(&self) -> SolverLimits {
        self.limits
    }

    pub fn engine(&self) -> &Engine {
        &self.engine
    }

    /// Compile a `.wasm` binary into a reusable module.
    pub fn compile(&self, wasm: &[u8]) -> Result<Module, SolverError> {
        Module::new(&self.engine, wasm).map_err(|e| SolverError::InvalidModule(e.to_string()))
    }

    /// Open a fresh session backed by this runtime's engine and limits.
    /// Each session has an independent store — no state bleeds between calls.
    pub fn session<'m>(&self, module: &'m Module) -> Result<Session<'m>, SolverError> {
        Session::new(&self.engine, module, self.limits)
    }
}
