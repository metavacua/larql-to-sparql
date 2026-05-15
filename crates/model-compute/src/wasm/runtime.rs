//! Engine + compiled-module management. The runtime holds a long-lived
//! `wasmi::Engine`; each `Session::new` creates a fresh `Store` with
//! the configured limits so calls are isolated.

use wasmi::{Config, Engine, Module};

use super::error::SolverError;
use super::session::Session;

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
        let mut config = Config::default();
        config.consume_fuel(true);
        let engine = Engine::new(&config);
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

#[cfg(test)]
mod tests {
    use super::SolverRuntime;

    // Verify that the engine produced by SolverRuntime has fuel metering
    // enabled. set_fuel() errors immediately if Config::consume_fuel(true)
    // was not set; catching that here gives a single clear failure point
    // instead of cryptic panics spread across every wasm_roundtrip test.
    #[test]
    fn engine_fuel_metering_is_enabled() {
        struct Dummy;
        let runtime = SolverRuntime::new().expect("runtime");
        let mut store = wasmi::Store::new(runtime.engine(), Dummy);
        store
            .set_fuel(1_000)
            .expect("Engine must have fuel metering enabled (Config::consume_fuel(true))");
    }
}
