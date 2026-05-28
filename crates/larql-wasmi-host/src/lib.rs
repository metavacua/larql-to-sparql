//! Wasmi-based host runtime for the `larql-wasm32v1-none` module.
//!
//! Mirrors the `model-compute` `SolverRuntime` + `Session` pattern exactly.
//! Each session has an isolated `Store` (no state bleeds between calls).
//!
//! # Example
//!
//! ```no_run
//! use larql_wasmi_host::{LarqlCoreRuntime, wire::{LayerData, Dtype}};
//!
//! let wasm_bytes = std::fs::read("larql-wasm32v1-none.wasm").unwrap();
//! let runtime = LarqlCoreRuntime::new().unwrap();
//! let module = runtime.compile(&wasm_bytes).unwrap();
//! let mut session = runtime.session(&module).unwrap();
//!
//! // gate_knn: hidden_size=2, one layer with 2 features, query=[1,0], k=1
//! let gate: Vec<u8> = [1.0f32, 0.0, 0.0, 1.0].iter()
//!     .flat_map(|f| f.to_le_bytes()).collect();
//! let results = session.gate_knn(
//!     2,
//!     &[Some(LayerData { bytes: &gate, num_features: 2, dtype: Dtype::F32 })],
//!     0, &[1.0, 0.0], 1,
//! ).unwrap();
//! assert_eq!(results[0].feature, 0);
//! ```

pub mod error;
pub mod wire;

pub use error::LarqlHostError;
pub use wire::{Dtype, KnnResult, LayerData};

use wasmi::{Config, Engine, Instance, Linker, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc};

// ABI export names — identical to model-compute's constants
const WASM_MEMORY: &str = "memory";
const WASM_ALLOC: &str = "alloc";
const WASM_DEALLOC: &str = "dealloc";
const WASM_SOLVE: &str = "solve";
const WASM_SOLUTION_PTR: &str = "solution_ptr";
const WASM_SOLUTION_LEN: &str = "solution_len";

// ── Limits ────────────────────────────────────────────────────────────────────

/// Per-session resource budget.
///
/// Defaults: 1 billion fuel units, 512 linear-memory pages (32 MiB).
/// Increase memory pages for larger gate indexes.
#[derive(Debug, Clone, Copy)]
pub struct LarqlLimits {
    pub fuel: u64,
    pub memory_pages: u32,
}

impl Default for LarqlLimits {
    fn default() -> Self {
        Self {
            fuel: 1_000_000_000,
            memory_pages: 512,
        }
    }
}

// ── Runtime ───────────────────────────────────────────────────────────────────

/// Long-lived engine + limits. Clone-cheap (Engine is ref-counted internally).
pub struct LarqlCoreRuntime {
    engine: Engine,
    limits: LarqlLimits,
}

impl LarqlCoreRuntime {
    pub fn new() -> Result<Self, LarqlHostError> {
        Self::with_limits(LarqlLimits::default())
    }

    pub fn with_limits(limits: LarqlLimits) -> Result<Self, LarqlHostError> {
        let mut config = Config::default();
        config.consume_fuel(true);
        Ok(Self {
            engine: Engine::new(&config),
            limits,
        })
    }

    /// Compile a `.wasm` binary into a reusable `Module`.
    pub fn compile(&self, wasm: &[u8]) -> Result<Module, LarqlHostError> {
        Module::new(&self.engine, wasm)
            .map_err(|e| LarqlHostError::InvalidModule(e.to_string()))
    }

    /// Open a fresh `LarqlCoreSession` backed by this runtime.
    pub fn session<'m>(&self, module: &'m Module) -> Result<LarqlCoreSession<'m>, LarqlHostError> {
        LarqlCoreSession::new(&self.engine, module, self.limits)
    }

    pub fn limits(&self) -> LarqlLimits {
        self.limits
    }
}

// ── Session ───────────────────────────────────────────────────────────────────

/// Per-call session — fresh `Store` with fuel/memory caps.
pub struct LarqlCoreSession<'m> {
    store: Store<State>,
    instance: Instance,
    _module: &'m Module,
    // Cached ABI typed funcs — stable after instantiation, resolved once.
    alloc_fn: TypedFunc<u32, i32>,
    solve_fn: TypedFunc<(i32, u32), u32>,
    sol_ptr_fn: TypedFunc<(), i32>,
    sol_len_fn: TypedFunc<(), u32>,
    dealloc_fn: Option<TypedFunc<(i32, u32), ()>>,
}

struct State {
    limits: StoreLimits,
}

impl<'m> LarqlCoreSession<'m> {
    fn new(engine: &Engine, module: &'m Module, limits: LarqlLimits) -> Result<Self, LarqlHostError> {
        let page_bytes = (limits.memory_pages as usize) * 64 * 1024;
        let store_limits = StoreLimitsBuilder::new().memory_size(page_bytes).build();
        let mut store = Store::new(engine, State { limits: store_limits });
        store.limiter(|s| &mut s.limits);
        store
            .set_fuel(limits.fuel)
            .map_err(|e| LarqlHostError::Instantiate(e.to_string()))?;

        let linker = Linker::<State>::new(engine);
        let instance = linker
            .instantiate_and_start(&mut store, module)
            .map_err(|e| LarqlHostError::Instantiate(e.to_string()))?;

        let alloc_fn = instance
            .get_typed_func::<u32, i32>(&store, WASM_ALLOC)
            .map_err(|_| LarqlHostError::MissingExport(WASM_ALLOC.into()))?;
        let solve_fn = instance
            .get_typed_func::<(i32, u32), u32>(&store, WASM_SOLVE)
            .map_err(|_| LarqlHostError::MissingExport(WASM_SOLVE.into()))?;
        let sol_ptr_fn = instance
            .get_typed_func::<(), i32>(&store, WASM_SOLUTION_PTR)
            .map_err(|_| LarqlHostError::MissingExport(WASM_SOLUTION_PTR.into()))?;
        let sol_len_fn = instance
            .get_typed_func::<(), u32>(&store, WASM_SOLUTION_LEN)
            .map_err(|_| LarqlHostError::MissingExport(WASM_SOLUTION_LEN.into()))?;
        let dealloc_fn = instance.get_typed_func::<(i32, u32), ()>(&store, WASM_DEALLOC).ok();

        Ok(Self { store, instance, _module: module, alloc_fn, solve_fn, sol_ptr_fn, sol_len_fn, dealloc_fn })
    }

    /// Fuel remaining after the last call.
    pub fn fuel_remaining(&mut self) -> u64 {
        self.store.get_fuel().unwrap_or(0)
    }

    // ── High-level ops ───────────────────────────────────────────────────────

    /// Run a `gate_knn` request through the wasm boundary.
    ///
    /// * `hidden_size` — per-vector dimension
    /// * `layers` — raw gate bytes per layer; `None` entries are transmitted as absent
    /// * `query_layer` — index of the layer to query
    /// * `query` — query vector (length `hidden_size`)
    /// * `k` — number of nearest neighbours
    pub fn gate_knn(
        &mut self,
        hidden_size: u32,
        layers: &[Option<LayerData<'_>>],
        query_layer: u32,
        query: &[f32],
        k: u32,
    ) -> Result<Vec<KnnResult>, LarqlHostError> {
        let request = wire::encode_gate_knn(hidden_size, layers, query_layer, query, k);
        let response = self.call_solve(&request)?;
        wire::decode_gate_knn(&response)
            .map_err(|e| LarqlHostError::MalformedResponse(e))
    }

    /// Compute `dot(a, b)` in the wasm sandbox.
    pub fn dot(&mut self, a: &[f32], b: &[f32]) -> Result<f32, LarqlHostError> {
        let response = self.call_solve(&wire::encode_dot(a, b))?;
        wire::decode_scalar(&response).map_err(|e| LarqlHostError::MalformedResponse(e))
    }

    /// Compute `norm(a)` in the wasm sandbox.
    pub fn norm(&mut self, a: &[f32]) -> Result<f32, LarqlHostError> {
        let response = self.call_solve(&wire::encode_norm(a))?;
        wire::decode_scalar(&response).map_err(|e| LarqlHostError::MalformedResponse(e))
    }

    /// Compute `cosine(a, b)` in the wasm sandbox.
    pub fn cosine(&mut self, a: &[f32], b: &[f32]) -> Result<f32, LarqlHostError> {
        let response = self.call_solve(&wire::encode_cosine(a, b))?;
        wire::decode_scalar(&response).map_err(|e| LarqlHostError::MalformedResponse(e))
    }

    // ── Low-level ABI call ───────────────────────────────────────────────────

    /// Canonical alloc-write-solve-read ABI call.
    fn call_solve(&mut self, input: &[u8]) -> Result<Vec<u8>, LarqlHostError> {
        let memory = self.memory()?;

        // 1. alloc(len) — reserve input buffer in guest
        let in_len = input.len() as u32;
        let in_ptr = self.alloc_fn
            .call(&mut self.store, in_len)
            .map_err(|e| trap_or_fuel(WASM_ALLOC, e))?;
        let in_offset = checked_ptr(in_ptr, input.len(), &memory, &self.store)?;

        // 2. write input bytes into guest memory
        memory.data_mut(&mut self.store)[in_offset..in_offset + input.len()]
            .copy_from_slice(input);

        // 3. solve(ptr, len) — guest processes and writes response
        let status = self.solve_fn
            .call(&mut self.store, (in_ptr, in_len))
            .map_err(|e| trap_or_fuel(WASM_SOLVE, e))?;
        if status != 0 {
            return Err(LarqlHostError::SolveFailed(status));
        }

        // 4. read solution_ptr + solution_len, copy output
        let out_ptr = self.sol_ptr_fn
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel(WASM_SOLUTION_PTR, e))?;
        let out_len = self.sol_len_fn
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel(WASM_SOLUTION_LEN, e))?;

        let out_offset = checked_ptr(out_ptr, out_len as usize, &memory, &self.store)?;
        let out = memory.data(&self.store)[out_offset..out_offset + out_len as usize].to_vec();

        // 5. free the input buffer (cached dealloc, absent means input leaks intentionally)
        if let Some(dealloc_fn) = self.dealloc_fn {
            let _ = dealloc_fn.call(&mut self.store, (in_ptr, in_len));
        }

        Ok(out)
    }

    fn memory(&self) -> Result<Memory, LarqlHostError> {
        self.instance
            .get_memory(&self.store, WASM_MEMORY)
            .ok_or_else(|| LarqlHostError::MissingExport(WASM_MEMORY.into()))
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn checked_ptr(
    ptr: i32,
    len: usize,
    memory: &Memory,
    store: &Store<State>,
) -> Result<usize, LarqlHostError> {
    if ptr <= 0 {
        return Err(LarqlHostError::InvalidGuestPointer(format!("ptr={ptr}")));
    }
    let start = ptr as usize;
    let end = start
        .checked_add(len)
        .ok_or_else(|| LarqlHostError::InvalidGuestPointer(format!("ptr {ptr} + len {len} overflows")))?;
    let mem_size = memory.data(store).len();
    if end > mem_size {
        return Err(LarqlHostError::InvalidGuestPointer(format!(
            "ptr {ptr} + len {len} exceeds memory size {mem_size}"
        )));
    }
    Ok(start)
}

fn trap_or_fuel(call: &str, e: wasmi::Error) -> LarqlHostError {
    let msg = e.to_string();
    if msg.contains("fuel") || msg.contains("out of fuel") {
        return LarqlHostError::FuelExhausted { budget: 0 };
    }
    LarqlHostError::Trap {
        call: call.into(),
        trap: msg,
    }
}
