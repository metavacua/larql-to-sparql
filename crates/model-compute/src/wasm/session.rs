//! Per-call session — fresh Store with fuel/memory caps, implements the
//! alloc-write-solve-read ABI over a compiled `Module`.

use wasmi::{Engine, Instance, Memory, Module, Store, StoreLimits, StoreLimitsBuilder};

use super::error::SolverError;
use super::runtime::SolverLimits;

// Guest ABI export names.
const WASM_MEMORY: &str = "memory";
const WASM_ALLOC: &str = "alloc";
const WASM_SOLVE: &str = "solve";
const WASM_SOLUTION_PTR: &str = "solution_ptr";
const WASM_SOLUTION_LEN: &str = "solution_len";

pub struct Session<'m> {
    store: Store<State>,
    instance: Instance,
    _module: &'m Module,
}

struct State {
    limits: StoreLimits,
}

impl<'m> Session<'m> {
    pub(crate) fn new(
        engine: &Engine,
        module: &'m Module,
        limits: SolverLimits,
    ) -> Result<Self, SolverError> {
        let page_bytes = (limits.memory_pages as usize) * 64 * 1024;
        let store_limits = StoreLimitsBuilder::new().memory_size(page_bytes).build();
        let mut store = Store::new(
            engine,
            State {
                limits: store_limits,
            },
        );
        store.limiter(|s: &mut State| &mut s.limits);
        store
            .set_fuel(limits.fuel)
            .map_err(|e| SolverError::Engine(e.to_string()))?;

        let linker = wasmi::Linker::<State>::new(engine);
        let instance = linker
            .instantiate_and_start(&mut store, module)
            .map_err(|e| SolverError::Instantiate(e.to_string()))?;

        Ok(Self {
            store,
            instance,
            _module: module,
        })
    }

    /// Fuel remaining after the last call.
    pub fn fuel_remaining(&mut self) -> u64 {
        self.store.get_fuel().unwrap_or(0)
    }

    /// Run one solve call with the canonical alloc-write-solve-read ABI.
    pub fn solve(&mut self, input: &[u8]) -> Result<Vec<u8>, SolverError> {
        let memory = self.memory()?;

        let alloc = self
            .instance
            .get_typed_func::<u32, i32>(&self.store, WASM_ALLOC)
            .map_err(|_| SolverError::MissingExport(WASM_ALLOC.into()))?;
        let solve = self
            .instance
            .get_typed_func::<(i32, u32), u32>(&self.store, WASM_SOLVE)
            .map_err(|_| SolverError::MissingExport(WASM_SOLVE.into()))?;
        let sol_ptr = self
            .instance
            .get_typed_func::<(), i32>(&self.store, WASM_SOLUTION_PTR)
            .map_err(|_| SolverError::MissingExport(WASM_SOLUTION_PTR.into()))?;
        let sol_len = self
            .instance
            .get_typed_func::<(), u32>(&self.store, WASM_SOLUTION_LEN)
            .map_err(|_| SolverError::MissingExport(WASM_SOLUTION_LEN.into()))?;

        // 1. alloc(len) — guest reserves input buffer
        let input_len = input.len() as u32;
        let in_ptr = alloc
            .call(&mut self.store, input_len)
            .map_err(|e| trap_or_fuel(WASM_ALLOC, e))?;
        let in_ptr_usize = checked_ptr(in_ptr, input.len(), &memory, &self.store)?;

        // 2. write input to guest memory
        memory.data_mut(&mut self.store)[in_ptr_usize..in_ptr_usize + input.len()]
            .copy_from_slice(input);

        // 3. solve(ptr, len)
        let status = solve
            .call(&mut self.store, (in_ptr, input_len))
            .map_err(|e| trap_or_fuel(WASM_SOLVE, e))?;
        if status != 0 {
            return Err(SolverError::SolveFailed(status));
        }

        // 4. read solution_ptr + solution_len, copy output out
        let out_ptr = sol_ptr
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel(WASM_SOLUTION_PTR, e))?;
        let out_len = sol_len
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel(WASM_SOLUTION_LEN, e))?;

        let out_ptr_usize = checked_ptr(out_ptr, out_len as usize, &memory, &self.store)?;
        let out =
            memory.data(&self.store)[out_ptr_usize..out_ptr_usize + out_len as usize].to_vec();
        Ok(out)
    }

    fn memory(&self) -> Result<Memory, SolverError> {
        self.instance
            .get_memory(&self.store, WASM_MEMORY)
            .ok_or_else(|| SolverError::MissingExport(WASM_MEMORY.into()))
    }
}

fn checked_ptr(
    ptr: i32,
    len: usize,
    memory: &Memory,
    store: &Store<State>,
) -> Result<usize, SolverError> {
    if ptr < 0 {
        return Err(SolverError::InvalidGuestPointer(format!(
            "negative pointer: {}",
            ptr
        )));
    }
    let start = ptr as usize;
    let end = start.checked_add(len).ok_or_else(|| {
        SolverError::InvalidGuestPointer(format!("ptr {} + len {} overflows", ptr, len))
    })?;
    let size = memory.data(store).len();
    if end > size {
        return Err(SolverError::InvalidGuestPointer(format!(
            "ptr {} + len {} exceeds memory size {}",
            ptr, len, size
        )));
    }
    Ok(start)
}

fn trap_or_fuel(call: &str, e: wasmi::Error) -> SolverError {
    let msg = e.to_string();
    if msg.contains("fuel") || msg.contains("out of fuel") {
        return SolverError::FuelExhausted { budget: 0 };
    }
    SolverError::Trap {
        call: call.into(),
        trap: msg,
    }
}
