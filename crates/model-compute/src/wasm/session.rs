//! Per-call session — fresh Store with fuel/memory caps, implements the
//! alloc-write-solve-read ABI over a compiled `Module`.

use wasmtime::{Engine, Instance, Memory, Module, Store, StoreLimits, StoreLimitsBuilder, TypedFunc};

use super::error::SolverError;
use super::runtime::SolverLimits;

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
        let store_limits = StoreLimitsBuilder::new()
            .memory_size(page_bytes)
            .build();
        let mut store = Store::new(engine, State { limits: store_limits });
        store.limiter(|s: &mut State| &mut s.limits);
        store
            .set_fuel(limits.fuel)
            .map_err(|e| SolverError::Engine(e.to_string()))?;

        let instance = Instance::new(&mut store, module, &[])
            .map_err(|e| SolverError::Instantiate(e.to_string()))?;

        Ok(Self { store, instance, _module: module })
    }

    /// Fuel remaining. Useful for tests and for callers who want to
    /// observe the cost of a solve.
    pub fn fuel_remaining(&mut self) -> u64 {
        self.store.get_fuel().unwrap_or(0)
    }

    /// Run one solve call with the canonical alloc-write-solve-read ABI.
    pub fn solve(&mut self, input: &[u8]) -> Result<Vec<u8>, SolverError> {
        let memory = self.memory()?;

        let alloc: TypedFunc<u32, i32> = self
            .instance
            .get_typed_func::<u32, i32>(&mut self.store, "alloc")
            .map_err(|_| SolverError::MissingExport("alloc".into()))?;
        let solve: TypedFunc<(i32, u32), u32> = self
            .instance
            .get_typed_func::<(i32, u32), u32>(&mut self.store, "solve")
            .map_err(|_| SolverError::MissingExport("solve".into()))?;
        let sol_ptr: TypedFunc<(), i32> = self
            .instance
            .get_typed_func::<(), i32>(&mut self.store, "solution_ptr")
            .map_err(|_| SolverError::MissingExport("solution_ptr".into()))?;
        let sol_len: TypedFunc<(), u32> = self
            .instance
            .get_typed_func::<(), u32>(&mut self.store, "solution_len")
            .map_err(|_| SolverError::MissingExport("solution_len".into()))?;

        // 1. alloc(len) — guest reserves input buffer
        let input_len = input.len() as u32;
        let in_ptr = alloc
            .call(&mut self.store, input_len)
            .map_err(|e| trap_or_fuel("alloc", e))?;
        let in_ptr_usize = checked_ptr(in_ptr, input.len(), &memory, &mut self.store)?;

        // 2. write input to guest memory
        memory
            .write(&mut self.store, in_ptr_usize, input)
            .map_err(|e| SolverError::InvalidGuestPointer(e.to_string()))?;

        // 3. solve(ptr, len)
        let status = solve
            .call(&mut self.store, (in_ptr, input_len))
            .map_err(|e| trap_or_fuel("solve", e))?;
        if status != 0 {
            return Err(SolverError::SolveFailed(status));
        }

        // 4. read solution_ptr + solution_len, copy output out
        let out_ptr = sol_ptr
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel("solution_ptr", e))?;
        let out_len = sol_len
            .call(&mut self.store, ())
            .map_err(|e| trap_or_fuel("solution_len", e))?;

        let out_ptr_usize = checked_ptr(out_ptr, out_len as usize, &memory, &mut self.store)?;
        let mut out = vec![0u8; out_len as usize];
        memory
            .read(&self.store, out_ptr_usize, &mut out)
            .map_err(|e| SolverError::InvalidGuestPointer(e.to_string()))?;
        Ok(out)
    }

    fn memory(&mut self) -> Result<Memory, SolverError> {
        self.instance
            .get_memory(&mut self.store, "memory")
            .ok_or_else(|| SolverError::MissingExport("memory".into()))
    }
}

fn checked_ptr(
    ptr: i32,
    len: usize,
    memory: &Memory,
    store: &mut Store<State>,
) -> Result<usize, SolverError> {
    if ptr < 0 {
        return Err(SolverError::InvalidGuestPointer(format!("negative pointer: {}", ptr)));
    }
    let start = ptr as usize;
    let end = start.checked_add(len).ok_or_else(|| {
        SolverError::InvalidGuestPointer(format!("ptr {} + len {} overflows", ptr, len))
    })?;
    let size = memory.data_size(&mut *store);
    if end > size {
        return Err(SolverError::InvalidGuestPointer(format!(
            "ptr {} + len {} exceeds memory size {}",
            ptr, len, size
        )));
    }
    Ok(start)
}

fn trap_or_fuel(call: &str, e: wasmtime::Error) -> SolverError {
    let msg = e.to_string();
    if msg.contains("fuel") || msg.contains("out of fuel") {
        return SolverError::FuelExhausted { budget: 0 };
    }
    SolverError::Trap { call: call.into(), trap: msg }
}
