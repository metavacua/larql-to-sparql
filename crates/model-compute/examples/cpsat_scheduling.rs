//! Rust-native port of `experiments/07_wasm_compute/wasm_solver_demo_v11.py`
//! scheduling benchmark, using `model-compute::wasm` as the host runtime.
//!
//! Problem: assign N tasks to distinct time slots in [0, max_time-1],
//! minimise the largest slot used. With `N=5, max_time=10`, optimal
//! makespan = 4 (tasks go to slots 0..4).
//!
//! The WASM guest is the CP-SAT solver from
//! `experiments/07_wasm_compute/solver/` — the same 22 KB module that
//! demonstrated "constraint solving inside a transformer forward pass".
//! This example shows the host-side path in Rust: load module, encode
//! problem bytes, call solve, decode result.
//!
//! Run with:
//!   cargo run --example cpsat_scheduling -p model-compute --features wasm
//!
//! The example auto-discovers the prebuilt .wasm at
//! `experiments/07_wasm_compute/solver/target/wasm32-unknown-unknown/release/larql_wasm_solver.wasm`.
//! To rebuild the module:
//!   (cd experiments/07_wasm_compute/solver && cargo build --release --target wasm32-unknown-unknown)

#[cfg(not(feature = "wasm"))]
fn main() {
    eprintln!("This example requires the `wasm` feature. Re-run with --features wasm.");
}

#[cfg(feature = "wasm")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use std::time::Instant;

    use model_compute::wasm::SolverRuntime;

    let wasm_path = find_wasm()?;
    println!("Loading WASM solver: {}", wasm_path.display());
    let wasm_bytes = std::fs::read(&wasm_path)?;
    println!("  module size: {} bytes", wasm_bytes.len());

    let runtime = SolverRuntime::new()?;
    let compile_start = Instant::now();
    let module = runtime.compile(&wasm_bytes)?;
    println!("  compile time: {:.2} ms", compile_start.elapsed().as_secs_f64() * 1e3);

    // ── Problem: 5 tasks, each needs a distinct time slot in [0, 9] ──
    let n_tasks = 5;
    let max_time = 10;
    let problem = encode_scheduling_problem(n_tasks, max_time);
    println!("\nProblem: schedule {} tasks into distinct slots in [0, {}]", n_tasks, max_time - 1);
    println!("  payload size: {} bytes", problem.len());
    println!("  expected: all-different assignment, optimal makespan = {}", n_tasks - 1);

    // ── Solve ──
    let mut session = runtime.session(&module)?;
    let solve_start = Instant::now();
    let solution = session.solve(&problem)?;
    let solve_time = solve_start.elapsed();
    let fuel_remaining = session.fuel_remaining();
    println!("\nSolved in {:.2} ms", solve_time.as_secs_f64() * 1e3);
    println!("  fuel remaining: {}", fuel_remaining);

    // ── Decode result ──
    let (status, assignment) = decode_solution(&solution, n_tasks);
    let status_name = match status {
        0 => "FEASIBLE",
        1 => "INFEASIBLE",
        2 => "OPTIMAL",
        other => {
            println!("  status: unknown ({})", other);
            return Err(format!("unexpected status byte {}", other).into());
        }
    };
    println!("  status: {} ({})", status_name, status);

    if assignment.is_empty() {
        println!("  no solution returned");
        return Ok(());
    }

    print!("  assignment: [");
    for (i, slot) in assignment.iter().enumerate() {
        if i > 0 { print!(", "); }
        print!("task{}→slot{}", i, slot);
    }
    println!("]");

    let makespan = *assignment.iter().max().unwrap_or(&0);
    println!("  makespan: {}", makespan);

    // ── Verify ──
    let mut distinct = assignment.clone();
    distinct.sort_unstable();
    distinct.dedup();
    let all_different = distinct.len() == assignment.len();
    let optimal = makespan == (n_tasks as i32 - 1);
    println!("\nVerification:");
    println!("  all-different:   {}", if all_different { "PASS" } else { "FAIL" });
    println!("  optimal:         {}", if optimal { "PASS" } else { "FAIL" });

    Ok(())
}

#[cfg(feature = "wasm")]
fn find_wasm() -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    // Walk up from this file to the workspace root, then path to the
    // experiments/ prebuilt module. CARGO_MANIFEST_DIR points at model-compute.
    let manifest = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"));
    let workspace = manifest
        .parent()
        .and_then(|p| p.parent())
        .ok_or("failed to locate workspace root")?;
    let wasm = workspace.join(
        "experiments/07_wasm_compute/solver/target/wasm32-unknown-unknown/release/larql_wasm_solver.wasm",
    );
    if !wasm.exists() {
        return Err(format!(
            "WASM module not found at {}\n\
             Build it first:\n  (cd experiments/07_wasm_compute/solver && \\\n  \
             cargo build --release --target wasm32-unknown-unknown)",
            wasm.display()
        )
        .into());
    }
    Ok(wasm)
}

#[cfg(feature = "wasm")]
fn encode_scheduling_problem(n_tasks: usize, max_time: i32) -> Vec<u8> {
    // Binary protocol matches solver/src/lib.rs decode_problem:
    //   u32 n_vars | u32 n_constraints | u8 obj_type
    //   [u32 n_obj; u32 × n_obj] if obj_type == 1 (MinimizeMax)
    //   for each var: i32 lo | i32 hi
    //   for each constraint: u8 ctype | payload
    //
    // Layout: n_tasks variables, one all-different constraint,
    // minimize-max over all variables.
    let mut buf = Vec::new();

    // header
    buf.extend_from_slice(&(n_tasks as u32).to_le_bytes());
    buf.extend_from_slice(&1_u32.to_le_bytes()); // 1 constraint

    // objective = MinimizeMax over all vars
    buf.push(1_u8);
    buf.extend_from_slice(&(n_tasks as u32).to_le_bytes());
    for i in 0..n_tasks {
        buf.extend_from_slice(&(i as u32).to_le_bytes());
    }

    // variables: [0, max_time-1]
    for _ in 0..n_tasks {
        buf.extend_from_slice(&0_i32.to_le_bytes());
        buf.extend_from_slice(&(max_time - 1).to_le_bytes());
    }

    // constraint: all-different across all vars
    buf.push(4_u8);
    buf.extend_from_slice(&(n_tasks as u32).to_le_bytes());
    for i in 0..n_tasks {
        buf.extend_from_slice(&(i as u32).to_le_bytes());
    }

    buf
}

#[cfg(feature = "wasm")]
fn decode_solution(buf: &[u8], n_tasks: usize) -> (u8, Vec<i32>) {
    if buf.is_empty() {
        return (255, Vec::new());
    }
    let status = buf[0];
    if status != 0 && status != 2 {
        return (status, Vec::new());
    }
    let mut assignment = Vec::with_capacity(n_tasks);
    let mut off = 1;
    for _ in 0..n_tasks {
        if off + 4 > buf.len() { break; }
        let v = i32::from_le_bytes([buf[off], buf[off + 1], buf[off + 2], buf[off + 3]]);
        assignment.push(v);
        off += 4;
    }
    (status, assignment)
}
