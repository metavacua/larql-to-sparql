//! Wasmtime cost baseline.
//!
//! Three measurements that inform whether embedding a WASM solver in a
//! neural-model forward pass is viable:
//!
//! - **compile:** parse + JIT-compile a small .wasm module (one-time cost).
//! - **instantiate:** create a fresh Store + instantiate (per-call cost in
//!   the current design).
//! - **round_trip:** full alloc-write-solve-read on the echo fixture.
//!
//! Run with: `cargo bench -p model-compute --features wasm`

use criterion::{criterion_group, criterion_main, Criterion, Throughput, BenchmarkId};

use model_compute::wasm::SolverRuntime;

const ECHO_WAT: &str = r#"
(module
  (memory (export "memory") 1)
  (global $in_ptr i32 (i32.const 0))
  (global $out_ptr i32 (i32.const 4096))
  (global $in_len (mut i32) (i32.const 0))
  (global $out_len (mut i32) (i32.const 0))
  (func (export "alloc") (param $size i32) (result i32)
    (global.set $in_len (local.get $size))
    (global.get $in_ptr))
  (func (export "solve") (param $ptr i32) (param $len i32) (result i32)
    (memory.copy
      (global.get $out_ptr)
      (local.get $ptr)
      (local.get $len))
    (global.set $out_len (local.get $len))
    (i32.const 0))
  (func (export "solution_ptr") (result i32) (global.get $out_ptr))
  (func (export "solution_len") (result i32) (global.get $out_len)))
"#;

fn bench_compile(c: &mut Criterion) {
    let runtime = SolverRuntime::new().unwrap();
    let wasm = wat::parse_str(ECHO_WAT).unwrap();

    c.bench_function("compile_echo_module", |b| {
        b.iter(|| runtime.compile(&wasm).unwrap())
    });
}

fn bench_instantiate(c: &mut Criterion) {
    let runtime = SolverRuntime::new().unwrap();
    let wasm = wat::parse_str(ECHO_WAT).unwrap();
    let module = runtime.compile(&wasm).unwrap();

    c.bench_function("instantiate_session", |b| {
        b.iter(|| runtime.session(&module).unwrap())
    });
}

fn bench_round_trip(c: &mut Criterion) {
    let runtime = SolverRuntime::new().unwrap();
    let wasm = wat::parse_str(ECHO_WAT).unwrap();
    let module = runtime.compile(&wasm).unwrap();

    let mut group = c.benchmark_group("solve_round_trip");
    for &size in &[16_usize, 256, 4096] {
        let input = vec![0u8; size];
        group.throughput(Throughput::Bytes(size as u64));
        group.bench_with_input(BenchmarkId::from_parameter(size), &input, |b, input| {
            b.iter(|| {
                let mut session = runtime.session(&module).unwrap();
                session.solve(input).unwrap()
            });
        });
    }
    group.finish();
}

criterion_group!(benches, bench_compile, bench_instantiate, bench_round_trip);
criterion_main!(benches);
