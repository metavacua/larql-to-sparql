//! End-to-end demo of every shipped engine on synthetic weights.
//!
//! Loads the `larql_inference::test_utils` 2-layer fixture, then runs each
//! engine through prefill + a few decode steps, printing the per-engine
//! diagnostics so you can see the trait surface in action.
//!
//! Covers all 9 `EngineKind` variants (8 K/V engines via `KvEngine`
//! plus Apollo via `RetrievalEngine`). Apollo prefills with no store
//! attached, so the demo intentionally surfaces its
//! `EngineError::RetrievalMiss` — that's the path the typed-error
//! migration was designed to make visible (the harness previously
//! collapsed it into a silent `None`).
//!
//! Run with:
//!
//! ```sh
//! cargo run -p larql-demos --example engine_ladder
//! ```

use larql_inference::cpu_engine_backend;
use larql_inference::ffn::WeightFfn;
use larql_inference::test_utils::make_test_weights;
use larql_kv::{AnyEngine, EngineKind};

fn run_engine(label: &str, mut engine: AnyEngine) {
    let weights = make_test_weights();
    let ffn = WeightFfn { weights: &weights };
    let prompt: Vec<u32> = (0..8).collect();

    print!("{label:<32} ");
    let prefill = engine.prefill(&weights, &ffn, &prompt);
    if let Err(err) = prefill {
        println!("prefill failed (engine not configured): {err}");
        return;
    }

    for tok in 0..3 {
        let _ = engine.decode_step(&weights, &ffn, tok as u32);
    }

    let info = engine.info();
    println!(
        "memory={:>8} bytes  window={:<5}  cold={:>8} bytes  [{}]",
        engine.memory_bytes(),
        engine.window_tokens(),
        engine.cold_bytes(),
        info.summary(),
    );
}

fn main() {
    let specs = [
        "standard",
        "standard:window=8",
        "no-cache",
        "markov-rs",
        "markov-rs:window=4",
        "unlimited-context:window=4",
        "turbo-quant:bits=4",
        "tq3",
        "apollo:layer=1,coef=8.0,top_k=4",
        "boundary-kv:chunk_tokens=4,sequence_id=demo",
        "boundary-kv:window=4,chunk_tokens=4,sequence_id=demo-bounded",
        "markov-rs-codec",
        "markov-rs-codec:window=4",
        "boundary-per-layer:layers=2",
        "boundary-per-layer:window=4,layers=2",
    ];

    println!("larql-kv engine ladder (synthetic 2-layer model)\n");
    println!("{:<32} diagnostics", "engine");
    println!("{}", "-".repeat(96));

    for spec in specs {
        let kind = match EngineKind::from_name(spec) {
            Some(k) => k,
            None => {
                println!("{spec:<32} <unparseable>");
                continue;
            }
        };
        run_engine(spec, kind.build(cpu_engine_backend()));
    }
}
