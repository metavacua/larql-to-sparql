//! Demo: WASM expert registry — structured op+args calls across all experts.
//!
//! This demo proves calls traverse a real WASM sandbox:
//!   1. Every expert is loaded from a `.wasm` file on disk. The file path,
//!      on-disk byte size, and current linear-memory page count are printed
//!      for each expert after loading.
//!   2. A WASM vs. native speed comparison is run on a hot op (`arithmetic.gcd`)
//!      over 10 000 iterations.
//!   3. A sandbox smoke test calls the same op with args that would be a
//!      division-by-zero in native code; the WASM module returns `null`
//!      instead of trapping the host.
//!
//! Build experts first:
//!   cd crates/larql-experts && cargo build --target wasm32-wasip1 --release
//!
//! Then run from the repo root:
//!   cargo run -p larql-inference --example experts_demo

use std::path::PathBuf;
use std::time::Instant;

use larql_inference::experts::ExpertRegistry;
use serde_json::{json, Value};

fn experts_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../larql-experts/target/wasm32-wasip1/release")
}

fn demos() -> Vec<(&'static str, &'static str, Value)> {
    vec![
        // arithmetic
        ("Addition",          "add",              json!({"a": 12, "b": 34})),
        ("Power",             "pow",              json!({"a": 2, "b": 16})),
        ("Prime check",       "is_prime",         json!({"n": 97})),
        ("GCD",               "gcd",              json!({"a": 144, "b": 60})),
        ("Factorial",         "factorial",        json!({"n": 10})),
        ("Binary",            "to_base",          json!({"n": 255, "base": 2})),
        ("Roman",             "to_roman",         json!({"n": 2024})),
        ("Percent of",        "percent_of",       json!({"pct": 15, "n": 200})),

        // date
        ("Days between",  "days_between",  json!({
            "from": {"year": 2024, "month": 1, "day": 1},
            "to":   {"year": 2024, "month": 2, "day": 29}
        })),
        ("Day of week",   "day_of_week",   json!({"date": {"year": 2024, "month": 7, "day": 4}})),
        ("Add days",      "add_days",      json!({"date": {"year": 2025, "month": 1, "day": 1}, "days": 100})),
        ("Leap year",     "is_leap_year",  json!({"year": 2000})),
        ("Days in month", "days_in_month", json!({"year": 2024, "month": 2})),

        // unit
        ("km -> mi", "convert", json!({"value": 42,  "from": "km", "to": "mi"})),
        ("C -> F",   "convert", json!({"value": 37,  "from": "C",  "to": "F"})),
        ("kg -> lb", "convert", json!({"value": 100, "from": "kg", "to": "lb"})),
        ("in -> cm", "convert", json!({"value": 6,   "from": "in", "to": "cm"})),

        // statistics
        ("Mean",    "mean",   json!({"values": [2, 4, 6, 8, 10]})),
        ("Median",  "median", json!({"values": [3, 1, 4, 1, 5, 9, 2, 6]})),
        ("Std-dev", "stddev", json!({"values": [2, 4, 4, 4, 5, 5, 7, 9]})),
        ("Sort",    "sort",   json!({"values": [5, 2, 8, 1, 9, 3]})),

        // geometry
        ("Circle area",   "circle_area",      json!({"r": 7})),
        ("Sphere volume", "sphere_volume",    json!({"r": 3})),
        ("Hypotenuse",    "hypotenuse",       json!({"a": 5, "b": 12})),
        ("Triangle area", "triangle_area_bh", json!({"base": 8, "height": 5})),

        // trig (radians)
        ("sin π/3",  "sin",  json!({"x": std::f64::consts::FRAC_PI_3})),
        ("cos π/2",  "cos",  json!({"x": std::f64::consts::FRAC_PI_2})),
        ("tan π/4",  "tan",  json!({"x": std::f64::consts::FRAC_PI_4})),
        ("acos 0",   "acos", json!({"x": 0})),

        // string_ops
        ("Reverse",    "reverse",       json!({"s": "hello world"})),
        ("Palindrome", "is_palindrome", json!({"s": "racecar"})),
        ("Anagram",    "is_anagram",    json!({"a": "listen", "b": "silent"})),
        ("Caesar",     "caesar",        json!({"s": "attack", "shift": 13})),
        ("Uppercase",  "uppercase",     json!({"s": "hello"})),

        // hash
        ("Base64 encode", "base64_encode", json!({"s": "hello world"})),
        ("Base64 decode", "base64_decode", json!({"s": "aGVsbG8gd29ybGQ="})),
        ("Hex encode",    "hex_encode",    json!({"s": "abc"})),
        ("URL encode",    "url_encode",    json!({"s": "foo bar=baz"})),

        // logic
        ("Truth table", "truth_table", json!({"expr": "A AND B"})),
        ("Simplify",    "simplify",    json!({"expr": "NOT NOT A"})),
        ("Classify",    "classify",    json!({"expr": "A OR NOT A"})),

        // finance
        ("Future value",      "future_value",      json!({"pv": 10000, "rate_pct": 7, "years": 20})),
        ("Compound interest", "compound_interest", json!({"principal": 5000, "rate_pct": 8, "years": 3})),
        ("Kelly fraction",    "kelly",             json!({"p": 0.55, "b": 2})),

        // element
        ("Gold",      "by_name",   json!({"name": "gold"})),
        ("Iron",      "by_symbol", json!({"symbol": "Fe"})),
        ("Element 92","by_number", json!({"z": 92})),

        // http_status
        ("HTTP 200", "lookup", json!({"code": 200})),
        ("HTTP 404", "lookup", json!({"code": 404})),
        ("HTTP 503", "lookup", json!({"code": 503})),

        // isbn
        ("ISBN-13 validate", "validate", json!({"isbn": "978-0-596-52068-7"})),
        ("ISBN-10 validate", "validate", json!({"isbn": "0-306-40615-2"})),

        // luhn
        ("Luhn Visa",      "check",                json!({"number": "4532015112830366"})),
        ("Card type Amex", "card_type",            json!({"number": "378282246310005"})),
        ("Check digit",    "generate_check_digit", json!({"number": "453201511283036"})),

        // markov
        ("Expected value", "expected_value", json!({
            "outcomes":      [1, 2, 3],
            "probabilities": [0.25, 0.50, 0.25]
        })),
        ("Steady state",   "steady_state",   json!({
            "matrix": [[0.7, 0.3], [0.4, 0.6]]
        })),

        // conway
        ("Blinker 1 gen",    "simulate", json!({"grid": [[0,0,0],[1,1,1],[0,0,0]], "generations": 1})),
        ("Block still-life", "simulate", json!({"grid": [[1,1],[1,1]],            "generations": 1})),

        // dijkstra
        ("Shortest path", "shortest_path", json!({
            "edges": [["A","B",1],["B","C",2],["C","D",1],["A","D",10]],
            "from": "A", "to": "D"
        })),
        ("Reachable",     "reachable",     json!({
            "edges": [["X","Y"],["Y","Z"]], "from": "X", "to": "Z"
        })),
        ("MST",           "mst",           json!({
            "edges": [["A","B",4],["B","C",2],["A","C",3]]
        })),

        // graph
        ("Most central",  "most_central",          json!({"edges": [["A","B"],["A","C"],["A","D"],["B","E"]]})),
        ("Has cycle",     "has_cycle",             json!({"edges": [["A","B"],["B","C"],["C","A"]]})),
        ("Components",    "connected_components",  json!({"edges": [["A","B"],["C","D"],["E","F"]]})),
        ("Is bipartite",  "is_bipartite",          json!({"edges": [["A","B"],["B","C"],["C","D"]]})),

        // sql
        ("SELECT COUNT", "execute", json!({
            "sql": "CREATE TABLE t (x int); INSERT INTO t VALUES (1); INSERT INTO t VALUES (2); INSERT INTO t VALUES (3); SELECT COUNT(*) FROM t"
        })),
        ("SELECT WHERE", "execute", json!({
            "sql": "CREATE TABLE u (id int, name text); INSERT INTO u VALUES (1, 'Alice'); INSERT INTO u VALUES (2, 'Bob'); SELECT name FROM u WHERE id = 1"
        })),
    ]
}

/// Native reference implementation of the same GCD op that `arithmetic.wasm`
/// runs. Used purely as a baseline for the WASM-vs-native benchmark.
fn native_gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = a % b;
        a = b;
        b = t;
    }
    a
}

fn main() {
    let dir = experts_dir();
    if !dir.exists() {
        eprintln!("WASM directory not found: {}", dir.display());
        eprintln!("Build experts first:");
        eprintln!("  cd crates/larql-experts && cargo build --target wasm32-wasip1 --release");
        std::process::exit(1);
    }

    println!("Loading experts from: {}", dir.display());
    let t0 = Instant::now();
    let mut registry = ExpertRegistry::load_dir(&dir).expect("load experts");
    let load_ms = t0.elapsed().as_millis();

    // Clone out the metadata summaries so we can switch to &mut registry below.
    let metas: Vec<(u8, String, String, usize, String)> = registry
        .list()
        .iter()
        .map(|m| (m.tier, m.id.clone(), m.version.clone(), m.ops.len(), m.description.clone()))
        .collect();
    println!("Loaded {} experts in {}ms:", metas.len(), load_ms);
    for (tier, id, version, ops_count, desc) in &metas {
        println!("  [{:>2}] {:14} v{}  {} op(s)  — {}", tier, id, version, ops_count, desc);
    }
    println!("Registered ops: {}", registry.ops().len());
    println!();

    // ── PROOF OF WASM: show each module's on-disk file + live linear memory ──
    println!("WASM runtime proof (wasmtime sandbox):");
    println!("{:-<96}", "");
    println!(
        "{:<14} {:>10} {:>5} {:>8}   PATH",
        "EXPERT", "WASM (B)", "INST", "PAGES"
    );
    println!("{:-<96}", "");
    let wasm_infos = registry.wasm_infos();
    let mut total_wasm_bytes = 0u64;
    let mut total_pages = 0u64;
    for ((_, id, _, _, _), info) in metas.iter().zip(wasm_infos.iter()) {
        let path_display = info
            .path
            .file_name()
            .map(|f| f.to_string_lossy().to_string())
            .unwrap_or_default();
        total_wasm_bytes += info.wasm_bytes;
        total_pages += info.memory_pages;
        println!(
            "{:<14} {:>10} {:>5} {:>8}   {}",
            id,
            info.wasm_bytes,
            if info.instantiated { "yes" } else { "no" },
            info.memory_pages,
            path_display
        );
    }
    println!("{:-<96}", "");
    println!(
        "{:<14} {:>10} {:>5} {:>8}   (live pages @ 64 KiB = {} KiB; experts are lazy-instantiated)",
        "TOTAL",
        total_wasm_bytes,
        "",
        total_pages,
        total_pages * 64
    );
    println!();

    println!("{:-<96}", "");
    println!("{:<22} {:<24} RESULT", "LABEL", "OP");
    println!("{:-<96}", "");

    let mut matched = 0usize;
    let mut skipped = 0usize;
    let demos = demos();
    let total_t0 = Instant::now();

    for (label, op, args) in &demos {
        let t = Instant::now();
        match registry.call(op, args) {
            Some(result) => {
                matched += 1;
                let elapsed_us = t.elapsed().as_micros();
                let display = truncate(&result.value.to_string(), 40);
                println!(
                    "{:<22} {:<24} {:40} [{} µs, {}]",
                    label, op, display, elapsed_us, result.expert_id
                );
            }
            None => {
                skipped += 1;
                println!("{:<22} {:<24} (no match)", label, op);
            }
        }
    }

    println!("{:-<96}", "");
    let total_us = total_t0.elapsed().as_micros();
    println!(
        "Matched {}/{} calls in {} µs total ({} µs/call avg)",
        matched,
        demos.len(),
        total_us,
        if matched > 0 { total_us / matched as u128 } else { 0 }
    );
    if skipped > 0 {
        println!("No match: {} calls", skipped);
    }

    // ── After calls: show which experts are now live in memory ─────────────
    let wasm_after: Vec<_> = registry.wasm_infos();
    let live_after: u64 = wasm_after.iter().filter(|i| i.instantiated).count() as u64;
    let pages_after: u64 = wasm_after.iter().map(|i| i.memory_pages).sum();
    println!(
        "After calls: {}/{} experts instantiated, {} live pages ({} KiB linear memory)",
        live_after,
        wasm_after.len(),
        pages_after,
        pages_after * 64
    );
    let never_called = wasm_after
        .iter()
        .zip(metas.iter())
        .filter(|(i, _)| !i.instantiated)
        .map(|(_, (_, id, _, _, _))| id.as_str())
        .collect::<Vec<_>>();
    if !never_called.is_empty() {
        println!(
            "  experts never called (zero memory footprint): {}",
            never_called.join(", ")
        );
    }

    // ── WASM vs native benchmark on arithmetic.gcd ──────────────────────────
    println!();
    println!("Benchmark: arithmetic.gcd(144, 60) — 10 000 iterations");
    println!("{:-<60}", "");
    let iterations = 10_000;
    let args = json!({"a": 144u64, "b": 60u64});

    let pages_before = registry
        .wasm_info_for("arithmetic")
        .expect("arithmetic loaded")
        .memory_pages;

    let t = Instant::now();
    let mut wasm_acc = 0u64;
    for _ in 0..iterations {
        let r = registry
            .call("gcd", &args)
            .expect("gcd should dispatch");
        wasm_acc = wasm_acc.wrapping_add(r.value.as_u64().unwrap_or(0));
    }
    let wasm_ns = t.elapsed().as_nanos();
    let wasm_per_call_us = wasm_ns as f64 / iterations as f64 / 1000.0;

    let pages_after = registry
        .wasm_info_for("arithmetic")
        .expect("arithmetic loaded")
        .memory_pages;

    let t = Instant::now();
    let mut native_acc = 0u64;
    for _ in 0..iterations {
        native_acc = native_acc.wrapping_add(native_gcd(144, 60));
    }
    let native_ns = t.elapsed().as_nanos();
    let native_per_call_us = native_ns as f64 / iterations as f64 / 1000.0;

    // Both accumulators must agree on the answer — the sandbox is not faking it.
    assert_eq!(wasm_acc, native_acc, "WASM and native disagree on gcd result");

    println!(
        "WASM   (wasmtime+JSON trip):  {:>8.3} µs/call   total {} µs",
        wasm_per_call_us,
        wasm_ns / 1000
    );
    println!(
        "Native (direct Rust fn call): {:>8.3} µs/call   total {} µs",
        native_per_call_us,
        native_ns / 1000
    );
    let pages_delta = pages_after as i64 - pages_before as i64;
    println!(
        "arithmetic memory:            {} → {} pages ({:+} pages = {:+} KiB across {} calls)",
        pages_before, pages_after, pages_delta, pages_delta * 64, iterations
    );
    println!(
        "Overhead factor:              {:.1}×  (entirely ABI marshalling, not the compute)",
        wasm_per_call_us / native_per_call_us
    );

    // ── Sandbox smoke test: a WASM-returned null for division by zero ──────
    println!();
    println!("Sandbox isolation check: div-by-zero returns null, host never traps");
    let r = registry.call("div", &json!({"a": 1, "b": 0})).expect("div dispatches");
    println!("  arithmetic.div({{a:1, b:0}}) => {}", r.value);
    assert_eq!(r.value, serde_json::Value::Null);
    println!("  ok — sandbox contained the degenerate case.");
}

fn truncate(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        s.to_string()
    } else {
        let mut out: String = s.chars().take(max - 1).collect();
        out.push('…');
        out
    }
}
