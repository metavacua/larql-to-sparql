//! Demo: `memit_solve` + `MemitStore` — the COMPACT MAJOR pipeline in
//! miniature.
//!
//! Runs the vanilla MEMIT closed-form decomposition, packages each
//! per-fact `(key, decomposed_down)` pair into a `MemitFact`, and
//! adds a cycle to a fresh `MemitStore`. Concludes with an
//! entity/relation lookup against the store.
//!
//! Run:  cargo run --release -p larql-vindex --example demo_memit_solve

use larql_vindex::{memit_solve, MemitFact, MemitStore};
use ndarray::Array2;

fn main() {
    println!("=== memit_solve + MemitStore demo ===\n");

    // Five "facts" — entity, relation, target. Each fact is encoded by
    // a (key, target) pair where keys live in the FFN activation space
    // and targets are direction vectors in residual space.
    let facts = [
        ("France", "capital", "Paris"),
        ("Germany", "capital", "Berlin"),
        ("Italy", "capital", "Rome"),
        ("Spain", "capital", "Madrid"),
        ("Portugal", "capital", "Lisbon"),
    ];
    let n = facts.len();
    let d = 32; // toy hidden_dim

    // Synthesise orthogonal-ish keys (one-hot in the toy demo) and
    // distinct target directions.
    let mut keys = Array2::<f32>::zeros((n, d));
    let mut targets = Array2::<f32>::zeros((n, d));
    for i in 0..n {
        keys[[i, i]] = 1.0;
        targets[[i, (i + n) % d]] = 1.0;
    }

    println!("Solving MEMIT: N={n} facts, d={d}, λ=1e-3");
    let solve = memit_solve(&keys, &targets, 1e-3).expect("solve");

    println!("  ‖ΔW‖             = {:.4}", solve.frobenius_norm);
    println!("  max off-diagonal = {:.4}", solve.max_off_diagonal);
    let mean_cos: f32 = solve.reconstruction_cos.iter().sum::<f32>() / n as f32;
    let min_cos = solve
        .reconstruction_cos
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    println!("  reconstruction   = mean {mean_cos:.4}, min {min_cos:.4}");

    // Package decomposed pairs into MemitFact records.
    let memit_facts: Vec<MemitFact> = facts
        .iter()
        .enumerate()
        .map(|(i, (entity, relation, target))| MemitFact {
            entity: (*entity).into(),
            relation: (*relation).into(),
            target: (*target).into(),
            key: keys.row(i).to_owned(),
            decomposed_down: solve.decomposed[i].clone(),
            reconstruction_cos: solve.reconstruction_cos[i],
        })
        .collect();

    // Persist as one COMPACT MAJOR cycle on a fresh store.
    let mut store = MemitStore::new();
    let layer = 33;
    let cycle_id = store.add_cycle(
        layer,
        memit_facts,
        solve.frobenius_norm,
        min_cos,
        solve.max_off_diagonal,
    );
    println!(
        "\nMemitStore: cycle #{cycle_id} added at layer {layer} ({} facts total)",
        store.total_facts()
    );

    // Lookups.
    println!("\nLookups:");
    for (entity, relation, expected) in facts.iter() {
        let hits = store.lookup(entity, relation);
        let ok = hits.iter().any(|f| f.target == *expected);
        let recon = hits.first().map(|f| f.reconstruction_cos).unwrap_or(0.0);
        println!(
            "  {entity:<10} {relation:<10} → {expected:<10}  {} (cos={recon:.3})",
            if ok { "OK" } else { "MISS" }
        );
    }

    // Bonus: enumerate all France facts (would be multi-relation in practice).
    println!("\nfacts_for_entity(\"France\"):");
    for f in store.facts_for_entity("France") {
        println!("  {} {} → {} (cos={:.3})", f.entity, f.relation, f.target, f.reconstruction_cos);
    }

    println!("\nDone.");
}
