//! VINDEX3 reference MoE demo — fixture A, checkpoint by checkpoint.
//!
//! The counterpart to `q4k_demo` / `walker_demo` for the new execution path.
//! It prints every instrumented boundary inside one routed-MoE layer, run
//! twice: once from decomposed `gate`/`up` regions and once from a single
//! concatenated `gate_up_fused` region.
//!
//! Both runs must agree exactly. That is the property worth being able to see
//! by eye: the executor is following a bound recipe, not inferring a layout.
//!
//! Usage: `cargo run --release -p larql-vindex --example vindex3_reference_moe`

use larql_vindex::runtime::execute_traced;
use larql_vindex::runtime::fixtures::direct_moe::{input, DirectMoeFixture, POPULATION, TOP_K};
use larql_vindex::runtime::fixtures::direct_moe_oracle::oracle;
use larql_vindex::runtime::MoeInputs;
use larql_vindex::runtime::{CollectedTrace, ProjectionArrangement};

/// Column width for the numeric dumps.
const PRECISION: usize = 6;

fn render(values: &[f32]) -> String {
    values
        .iter()
        .map(|v| format!("{v:+.PRECISION$}"))
        .collect::<Vec<_>>()
        .join("  ")
}

fn report(arrangement: ProjectionArrangement, trace: &CollectedTrace, delta: &[f32]) {
    println!("\n── {} storage {}", arrangement.name(), "─".repeat(46));
    println!("  router scores   ({POPULATION} experts)");
    println!("      {}", render(&trace.router_scores));
    println!("  selection       (top-{TOP_K}, in order)");
    for selected in &trace.selection {
        println!(
            "      expert {:>3}   weight {:+.PRECISION$}   raw {:+.PRECISION$}",
            selected.expert_id, selected.weight, selected.raw_score
        );
    }
    println!("  expert outputs  (unweighted)");
    for out in &trace.expert_outputs {
        println!(
            "      expert {:>3}   {}",
            out.expert_id,
            render(&out.values)
        );
    }
    println!("  reduction");
    println!("      {}", render(&trace.reduced));
    println!("  residual delta");
    println!("      {}", render(delta));
}

fn main() {
    let fixture = DirectMoeFixture::new();
    let residual = input();

    println!("VINDEX3 reference MoE — fixture A");
    println!("  residual in");
    println!("      {}", render(&residual));

    let mut runs = Vec::new();
    for arrangement in ProjectionArrangement::ALL {
        let operation = fixture.operation(arrangement);
        operation
            .validate()
            .expect("fixture A binds a consistent operation");
        println!("\n  bound: {}", operation.describe());

        let (delta, trace) = execute_traced(&operation, MoeInputs::shared(&residual))
            .expect("fixture A executes cleanly");
        report(arrangement, &trace, &delta);
        runs.push((arrangement, delta, trace));
    }

    // ── The two properties worth asserting out loud ────────────────────────
    println!("\n── verdict {}", "─".repeat(52));

    let [(_, first_delta, first_trace), (_, second_delta, second_trace)] = &runs[..] else {
        unreachable!("exactly one run per arrangement");
    };
    let arrangements_agree = first_delta == second_delta && first_trace == second_trace;
    println!(
        "  {} fused and decomposed agree at every checkpoint",
        if arrangements_agree { "PASS" } else { "FAIL" }
    );

    let expected = oracle(&residual);
    let epsilon = 1e-6;
    let matches_oracle = first_delta
        .iter()
        .zip(&expected.reduced)
        .all(|(a, e)| (a - e).abs() < epsilon);
    println!(
        "  {} output matches the independent oracle (< {epsilon:e})",
        if matches_oracle { "PASS" } else { "FAIL" }
    );

    if !(arrangements_agree && matches_oracle) {
        std::process::exit(1);
    }
}
