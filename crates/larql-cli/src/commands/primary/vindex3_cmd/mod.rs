//! `larql vindex3` — VINDEX3 container programme verbs.
//!
//! `plan` is the G1 gate: a semantic representability check over one or
//! more artifacts, run *before* any conversion. It prints the full
//! [`larql_vindex::format::vindex3::plan::SystemPlan`] as JSON and exits
//! non-zero when the plan is inadmissible, so scripts can gate on it.

use std::path::{Path, PathBuf};

use clap::{Args, Subcommand};

use larql_models::inventory::{build_inventory, ArchitectureInventory};
use larql_vindex::format::vindex3::plan::plan_system;

/// Extension distinguishing a saved inventory JSON from a checkpoint dir.
const INVENTORY_EXT: &str = "json";

#[derive(Subcommand)]
pub enum Vindex3Command {
    /// Semantic representability plan over HF checkpoint dirs and/or saved
    /// `inspect-hf` inventory JSONs, treated together as one model system.
    /// Exits non-zero when the plan is inadmissible.
    Plan(PlanArgs),
    /// Encode the system into a self-contained container (G3). Refuses an
    /// inadmissible plan; consumes the built graph, never re-interprets
    /// the checkpoint.
    Encode(EncodeArgs),
    /// Reconstruct and check a container solely from its own contents —
    /// no source checkpoint, no architecture registry (the G3 gate).
    Inspect(InspectArgs),
    /// Prove source ≡ encoded (the G4 gate): four-authority semantic
    /// comparison plus per-representation byte equivalence, both ends
    /// re-hashed now. Exits non-zero on any disagreement.
    Verify(VerifyArgs),
    /// Emit the generic operation plan of one component, solely from the
    /// container (G5b-1). Operand closure is the gate: every stack tensor
    /// must classify into a role a declared op consumes, with the
    /// geometry the surface states. Exits non-zero on any closure defect.
    Ops(OpsArgs),
}

#[derive(Args)]
pub struct OpsArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Component to plan.
    #[arg(long, default_value = "target")]
    pub component: String,

    /// Print one layer's full program instead of the per-layer summary.
    #[arg(long)]
    pub layer: Option<usize>,

    /// Print the full plan as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct VerifyArgs {
    /// Checkpoint directories or inventory JSON files — the same artifact
    /// set the container was encoded from.
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Container directory to verify against.
    #[arg(long)]
    pub container: PathBuf,

    /// Print the full report as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct EncodeArgs {
    /// Checkpoint directories or inventory JSON files (one per artifact).
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Container directory to write.
    #[arg(long)]
    pub output: PathBuf,
}

#[derive(Args)]
pub struct InspectArgs {
    /// Container directory.
    pub container: PathBuf,

    /// Additionally re-hash every segment against the directory.
    #[arg(long)]
    pub verify: bool,

    /// Additionally require execution completeness (the G5a gate): every
    /// component with executable objects must carry the surface those
    /// operations read. Exits non-zero when incomplete.
    #[arg(long)]
    pub execution_complete: bool,

    /// Print the full reconstruction as JSON instead of the summary.
    #[arg(long)]
    pub json: bool,
}

#[derive(Args)]
pub struct PlanArgs {
    /// Checkpoint directories or inventory JSON files (one per artifact).
    #[arg(required = true)]
    pub artifacts: Vec<PathBuf>,

    /// Write the plan JSON here instead of stdout.
    #[arg(long)]
    pub output: Option<PathBuf>,
}

pub fn run(cmd: Vindex3Command) -> Result<(), Box<dyn std::error::Error>> {
    match cmd {
        Vindex3Command::Plan(args) => run_plan(args),
        Vindex3Command::Encode(args) => run_encode(args),
        Vindex3Command::Inspect(args) => run_inspect(args),
        Vindex3Command::Verify(args) => run_verify(args),
        Vindex3Command::Ops(args) => run_ops(args),
    }
}

mod ops;
use ops::run_ops;

fn run_verify(args: VerifyArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let verification =
        larql_vindex::format::vindex3::verify_system::verify_system(&named, &args.container)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&verification)?);
    } else {
        for defect in &verification.container_defects {
            println!("container defect: {defect}");
        }
        let semantic_pass = verification.semantic.iter().filter(|c| c.pass).count();
        println!(
            "semantic: {semantic_pass}/{} authority checks pass",
            verification.semantic.len()
        );
        for failure in verification.semantic_failures() {
            println!(
                "  FAIL {} ≡ {}  {}: {}",
                failure.left, failure.right, failure.subject, failure.detail
            );
        }
        println!("payloads:");
        for check in &verification.payloads {
            println!(
                "  {} {:40} {:8.3} GB  {}",
                if check.pass { "PASS" } else { "FAIL" },
                check.representation,
                check.payload_bytes as f64 / 1e9,
                if check.pass {
                    format!(
                        "sha256 {}…",
                        &check.recorded_sha256[..12.min(check.recorded_sha256.len())]
                    )
                } else {
                    check.detail.clone()
                },
            );
        }
    }
    if verification.verified {
        eprintln!("verified: Declared ≡ Resolved ≡ Graph ≡ Encoded; payloads byte-equal");
        Ok(())
    } else {
        Err(format!(
            "NOT verified: {} semantic failure(s), {} payload failure(s), {} container defect(s)",
            verification.semantic_failures().count(),
            verification.payload_failures().count(),
            verification.container_defects.len(),
        )
        .into())
    }
}

fn run_encode(args: EncodeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let outcome = larql_vindex::format::vindex3::encode::encode_system(&named, &args.output)?;
    eprintln!(
        "encoded {} representation(s), {:.2} GB payload → {}",
        outcome.representations,
        outcome.total_payload_bytes as f64 / 1e9,
        outcome.container.display(),
    );
    Ok(())
}

fn run_inspect(args: InspectArgs) -> Result<(), Box<dyn std::error::Error>> {
    let inspection =
        larql_vindex::format::vindex3::inspect::inspect_container(&args.container, args.verify)?;
    if args.json {
        println!("{}", serde_json::to_string_pretty(&inspection)?);
    } else {
        println!("components:");
        for c in &inspection.components {
            let policy = match (c.sliding_layers, c.full_layers, c.nope_layers) {
                (Some(s), Some(f), Some(n)) => {
                    format!(", {s} sliding / {f} full, {n} NoPE, window {:?}", c.window)
                }
                _ => ", (no per-layer table)".to_string(),
            };
            println!(
                "  {:8} {:12} {} layers, hidden {}{policy}",
                c.id, c.role, c.num_layers, c.hidden_size
            );
        }
        println!("objects:");
        for (id, entry) in &inspection.index.representations {
            println!(
                "  {:40} {:8.3} GB  {} tensors  sha256 {}…",
                id,
                entry.payload_bytes as f64 / 1e9,
                entry.tensor_count,
                &entry.payload_sha256[..12],
            );
        }
        println!("edges:");
        for e in &inspection.graph.edges {
            println!(
                "  {}.hidden{:?} -> {} via {} (block {:?})",
                e.producer_component,
                e.producer_layers,
                e.consumer_component,
                e.consumer_object,
                e.block_size,
            );
        }
    }
    let surface_defects = if args.execution_complete {
        let defects = inspection.execution_completeness();
        if !args.json {
            println!("execution surfaces:");
            for component in &inspection.graph.components {
                match &component.execution {
                    Some(surface) => println!(
                        "  {:8} attention {}q/{}kv head {} q-scale {:.4} s-scale {:.4}{}, \
                         ffn {:?} {:?} {}, norm {:?} eps {:e}{}",
                        component.id,
                        surface.attention.num_q_heads,
                        surface.attention.num_kv_heads,
                        surface.attention.head_dim,
                        surface.attention.query_scale,
                        surface.attention.score_scale,
                        if surface.attention.output_gate.is_some() {
                            " gated"
                        } else {
                            ""
                        },
                        surface.ffn.activation,
                        surface.ffn.ffn_type,
                        surface.ffn.intermediate_size,
                        surface.norm.kind,
                        surface.norm.eps,
                        match &surface.head {
                            Some(head) => format!(", head vocab {}", head.vocab_size),
                            None => String::new(),
                        },
                    ),
                    None => println!("  {:8} (no execution surface)", component.id),
                }
            }
            for defect in &defects {
                println!("  defect: {defect}");
            }
            println!(
                "executable: {}",
                if defects.is_empty() { "yes" } else { "NO" }
            );
        }
        defects
    } else {
        Vec::new()
    };

    if !surface_defects.is_empty() {
        return Err(format!(
            "container not executable: {} execution-completeness defect(s)",
            surface_defects.len()
        )
        .into());
    }
    if inspection.is_coherent() {
        eprintln!(
            "container coherent{}",
            if args.verify {
                " (payloads verified)"
            } else {
                ""
            }
        );
        Ok(())
    } else {
        for defect in &inspection.defects {
            eprintln!("defect: {defect:?}");
        }
        Err(format!(
            "container incoherent: {} defect(s)",
            inspection.defects.len()
        )
        .into())
    }
}

/// Artifact display name: the file/directory stem.
fn artifact_name(path: &Path) -> String {
    path.file_stem()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_else(|| path.display().to_string())
}

/// Load one artifact: a `.json` file deserialises as a saved inventory;
/// anything else is inspected as a checkpoint directory.
fn load_artifact(path: &Path) -> Result<ArchitectureInventory, Box<dyn std::error::Error>> {
    if path.extension().is_some_and(|ext| ext == INVENTORY_EXT) {
        let text = std::fs::read_to_string(path)?;
        Ok(serde_json::from_str(&text)?)
    } else {
        Ok(build_inventory(path)?)
    }
}

fn run_plan(args: PlanArgs) -> Result<(), Box<dyn std::error::Error>> {
    let mut named = Vec::new();
    for path in &args.artifacts {
        named.push((artifact_name(path), load_artifact(path)?));
    }
    let plan = plan_system(&named);
    let json = serde_json::to_string_pretty(&plan)?;
    match &args.output {
        Some(path) => {
            std::fs::write(path, &json)?;
            eprintln!("plan written to {}", path.display());
        }
        None => println!("{json}"),
    }
    let summary = &plan.summary;
    eprintln!(
        "plan: {} representable, {} mismatched, {} unrepresented, {} interfaces — {} blocking",
        summary.representable,
        summary.mismatched,
        summary.unrepresented,
        summary.interfaces,
        summary.blocking,
    );
    if plan.admissible {
        Ok(())
    } else {
        Err(format!(
            "plan not admissible: {} blocking finding(s); \
             every one is a schema or resolution gap to close before conversion",
            summary.blocking
        )
        .into())
    }
}

#[cfg(test)]
mod tests;
