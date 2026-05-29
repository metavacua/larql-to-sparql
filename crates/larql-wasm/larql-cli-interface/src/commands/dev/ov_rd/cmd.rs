use clap::{Args, Subcommand};

use super::capture::{run_capture, CaptureArgs};
use super::edit_catalog::{run_oracle_edit_catalog, OracleEditCatalogArgs};
use super::eval_program::{run_eval_program, EvalProgramArgs};
use super::induce_program::{run_induce_program, InduceProgramArgs};
use super::normalize_program::{run_normalize_program, NormalizeProgramArgs};
use super::oracle::{
    run_oracle_lowrank, run_oracle_roundtrip, OracleLowrankArgs, OracleRoundtripArgs,
};
use super::oracle_pq::{run_oracle_pq, OraclePqArgs};
use super::pq_exception::{run_oracle_pq_exception, OraclePqExceptionArgs};
use super::probe_program_class::{run_probe_program_class, ProbeProgramClassArgs};
use super::program_cache::{run_build_program_cache, BuildProgramCacheArgs};
use super::sanity::{run_sanity_check, SanityCheckArgs};
use super::static_replace::{run_static_replace, StaticReplaceArgs};
use super::synthesize_program::{run_synthesize_program, SynthesizeProgramArgs};
use super::zero_ablate::{run_zero_ablate, ZeroAblateArgs};

#[derive(Args)]
pub struct OvRdArgs {
    #[command(subcommand)]
    command: OvRdCommand,
}

#[derive(Subcommand)]
enum OvRdCommand {
    /// Capture pre-W_O OV output statistics from a Q4K vindex.
    Capture(CaptureArgs),

    /// Gate 1: zero selected pre-W_O heads and measure final-logit KL.
    ZeroAblate(ZeroAblateArgs),

    /// Static replacement gate: zero/global/position/stratum pre-W_O means.
    StaticReplace(StaticReplaceArgs),

    /// Sanity checks for pre-W_O replacement and W_O block equivalence.
    SanityCheck(SanityCheckArgs),

    /// Oracle RD plumbing check: W_O-coordinate roundtrip with no truncation.
    OracleRoundtrip(OracleRoundtripArgs),

    /// Oracle RD: unquantized low-rank sweep in W_O-visible coordinates.
    OracleLowrank(OracleLowrankArgs),

    /// Oracle RD: oracle-addressed product quantization in PCA coordinates.
    OraclePq(OraclePqArgs),

    /// Oracle RD: full residual-edit catalogues in hidden/PCA spaces.
    OracleEditCatalog(OracleEditCatalogArgs),

    /// Oracle RD: base PQ table plus oracle-addressed exception residuals.
    OraclePqException(OraclePqExceptionArgs),

    /// AHORD: evaluate a behavioral rewrite program against oracle PQ codes.
    EvalProgram(EvalProgramArgs),

    /// AHORD: induce a behavioral rewrite program via pairwise merge proposals.
    InduceProgram(InduceProgramArgs),

    /// AHORD: normalize a program JSON (compute effective_map per stage).
    NormalizeProgram(NormalizeProgramArgs),

    /// AHORD: probe whether behavioral program classes are visible in residual/pre-W_O features.
    ProbeProgramClass(ProbeProgramClassArgs),

    /// AHORD: build program cache from existing variant measurements (zero forwards).
    BuildProgramCache(BuildProgramCacheArgs),

    /// AHORD: synthesize a program from the cache (zero forwards).
    SynthesizeProgram(SynthesizeProgramArgs),
}

pub fn run(args: OvRdArgs) -> Result<(), Box<dyn std::error::Error>> {
    match args.command {
        OvRdCommand::Capture(capture) => run_capture(capture),
        OvRdCommand::ZeroAblate(zero) => run_zero_ablate(zero),
        OvRdCommand::StaticReplace(static_replace) => run_static_replace(static_replace),
        OvRdCommand::SanityCheck(sanity) => run_sanity_check(sanity),
        OvRdCommand::OracleRoundtrip(roundtrip) => run_oracle_roundtrip(roundtrip),
        OvRdCommand::OracleLowrank(lowrank) => run_oracle_lowrank(lowrank),
        OvRdCommand::OraclePq(pq) => run_oracle_pq(pq),
        OvRdCommand::OracleEditCatalog(edit_catalog) => run_oracle_edit_catalog(edit_catalog),
        OvRdCommand::OraclePqException(exception) => run_oracle_pq_exception(exception),
        OvRdCommand::EvalProgram(eval_program) => run_eval_program(eval_program),
        OvRdCommand::InduceProgram(induce) => run_induce_program(induce),
        OvRdCommand::NormalizeProgram(norm) => run_normalize_program(norm),
        OvRdCommand::ProbeProgramClass(probe) => run_probe_program_class(probe),
        OvRdCommand::BuildProgramCache(cache) => run_build_program_cache(cache),
        OvRdCommand::SynthesizeProgram(synth) => run_synthesize_program(synth),
    }
}
