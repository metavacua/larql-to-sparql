use super::super::program::{
    BaseConfig, ConstructionMode, Program, ProgramRule, ProgramStage, TerminalClass,
};
use super::super::types::HeadId;

/// Build the oracle identity program (all codes separate, no rules).
///
/// Used as the baseline against which proposals are measured.
pub fn identity_program(head: HeadId, group: usize, config: &BaseConfig) -> Program {
    let num_codes = config.num_codes();
    let terminal_classes = (0..num_codes)
        .map(|c| TerminalClass {
            class: c,
            construction_mode: ConstructionMode::Representative,
            representative_code: c,
        })
        .collect();
    Program {
        head,
        group,
        base_config: config.clone(),
        name: Some("identity".to_string()),
        stages: vec![],
        terminal_classes,
        metrics: None,
        reference_metrics: None,
        tolerance: None,
        program_size: None,
        predicate_space_used: vec![],
        codebook_fingerprint: None,
    }
}

/// A program with one rule: `source -> target` in a fixed-point canonicalize stage.
pub fn single_merge_program(
    head: HeadId,
    group: usize,
    config: &BaseConfig,
    source: usize,
    target: usize,
) -> Program {
    Program {
        head,
        group,
        base_config: config.clone(),
        name: Some(format!("{source}->{target}")),
        stages: vec![ProgramStage {
            name: "canonicalize".to_string(),
            fixed_point: true,
            declared_rules: vec![ProgramRule::Map { source, target }],
            effective_map: None,
            guards: vec![],
        }],
        terminal_classes: vec![TerminalClass {
            class: target,
            construction_mode: ConstructionMode::Representative,
            representative_code: target,
        }],
        metrics: None,
        reference_metrics: None,
        tolerance: None,
        program_size: None,
        predicate_space_used: vec![],
        codebook_fingerprint: None,
    }
}

/// A program that merges all codes in `sources` to `target` simultaneously
/// (fixed_point: false to avoid chaining).
pub fn set_merge_program(
    head: HeadId,
    group: usize,
    config: &BaseConfig,
    sources: Vec<usize>,
    target: usize,
) -> Program {
    let label = format!(
        "{{{}}}->{target}",
        sources
            .iter()
            .map(|c| c.to_string())
            .collect::<Vec<_>>()
            .join(",")
    );
    Program {
        head,
        group,
        base_config: config.clone(),
        name: Some(label),
        stages: vec![ProgramStage {
            name: "canonicalize".to_string(),
            fixed_point: false,
            declared_rules: vec![ProgramRule::MapSet {
                source: sources,
                target,
            }],
            effective_map: None,
            guards: vec![],
        }],
        terminal_classes: vec![TerminalClass {
            class: target,
            construction_mode: ConstructionMode::Representative,
            representative_code: target,
        }],
        metrics: None,
        reference_metrics: None,
        tolerance: None,
        program_size: None,
        predicate_space_used: vec![],
        codebook_fingerprint: None,
    }
}
