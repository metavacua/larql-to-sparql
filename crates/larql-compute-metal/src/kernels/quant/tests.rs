//! Expert-route tests.
//!
//! These cover what decides correctness before a device exists: which
//! arm is exact, how each arm binds, and that arm selection cannot land
//! somewhere the caller did not ask for. `grouped_experts_for` and
//! `expert_matvec_for` need a live `QuantKernels` (and therefore a
//! device), so they are exercised by the Metal integration tests; what
//! is unit-testable here is the arm vocabulary those routes read.

use crate::shaders::mxfp4_grouped_experts::Mxfp4Arm;

/// Arm A2 is the default, and it is an exact one. Both halves matter:
/// the default must be servable under a losslessness claim without the
/// caller opting in to anything. (A2 additionally carries an alignment
/// precondition, but the encode path demotes it to arm A — also exact —
/// rather than weakening the claim.)
#[test]
fn default_arm_is_exact() {
    assert_eq!(Mxfp4Arm::default(), Mxfp4Arm::SplitLut16Vec);
    assert!(Mxfp4Arm::default().is_exact());
}

/// Every interleaved arm is inexact — the 1-bit exponent delta means
/// 2.88% of real superblocks can only be encoded by clamping.
#[test]
fn interleaved_arms_are_not_exact() {
    for arm in [
        Mxfp4Arm::InterLut16,
        Mxfp4Arm::InterPair,
        Mxfp4Arm::InterMagSign,
    ] {
        assert!(!arm.is_exact(), "{arm:?} must not claim exactness");
    }
}

/// Exactly the exact arm uses the split binding. The two are separate
/// facts about a layout — neither is derived from the other — so this
/// pins that they agree today rather than assuming they must.
#[test]
fn exactness_and_split_binding_coincide() {
    for arm in [
        Mxfp4Arm::SplitLut16,
        Mxfp4Arm::InterLut16,
        Mxfp4Arm::InterPair,
        Mxfp4Arm::InterMagSign,
    ] {
        assert_eq!(arm.is_exact(), arm.is_split_scale(), "{arm:?}");
    }
}

/// Both spellings resolve, and the tournament letters match the arm
/// names the shader module's own A–D table documents.
#[test]
fn arm_names_and_tournament_letters_both_parse() {
    let table = [
        ("split_lut16", "a", Mxfp4Arm::SplitLut16),
        ("inter_lut16", "b", Mxfp4Arm::InterLut16),
        ("inter_pair", "c", Mxfp4Arm::InterPair),
        ("inter_magsign", "d", Mxfp4Arm::InterMagSign),
    ];
    for (name, letter, expected) in table {
        assert_eq!(Mxfp4Arm::from_name(name), Some(expected), "{name}");
        assert_eq!(Mxfp4Arm::from_name(letter), Some(expected), "{letter}");
    }
}

#[test]
fn arm_names_are_case_insensitive() {
    assert_eq!(
        Mxfp4Arm::from_name("INTER_MAGSIGN"),
        Some(Mxfp4Arm::InterMagSign)
    );
    assert_eq!(Mxfp4Arm::from_name("D"), Some(Mxfp4Arm::InterMagSign));
}

/// An unknown name yields `None`, and the option layer then falls back
/// to the exact default — a typo must never silently select a lossy arm.
/// The ceiling probes are deliberately unselectable: they do not compute
/// a correct product.
#[test]
fn unknown_and_probe_arm_names_do_not_resolve() {
    for name in ["", "arm_e", "inter", "split", "inter_bits", "inter_nox"] {
        assert_eq!(Mxfp4Arm::from_name(name), None, "{name:?}");
    }
}

/// The default `BackendOptions` selects the exact arm without consulting
/// the environment — production code that builds options programmatically
/// must not inherit a lossy arm from a stray shell variable.
#[test]
fn default_backend_options_select_the_exact_arm() {
    let opts = crate::options::BackendOptions::default();
    assert!(opts.mxfp4_arm.is_exact());
}
