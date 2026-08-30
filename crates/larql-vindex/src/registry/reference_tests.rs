//! Colocated tests for the model-reference grammar — rung requirement
//! "define and test the model-name/variant grammar" (design doc §7.2).

use super::reference::{ExplicitReference, ModelName, ModelReference, VariantName};

// ── The four public forms, accepted ─────────────────────────────────────

#[test]
fn bare_name_is_a_registry_reference_with_no_variant() {
    let r = ModelReference::parse("qwen3.8").unwrap();
    assert_eq!(
        r,
        ModelReference::Registry {
            name: ModelName::parse("qwen3.8").unwrap(),
            variant: None,
        }
    );
}

#[test]
fn name_colon_variant_is_a_registry_reference_with_a_variant() {
    let r = ModelReference::parse("qwen3.8:27b-nvfp4").unwrap();
    assert_eq!(
        r,
        ModelReference::Registry {
            name: ModelName::parse("qwen3.8").unwrap(),
            variant: Some(VariantName::parse("27b-nvfp4").unwrap()),
        }
    );
}

#[test]
fn hf_prefix_is_explicit_huggingface_with_no_revision() {
    let r = ModelReference::parse("hf://owner/repo").unwrap();
    assert_eq!(
        r,
        ModelReference::Explicit(ExplicitReference::HuggingFace {
            repo: "owner/repo".to_string(),
            revision: None,
        })
    );
}

#[test]
fn hf_prefix_with_at_pins_a_revision() {
    let r = ModelReference::parse("hf://owner/repo@deadbeef").unwrap();
    assert_eq!(
        r,
        ModelReference::Explicit(ExplicitReference::HuggingFace {
            repo: "owner/repo".to_string(),
            revision: Some("deadbeef".to_string()),
        })
    );
}

#[test]
fn a_slash_containing_string_that_is_not_hf_is_an_explicit_local_path() {
    let r = ModelReference::parse("/explicit/local/path").unwrap();
    assert_eq!(
        r,
        ModelReference::Explicit(ExplicitReference::Local("/explicit/local/path".into()))
    );
}

#[test]
fn a_relative_slash_path_is_also_explicit_local() {
    let r = ModelReference::parse("some/dir").unwrap();
    assert_eq!(
        r,
        ModelReference::Explicit(ExplicitReference::Local("some/dir".into()))
    );
}

// A native Windows absolute path (`C:\Users\...`) contains no `/` at
// all, so the slash check alone can't catch it — it used to fall
// through to `parse_registry`, split on the drive letter's `:`, and
// refuse with a bogus "model name 'C' must be lowercase..." error
// (caught by Windows CI on a real tempdir path). `#[cfg(windows)]`
// because `PathBuf::is_absolute` only understands drive-letter syntax
// when compiled for Windows — on other hosts this string is just an
// oddly-shaped relative path, already exercised by the malformed-name
// cases below.
#[test]
#[cfg(windows)]
fn a_windows_drive_absolute_path_is_also_explicit_local() {
    let r = ModelReference::parse(r"C:\Users\runner\AppData\Local\Temp\some-vindex").unwrap();
    assert_eq!(
        r,
        ModelReference::Explicit(ExplicitReference::Local(
            r"C:\Users\runner\AppData\Local\Temp\some-vindex".into()
        ))
    );
}

// The Windows-absolute check must not swallow a short, otherwise-valid
// registry name that happens to contain a `:` — `x:foo` has no root
// after the colon, so Windows path semantics call it "drive-relative"
// rather than absolute, and it keeps resolving as `name:variant`.
#[test]
#[cfg(windows)]
fn a_drive_relative_string_without_a_root_is_still_a_registry_reference() {
    let r = ModelReference::parse("x:foo").unwrap();
    assert!(matches!(r, ModelReference::Registry { .. }), "{r:?}");
}

// ── Malformed references refuse by name ─────────────────────────────────

#[test]
fn empty_reference_refuses() {
    assert!(ModelReference::parse("").is_err());
}

#[test]
fn leading_whitespace_refuses() {
    assert!(ModelReference::parse(" qwen3.8").is_err());
}

#[test]
fn trailing_whitespace_refuses() {
    assert!(ModelReference::parse("qwen3.8 ").is_err());
}

#[test]
fn empty_variant_after_colon_refuses() {
    let err = ModelReference::parse("qwen3.8:").unwrap_err();
    assert!(err.to_string().contains("variant name"), "{err}");
}

#[test]
fn empty_name_before_colon_refuses() {
    let err = ModelReference::parse(":27b-nvfp4").unwrap_err();
    assert!(err.to_string().contains("model name"), "{err}");
}

#[test]
fn more_than_one_colon_refuses() {
    let err = ModelReference::parse("qwen3.8:27b-nvfp4:extra").unwrap_err();
    assert!(err.to_string().contains("more than one"), "{err}");
}

#[test]
fn uppercase_name_refuses() {
    assert!(ModelReference::parse("Qwen3.8").is_err());
}

#[test]
fn name_with_whitespace_refuses() {
    assert!(ModelReference::parse("qwen 3.8").is_err());
}

#[test]
fn name_starting_with_a_separator_refuses() {
    assert!(ModelName::parse(".qwen").is_err());
    assert!(ModelName::parse("-qwen").is_err());
}

#[test]
fn name_ending_with_a_separator_refuses() {
    assert!(ModelName::parse("qwen.").is_err());
    assert!(ModelName::parse("qwen-").is_err());
}

#[test]
fn hf_reference_naming_no_repo_refuses() {
    assert!(ModelReference::parse("hf://").is_err());
}

#[test]
fn hf_reference_with_empty_revision_pin_refuses() {
    assert!(ModelReference::parse("hf://owner/repo@").is_err());
}

#[test]
fn hf_reference_with_nested_path_refuses() {
    // Exactly `owner/name` — no nested paths, no bare owner.
    assert!(ModelReference::parse("hf://owner/repo/extra").is_err());
    assert!(ModelReference::parse("hf://owner").is_err());
}

#[test]
fn empty_model_name_component_refuses() {
    assert!(ModelName::parse("").is_err());
}

#[test]
fn empty_variant_name_component_refuses() {
    assert!(VariantName::parse("").is_err());
}

#[test]
fn model_name_display_round_trips_the_string() {
    assert_eq!(ModelName::parse("qwen3.8").unwrap().to_string(), "qwen3.8");
}

#[test]
fn variant_name_display_round_trips_the_string() {
    assert_eq!(
        VariantName::parse("27b-nvfp4").unwrap().to_string(),
        "27b-nvfp4"
    );
}

#[test]
fn model_name_as_str_matches_input() {
    assert_eq!(ModelName::parse("qwen3.8").unwrap().as_str(), "qwen3.8");
}

#[test]
fn variant_name_as_str_matches_input() {
    assert_eq!(
        VariantName::parse("27b-nvfp4").unwrap().as_str(),
        "27b-nvfp4"
    );
}
