//! The capture stamps which calibration set it consumed, and the digest it
//! stamps must be the one the Python side froze.
//!
//! SENSITIVITY-1B' is judged on a calibration set disjoint from Q-BANK-1.
//! The artefact that would invalidate it already exists — the 1B-a capture
//! carries a `num` field numerically equal to 1B''s numerator, computed
//! from bank-derived activations — so the scorer refuses moments by
//! provenance rather than by filename.
//!
//! That refusal is only as good as the digest agreeing across the two
//! languages that compute it. `freeze_calibration.py` hashes
//! `json.dumps([{id, ids}], sort_keys=True, separators=(",", ":"))`;
//! `calibration_digest` rebuilds that string by hand. A divergence would
//! make every capture look like it came from the wrong set, or — worse —
//! make a wrong set look right.

use super::super::sensitivity::{calibration_digest, Entry};

fn entries(json: &str) -> Vec<Entry> {
    serde_json::from_str(json).expect("fixture parses")
}

/// Cross-language contract. The expected value was produced by Python:
///
/// ```text
/// >>> json.dumps([{"id":"code-d00","ids":[9906,1917,13]},
/// ...             {"id":"prose-d01","ids":[1,2]}],
/// ...            sort_keys=True, separators=(",",":"))
/// '[{"id":"code-d00","ids":[9906,1917,13]},{"id":"prose-d01","ids":[1,2]}]'
/// >>> hashlib.sha256(_.encode()).hexdigest()
/// 'a464bf27fe11121faeee942571e7841855f887504b455cc3fae3a2873f4c4ffb'
/// ```
#[test]
fn digest_matches_the_python_canonical_form() {
    let e = entries(r#"[{"id":"code-d00","ids":[9906,1917,13]},{"id":"prose-d01","ids":[1,2]}]"#);
    assert_eq!(
        calibration_digest(&e),
        "a464bf27fe11121faeee942571e7841855f887504b455cc3fae3a2873f4c4ffb",
    );
}

/// Order is part of the identity: the entries run in file order, so two
/// banks holding the same prompts in a different order are different
/// captures and must not share a digest.
#[test]
fn entry_order_changes_the_digest() {
    let forward = entries(r#"[{"id":"a","ids":[1]},{"id":"b","ids":[2]}]"#);
    let reversed = entries(r#"[{"id":"b","ids":[2]},{"id":"a","ids":[1]}]"#);
    assert_ne!(calibration_digest(&forward), calibration_digest(&reversed));
}

/// A single token differing anywhere must move the digest — that is the
/// whole point of hashing ids rather than prompt text, which cannot see a
/// changed tokeniser, BOS convention or truncation.
#[test]
fn one_different_token_changes_the_digest() {
    let a = entries(r#"[{"id":"a","ids":[9906,1917,13]}]"#);
    let b = entries(r#"[{"id":"a","ids":[9906,1917,14]}]"#);
    assert_ne!(calibration_digest(&a), calibration_digest(&b));
}

/// Truncation is a real failure mode (`run_bank.py` clips at 128), so a
/// prefix must not collide with the full sequence.
#[test]
fn a_truncated_bank_is_not_the_full_bank() {
    let full = entries(r#"[{"id":"a","ids":[1,2,3,4]}]"#);
    let clipped = entries(r#"[{"id":"a","ids":[1,2]}]"#);
    assert_ne!(calibration_digest(&full), calibration_digest(&clipped));
}

/// An empty bank still hashes rather than panicking, so a mis-pointed
/// `--calibration` fails the provenance check instead of the process.
#[test]
fn an_empty_bank_hashes_the_empty_list() {
    assert_eq!(
        calibration_digest(&[]),
        // sha256("[]")
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    );
}
