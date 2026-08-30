//! Pure helpers for the local Metal/CPU bench path. The I/O-heavy
//! `run_larql` body lives in `local_runtime.rs`; this file owns:
//!   * `backend_name_for` — `"larql-metal"` / `"larql-cpu"`
//!   * `format_early_stop_note` — note string for partial / no-decode runs
//!   * `append_cpu_fallback_note` — makes CPU Q4K fallback rows explicit
//!   * `format_q4k_cache_log` — verbose `-v` cache-stats line
//!
//! All exercised by tests in this file.

/// Returns the table-row backend label for the local bench.
pub(super) fn backend_name_for(metal: bool) -> &'static str {
    if metal {
        "larql-metal"
    } else {
        "larql-cpu"
    }
}

/// Note string for the local bench row: either empty (full target reached),
/// "early stop @n/target …" (partial), or "no decode steps completed …"
/// when `measured_n == 0`.
pub(super) fn format_early_stop_note(
    measured_n: usize,
    target_tokens: usize,
    wall_ms: f64,
) -> String {
    if measured_n == 0 {
        format!("no decode steps completed (wall {:.0}ms)", wall_ms)
    } else if measured_n < target_tokens {
        format!(
            "early stop @{}/{} (EOS or GPU fallback)",
            measured_n, target_tokens
        )
    } else {
        String::new()
    }
}

/// Annotates CPU rows with which Q4K sub-path ran. The cached path
/// uses prefill + KV-cached single-row decode; the legacy path
/// reprocesses the full sequence at every step.
pub(super) fn append_cpu_fallback_note(note: String, cached: bool) -> String {
    let tag = if cached {
        "cpu q4k (KV-cached decode)"
    } else {
        "cpu q4k legacy (O(N²) per-step)"
    };
    if note.is_empty() {
        tag.to_string()
    } else {
        format!("{note}; {tag}")
    }
}

/// Stable fingerprint of a generated token sequence, as 16 hex digits.
///
/// Hashes both the token text and its probability bits, because the
/// probability moves for divergences too small to change the argmax —
/// which is most of them. An early-stop note only fires when a divergence
/// happens to cross an argmax boundary, so it is a far blunter detector
/// than this and needs correspondingly more samples to see the same bug.
///
/// `f64::to_bits` rather than the float: `NaN != NaN` would make a NaN run
/// unequal to itself and read as non-determinism, when the defect worth
/// reporting is that a NaN reached the sampler at all. Bit equality says
/// "same bytes", which is the question being asked.
pub(super) fn generation_fingerprint(tokens: &[(String, f64)]) -> String {
    use std::hash::{Hash, Hasher};
    let mut h = std::collections::hash_map::DefaultHasher::new();
    // Length first, so ["ab"] and ["a","b"] cannot collide via the text.
    tokens.len().hash(&mut h);
    for (text, prob) in tokens {
        text.hash(&mut h);
        prob.to_bits().hash(&mut h);
    }
    format!("{:016x}", h.finish())
}

/// Tags a row with its repeat index and generation fingerprint. Single-run
/// benches (`--repeat 1`) get the fingerprint but no index, since there is
/// no sequence for an index to place it in.
pub(super) fn append_repeat_note(
    note: String,
    repeat_idx: usize,
    repeats: usize,
    fp: &str,
) -> String {
    let tag = if repeats > 1 {
        format!("#{} fp={fp}", repeat_idx + 1)
    } else {
        format!("fp={fp}")
    };
    if note.is_empty() {
        tag
    } else {
        format!("{note}; {tag}")
    }
}

/// Verbose log line for the Q4K dequant-cache stats after a run.
pub(super) fn format_q4k_cache_log(backend_label: &str, slots: usize, bytes: usize) -> String {
    format!(
        "[bench] kquant_ffn_cache after {}: {} populated slots, {:.1} MB",
        backend_label,
        slots,
        bytes as f64 / 1_048_576.0,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn backend_name_for_picks_label() {
        assert_eq!(backend_name_for(true), "larql-metal");
        assert_eq!(backend_name_for(false), "larql-cpu");
    }

    #[test]
    fn early_stop_note_empty_when_target_reached() {
        assert!(format_early_stop_note(50, 50, 1234.0).is_empty());
    }

    #[test]
    fn early_stop_note_reports_partial_when_below_target() {
        let s = format_early_stop_note(20, 50, 5000.0);
        assert!(s.starts_with("early stop @20/50"));
        assert!(s.contains("EOS or GPU fallback"));
    }

    #[test]
    fn early_stop_note_reports_wall_when_zero_steps() {
        let s = format_early_stop_note(0, 50, 1234.0);
        assert!(s.starts_with("no decode steps completed"));
        assert!(s.contains("1234ms"));
    }

    fn toks(v: &[(&str, f64)]) -> Vec<(String, f64)> {
        v.iter().map(|(t, p)| ((*t).to_string(), *p)).collect()
    }

    #[test]
    fn fingerprint_is_stable_for_identical_sequences() {
        let a = toks(&[("hello", 0.5), (" world", 0.25)]);
        let b = toks(&[("hello", 0.5), (" world", 0.25)]);
        assert_eq!(generation_fingerprint(&a), generation_fingerprint(&b));
        assert_eq!(generation_fingerprint(&a).len(), 16);
    }

    #[test]
    fn fingerprint_moves_on_a_probability_that_leaves_the_token_alone() {
        // The whole point of hashing probabilities: this pair is identical
        // to any detector that only watches which token was picked.
        let a = toks(&[("hello", 0.5)]);
        let b = toks(&[("hello", 0.5000000001)]);
        assert_ne!(generation_fingerprint(&a), generation_fingerprint(&b));
    }

    #[test]
    fn fingerprint_moves_on_token_text_and_on_order() {
        let a = toks(&[("hello", 0.5), ("world", 0.25)]);
        let b = toks(&[("hellp", 0.5), ("world", 0.25)]);
        let c = toks(&[("world", 0.25), ("hello", 0.5)]);
        assert_ne!(generation_fingerprint(&a), generation_fingerprint(&b));
        assert_ne!(generation_fingerprint(&a), generation_fingerprint(&c));
    }

    #[test]
    fn fingerprint_does_not_collide_across_a_token_boundary() {
        // Without the length prefix these hash the same text stream.
        let a = toks(&[("ab", 1.0)]);
        let b = toks(&[("a", 1.0), ("b", 1.0)]);
        assert_ne!(generation_fingerprint(&a), generation_fingerprint(&b));
    }

    #[test]
    fn fingerprint_of_a_nan_run_equals_itself() {
        // `NaN != NaN` would make a NaN run read as non-determinism. The
        // reportable defect is that a NaN reached the sampler, not that
        // two NaNs compared unequal.
        let a = toks(&[("x", f64::NAN)]);
        let b = toks(&[("x", f64::NAN)]);
        assert_eq!(generation_fingerprint(&a), generation_fingerprint(&b));
    }

    #[test]
    fn fingerprint_of_an_empty_generation_is_defined() {
        // A zero-step run still needs a comparable row.
        assert_eq!(generation_fingerprint(&[]).len(), 16);
    }

    #[test]
    fn repeat_note_indexes_only_when_repeating() {
        assert_eq!(append_repeat_note(String::new(), 0, 1, "ff"), "fp=ff");
        assert_eq!(append_repeat_note(String::new(), 0, 4, "ff"), "#1 fp=ff");
        assert_eq!(append_repeat_note(String::new(), 3, 4, "ff"), "#4 fp=ff");
    }

    #[test]
    fn repeat_note_preserves_an_existing_note() {
        let n = append_repeat_note("early stop @5/32".to_string(), 1, 4, "ab");
        assert_eq!(n, "early stop @5/32; #2 fp=ab");
    }

    #[test]
    fn cpu_fallback_note_labels_cached_vs_legacy() {
        assert_eq!(
            append_cpu_fallback_note(String::new(), true),
            "cpu q4k (KV-cached decode)"
        );
        assert_eq!(
            append_cpu_fallback_note(String::new(), false),
            "cpu q4k legacy (O(N²) per-step)"
        );
    }

    #[test]
    fn cpu_fallback_note_appends_to_existing_note() {
        let s = append_cpu_fallback_note("early stop @4/5".to_string(), true);
        assert_eq!(s, "early stop @4/5; cpu q4k (KV-cached decode)");
    }

    #[test]
    fn q4k_cache_log_reports_slots_and_mb() {
        let s = format_q4k_cache_log("larql-metal", 12, 16 * 1024 * 1024);
        assert!(s.contains("after larql-metal"));
        assert!(s.contains("12 populated slots"));
        assert!(s.contains("16.0 MB"));
    }

    #[test]
    fn q4k_cache_log_zero_bytes_shows_zero() {
        let s = format_q4k_cache_log("larql-cpu", 0, 0);
        assert!(s.contains("0 populated slots"));
        assert!(s.contains("0.0 MB"));
    }
}
