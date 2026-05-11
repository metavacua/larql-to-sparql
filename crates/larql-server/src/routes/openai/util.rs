//! Shared helpers used across OpenAI endpoints.
//!
//! These were originally duplicated in `chat.rs` and `completions.rs`
//! (and partly in `embeddings.rs`); centralised here so both buffered
//! and SSE paths share one source of truth for id formatting, time
//! stamping, stop-string handling, and the SSE error envelope.

use std::time::{SystemTime, UNIX_EPOCH};

use serde::Deserialize;

/// Stop strings — accepted as either a single string or a list.
/// OpenAI's `stop` field allows both forms.
#[derive(Deserialize, Debug, Clone)]
#[serde(untagged)]
pub enum StopSpec {
    Single(String),
    Multi(Vec<String>),
}

impl StopSpec {
    pub fn as_slice(&self) -> &[String] {
        match self {
            StopSpec::Single(s) => std::slice::from_ref(s),
            StopSpec::Multi(v) => v.as_slice(),
        }
    }
}

/// Unix epoch seconds — used as the OpenAI `created` field on every
/// response and stream chunk.
pub fn unix_now() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Generate a short hex id suffix for `cmpl-...` / `chatcmpl-...`.
/// Not cryptographically strong; uniqueness across one server lifetime
/// is sufficient.
pub fn new_id_suffix() -> String {
    use std::sync::atomic::{AtomicU64, Ordering};
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    let now_ns = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0);
    format!("{now_ns:016x}{n:08x}")
}

/// Returns true if any non-empty needle appears as a substring of
/// `haystack`. Used to halt generation on stop strings.
pub fn contains_any(haystack: &str, needles: &[String]) -> bool {
    needles
        .iter()
        .any(|n| !n.is_empty() && haystack.contains(n.as_str()))
}

/// Trim `haystack` at the first occurrence of any (non-empty) needle.
/// Used by the buffered `/v1/completions` path to chop the matched
/// stop string off the returned text.
pub fn trim_at_stop(haystack: &str, needles: &[String]) -> String {
    let mut earliest: Option<usize> = None;
    for n in needles {
        if n.is_empty() {
            continue;
        }
        if let Some(idx) = haystack.find(n.as_str()) {
            earliest = Some(earliest.map_or(idx, |e| e.min(idx)));
        }
    }
    match earliest {
        Some(i) => haystack[..i].to_string(),
        None => haystack.to_string(),
    }
}

/// Format a JSON error chunk for SSE error paths. Wraps in OpenAI's
/// `{error: {message, type}}` envelope so clients see a structured
/// failure mid-stream rather than a truncated success response.
pub fn error_chunk(msg: &str) -> String {
    serde_json::json!({"error": {"message": msg, "type": "server_error"}}).to_string()
}

/// Sampling parameters extracted from an OpenAI completions /
/// chat-completions request. Grouped into a struct so the
/// [`build_sampling_eos`] signature stays readable as we add
/// repetition penalties / future fields.
#[derive(Debug, Clone, Copy, Default)]
pub struct SamplingParams {
    pub temperature: Option<f32>,
    pub top_p: Option<f32>,
    pub seed: Option<u64>,
    pub frequency_penalty: Option<f32>,
    pub presence_penalty: Option<f32>,
}

/// Build the sampling + EOS config from OpenAI request parameters.
///
/// - `temperature`: 0.0 (or `None`) → greedy. Otherwise temperature
///   sampling. OpenAI default is 1.0, but we default to greedy when
///   the field is omitted so existing tests / curl one-liners stay
///   deterministic.
/// - `top_p`: nucleus filter; only applied when temperature > 0.
/// - `seed`: deterministic RNG. Same seed + same inputs = same tokens.
/// - `frequency_penalty` / `presence_penalty`: OpenAI repetition
///   penalties applied to per-token logits before softmax. Clamped to
///   `[-2.0, 2.0]` to match OpenAI's documented range.
/// - `stop`: extends the model's built-in EOS stop strings; first
///   match halts generation mid-stream (not post-trimmed).
pub fn build_sampling_eos(
    params: SamplingParams,
    stop_strings: &[String],
) -> (larql_inference::SamplingConfig, larql_inference::EosConfig) {
    let temp = params.temperature.unwrap_or(0.0).max(0.0);
    let mut sampling = if temp > 0.0 {
        larql_inference::SamplingConfig::temperature(temp)
    } else {
        larql_inference::SamplingConfig::greedy()
    };
    if let Some(p) = params.top_p {
        // Only honour top_p when sampling is on; for greedy it's a no-op.
        if temp > 0.0 && (0.0..=1.0).contains(&p) {
            sampling = sampling.with_top_p(p);
        }
    }
    if let Some(s) = params.seed {
        sampling = sampling.with_seed(s);
    }
    if let Some(f) = params.frequency_penalty {
        sampling = sampling.with_frequency_penalty(f.clamp(-2.0, 2.0));
    }
    if let Some(p) = params.presence_penalty {
        sampling = sampling.with_presence_penalty(p.clamp(-2.0, 2.0));
    }
    let mut eos = larql_inference::EosConfig::builtin();
    for s in stop_strings {
        if !s.is_empty() {
            eos = eos.with_stop_string(s.clone());
        }
    }
    (sampling, eos)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_spec_single_or_multi() {
        let single: StopSpec = serde_json::from_value(serde_json::json!("\\n")).unwrap();
        assert_eq!(single.as_slice(), &["\\n".to_string()]);
        let multi: StopSpec = serde_json::from_value(serde_json::json!(["a", "b"])).unwrap();
        assert_eq!(multi.as_slice(), &["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn trim_at_stop_finds_earliest() {
        let s = "hello world stop here";
        let stops = vec!["stop".to_string(), "world".to_string()];
        assert_eq!(trim_at_stop(s, &stops), "hello ");
    }

    #[test]
    fn trim_at_stop_no_match_returns_input() {
        let s = "hello world";
        let stops = vec!["xx".to_string()];
        assert_eq!(trim_at_stop(s, &stops), s);
    }

    #[test]
    fn contains_any_matches_substring() {
        let stops = vec!["END".to_string()];
        assert!(contains_any("text END more", &stops));
        assert!(!contains_any("text only", &stops));
    }

    #[test]
    fn contains_any_skips_empty_needles() {
        let stops = vec!["".to_string()];
        assert!(!contains_any("text", &stops));
    }

    #[test]
    fn new_id_suffix_is_unique_within_thread() {
        let a = new_id_suffix();
        let b = new_id_suffix();
        assert_ne!(a, b);
        assert_eq!(a.len(), b.len());
    }

    #[test]
    fn unix_now_is_recent() {
        let now = unix_now();
        // 1 Jan 2024 in unix seconds = 1704067200; safety margin against
        // a clock badly out of sync.
        assert!(now > 1_700_000_000);
    }

    #[test]
    fn error_chunk_returns_openai_shape() {
        let chunk = error_chunk("oops");
        let v: serde_json::Value = serde_json::from_str(&chunk).unwrap();
        assert_eq!(v["error"]["message"], "oops");
        assert_eq!(v["error"]["type"], "server_error");
    }

    fn p() -> SamplingParams {
        SamplingParams::default()
    }

    #[test]
    fn build_sampling_eos_defaults_to_greedy() {
        let (sampling, _eos) = build_sampling_eos(p(), &[]);
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_zero_temperature_is_greedy() {
        let params = SamplingParams {
            temperature: Some(0.0),
            top_p: Some(0.9),
            seed: Some(7),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        // Zero temperature collapses to greedy regardless of top_p / seed.
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_temperature_enables_sampling() {
        let params = SamplingParams {
            temperature: Some(0.7),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        assert!(!sampling.is_greedy());
        assert!((sampling.temperature - 0.7).abs() < 1e-6);
        assert!(sampling.top_p.is_none());
        assert!(sampling.seed.is_none());
    }

    #[test]
    fn build_sampling_eos_top_p_only_with_temperature() {
        // top_p with temperature > 0 → applied.
        let on = SamplingParams {
            temperature: Some(0.8),
            top_p: Some(0.9),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(on, &[]);
        assert_eq!(sampling.top_p, Some(0.9));

        // top_p with temperature == 0 → ignored (greedy can't nucleus).
        let off = SamplingParams {
            temperature: Some(0.0),
            top_p: Some(0.9),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(off, &[]);
        assert!(sampling.top_p.is_none());
    }

    #[test]
    fn build_sampling_eos_top_p_out_of_range_dropped() {
        // OpenAI rejects top_p > 1.0; we silently drop instead of erroring.
        let high = SamplingParams {
            temperature: Some(0.8),
            top_p: Some(1.5),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(high, &[]);
        assert!(sampling.top_p.is_none());
        let neg = SamplingParams {
            temperature: Some(0.8),
            top_p: Some(-0.1),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(neg, &[]);
        assert!(sampling.top_p.is_none());
    }

    #[test]
    fn build_sampling_eos_seed_carried_through() {
        let params = SamplingParams {
            temperature: Some(0.7),
            seed: Some(42),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        assert_eq!(sampling.seed, Some(42));
    }

    #[test]
    fn build_sampling_eos_negative_temperature_clamped() {
        let params = SamplingParams {
            temperature: Some(-0.5),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        assert!(sampling.is_greedy());
    }

    #[test]
    fn build_sampling_eos_repetition_penalties_carry_through() {
        let params = SamplingParams {
            temperature: Some(0.7),
            frequency_penalty: Some(1.5),
            presence_penalty: Some(-0.5),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        assert!((sampling.frequency_penalty - 1.5).abs() < 1e-6);
        assert!((sampling.presence_penalty - (-0.5)).abs() < 1e-6);
        assert!(sampling.has_repetition_penalty());
    }

    #[test]
    fn build_sampling_eos_repetition_penalties_clamped_to_openai_range() {
        // OpenAI documents [-2.0, 2.0]; values outside get clamped.
        let params = SamplingParams {
            temperature: Some(0.7),
            frequency_penalty: Some(5.0),
            presence_penalty: Some(-10.0),
            ..p()
        };
        let (sampling, _eos) = build_sampling_eos(params, &[]);
        assert!((sampling.frequency_penalty - 2.0).abs() < 1e-6);
        assert!((sampling.presence_penalty - (-2.0)).abs() < 1e-6);
    }

    #[test]
    fn build_sampling_eos_stop_strings_added() {
        let (_, eos_baseline) = build_sampling_eos(p(), &[]);
        let (_, eos) = build_sampling_eos(p(), &["\n\n".into(), "STOP".into()]);
        assert_eq!(eos.stop_strings.len(), eos_baseline.stop_strings.len() + 2);
        assert!(eos.stop_strings.iter().any(|s| s == "\n\n"));
        assert!(eos.stop_strings.iter().any(|s| s == "STOP"));
    }

    #[test]
    fn build_sampling_eos_empty_stop_strings_skipped() {
        let (_, eos_baseline) = build_sampling_eos(p(), &[]);
        let (_, eos) = build_sampling_eos(p(), &["".into(), "x".into()]);
        assert_eq!(eos.stop_strings.len(), eos_baseline.stop_strings.len() + 1);
        assert!(eos.stop_strings.iter().any(|s| s == "x"));
        assert!(!eos.stop_strings.iter().any(|s| s.is_empty()));
    }
}
