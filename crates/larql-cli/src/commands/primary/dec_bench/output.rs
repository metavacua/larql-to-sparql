//! DEC replay run record — the full-fidelity JSON sidecar written by
//! `--output-file` (the pulse JSONL is the compact rig-facing view).

use serde::Serialize;

use super::capture_format::CaptureManifest;
use super::replay::DecPointSummary;

#[derive(Serialize)]
pub struct DecBenchJsonResult {
    /// Unix seconds, as a string (matches ADR-0012 bench JSON).
    pub timestamp: String,
    /// Which endpoint family the sweep measured (`--endpoint`): `walk-ffn`
    /// exercises the dense/shared-expert FFN path, `experts` the routed
    /// multi-layer path. Each point additionally records its concrete
    /// endpoint (e.g. `walk-ffn-q8k`) and code.
    pub endpoint: String,
    pub ffn_url: String,
    pub capture: CaptureSummary,
    /// `/v1/stats` echo taken before the sweep.
    pub stats_before: serde_json::Value,
    /// `/v1/stats` echo taken after the sweep (includes the server's own
    /// `layer_latency` EMA/p99 accumulated over the run).
    pub stats_after: serde_json::Value,
    /// Layers replayed. For `experts` runs this is the MoE subset of the
    /// requested range — non-MoE layers are excluded from the routed sweep
    /// and its denominator.
    pub layers: Vec<usize>,
    /// `layers.len()`, recorded explicitly so routed-run records surface
    /// how many layers survived the non-MoE exclusion.
    pub replayed_layer_count: usize,
    pub steps: usize,
    pub repeats: usize,
    pub warmup_passes: usize,
    // NOTE: `weight_bytes_tok` moved into each point's summary — the routed
    // union denominator is batch-dependent, so denominators are per-point
    // (the dense value is simply constant across points).
    /// Layers whose dense byte count was unavailable in `/v1/stats` — a
    /// non-zero value flags a partial denominator (dense endpoints only).
    pub weight_bytes_missing_layers: usize,
    /// Client-side rayon pool width at replay time (measurement-protocol
    /// context: q8k quantisation and any rayon-using codepaths share it).
    pub client_rayon_threads: usize,
    pub net_rtt_ms: Option<f64>,
    pub net_gbps: Option<f64>,
    pub points: Vec<DecPointSummary>,
}

/// Capture-pool identity embedded in the run record (not the full manifest —
/// prompt texts stay in the pool directory).
#[derive(Serialize)]
pub struct CaptureSummary {
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    pub steps: usize,
    pub num_prompts: usize,
    pub created_unix: u64,
}

impl From<&CaptureManifest> for CaptureSummary {
    fn from(m: &CaptureManifest) -> Self {
        Self {
            model: m.model.clone(),
            hidden_size: m.hidden_size,
            num_layers: m.num_layers,
            steps: m.steps,
            num_prompts: m.prompts.len(),
            created_unix: m.created_unix,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::capture_format::{CaptureManifest, PromptMeta, CAPTURE_VERSION};
    use super::*;

    #[test]
    fn capture_summary_from_manifest_drops_prompt_texts() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "gemma4-26b-a4b-q4k".into(),
            hidden_size: 5376,
            num_layers: 48,
            steps: 16,
            dtype: "f32-le".into(),
            prompts: vec![
                PromptMeta {
                    id: 0,
                    text: "secret-ish prompt text".into(),
                    steps_captured: 16,
                },
                PromptMeta {
                    id: 1,
                    text: "another".into(),
                    steps_captured: 20,
                },
            ],
            created_unix: 42,
            routing: None,
        };
        let s = CaptureSummary::from(&m);
        assert_eq!(s.num_prompts, 2);
        assert_eq!(s.steps, 16);
        let json = serde_json::to_value(&s).unwrap();
        assert!(json.get("prompts").is_none());
        assert_eq!(json["hidden_size"], 5376);
    }

    #[test]
    fn run_record_serializes_with_schema_fields() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "m".into(),
            hidden_size: 4,
            num_layers: 2,
            steps: 1,
            dtype: "f32-le".into(),
            prompts: vec![PromptMeta {
                id: 0,
                text: "p".into(),
                steps_captured: 1,
            }],
            created_unix: 0,
            routing: None,
        };
        let r = DecBenchJsonResult {
            timestamp: "123".into(),
            endpoint: "walk-ffn".into(),
            ffn_url: "http://127.0.0.1:8080".into(),
            capture: CaptureSummary::from(&m),
            stats_before: serde_json::json!({"layers": 2}),
            stats_after: serde_json::json!({"layers": 2}),
            layers: vec![0, 1],
            replayed_layer_count: 2,
            steps: 1,
            repeats: 3,
            warmup_passes: 1,
            weight_bytes_missing_layers: 0,
            client_rayon_threads: 8,
            net_rtt_ms: None,
            net_gbps: None,
            points: vec![],
        };
        let v = serde_json::to_value(&r).unwrap();
        assert_eq!(v["endpoint"], "walk-ffn");
        assert_eq!(v["capture"]["num_prompts"], 1);
        // weight_bytes_tok moved to the per-point summaries (batch-dependent
        // under the routed union denominator).
        assert!(v.get("weight_bytes_tok").is_none());
        assert_eq!(v["replayed_layer_count"], 2);
        assert_eq!(v["client_rayon_threads"], 8);
        assert!(v["points"].as_array().unwrap().is_empty());
    }
}
