//! `dec/*` pulse emission — JSONL lines consumable by the chuk-train rig.
//!
//! Contract (rig side): each line is one JSON object with a numeric `step`;
//! every other numeric field becomes a metric sample, non-numeric fields are
//! dropped. String axes (`dec/wire_format`, `dec/dispatch_mode`) are part of
//! the spec §7 schema and included for humans, with numeric `_code` twins so
//! the axes survive a numeric-only ingester.

use super::replay::DecPointSummary;

/// Build one pulse line for a sweep point. `step` is the sweep-point index.
/// Denominators (`dec/weight_bytes_tok[_naive|_union]`, `dec/movement_ratio`,
/// `dec/experts_union_frac`) come from the summary itself — per-point since
/// the routed union denominator is batch-dependent.
pub fn pulse_line(
    step: usize,
    s: &DecPointSummary,
    net_rtt_ms: Option<f64>,
    net_gbps: Option<f64>,
    per_layer: bool,
) -> serde_json::Value {
    let mut obj = serde_json::Map::new();
    obj.insert("step".into(), step.into());
    obj.insert("dec/batch".into(), s.batch.into());
    obj.insert("dec/endpoint".into(), s.endpoint.clone().into());
    obj.insert("dec/endpoint_code".into(), s.endpoint_code.into());
    obj.insert("dec/wire_format".into(), s.wire_format.clone().into());
    obj.insert("dec/wire_format_code".into(), s.wire_format_code.into());
    // Direction split (asymmetric pairs, DEC funnel v0.5 §3 DEC-1A):
    // `dec/wire_format` stays the combined label for continuity; the
    // per-direction requested dtypes ride alongside.
    obj.insert("dec/wire_in".into(), s.wire_in.clone().into());
    obj.insert("dec/wire_out".into(), s.wire_out.clone().into());
    obj.insert("dec/dispatch_mode".into(), s.dispatch_mode.clone().into());
    obj.insert("dec/dispatch_mode_code".into(), s.dispatch_mode_code.into());
    obj.insert("dec/step_ms_mean".into(), num(s.step_ms_mean));
    obj.insert("dec/step_ms_p50".into(), num(s.step_ms_p50));
    obj.insert("dec/step_ms_p99".into(), num(s.step_ms_p99));
    obj.insert("dec/tok_s".into(), num(s.tok_s));
    obj.insert("dec/payload_bytes_tok".into(), num(s.payload_bytes_tok));
    obj.insert(
        "dec/payload_bytes_tok_in".into(),
        num(s.payload_bytes_tok_in),
    );
    obj.insert(
        "dec/payload_bytes_tok_out".into(),
        num(s.payload_bytes_tok_out),
    );
    if let Some(w) = s.weight_bytes_tok {
        obj.insert("dec/weight_bytes_tok".into(), num(w));
    }
    if let Some(w) = s.weight_bytes_tok_naive {
        obj.insert("dec/weight_bytes_tok_naive".into(), num(w));
    }
    if let Some(w) = s.weight_bytes_tok_union {
        obj.insert("dec/weight_bytes_tok_union".into(), num(w));
    }
    if let Some(r) = s.movement_ratio {
        obj.insert("dec/movement_ratio".into(), num(r));
    }
    if let Some(f) = s.experts_union_frac {
        obj.insert("dec/experts_union_frac".into(), num(f));
    }
    if let Some(p50) = s.server_ms_p50 {
        obj.insert("dec/server_ms_p50".into(), num(p50));
    }
    if let Some(p99) = s.server_ms_p99 {
        obj.insert("dec/server_ms_p99".into(), num(p99));
    }
    // Two-scoreboard timing decomposition (dec-funnel §3 DEC-1A).
    // queue/encode/client_decode are client-measured and always exist
    // (queue_ms is structurally 0 at driver concurrency 1 — DEC-2's axis);
    // serve/transmit exist only when the server reported serve latency.
    obj.insert("dec/queue_ms".into(), num(s.queue_ms));
    obj.insert("dec/encode_us_p50".into(), num(s.encode_us_p50));
    obj.insert(
        "dec/client_decode_us_p50".into(),
        num(s.client_decode_us_p50),
    );
    if let Some(p50) = s.serve_us_p50 {
        obj.insert("dec/serve_us_p50".into(), num(p50));
    }
    if let Some(p99) = s.serve_us_p99 {
        obj.insert("dec/serve_us_p99".into(), num(p99));
    }
    if let Some(p50) = s.transmit_us_p50 {
        obj.insert("dec/transmit_us_p50".into(), num(p50));
    }
    if let Some(rtt) = net_rtt_ms {
        obj.insert("net/rtt_ms".into(), num(rtt));
    }
    if let Some(gbps) = net_gbps {
        obj.insert("net/gbps".into(), num(gbps));
    }
    if per_layer {
        for l in &s.per_layer {
            obj.insert(format!("dec/layer{}_ms_p50", l.layer), num(l.client_ms_p50));
            obj.insert(format!("dec/layer{}_ms_p99", l.layer), num(l.client_ms_p99));
        }
    }
    serde_json::Value::Object(obj)
}

fn num(v: f64) -> serde_json::Value {
    serde_json::Number::from_f64(v)
        .map(serde_json::Value::Number)
        .unwrap_or(serde_json::Value::Null)
}

/// Serialise pulse lines as JSONL (one compact object per line, trailing
/// newline).
pub fn to_jsonl(lines: &[serde_json::Value]) -> String {
    let mut out = String::new();
    for l in lines {
        out.push_str(&l.to_string());
        out.push('\n');
    }
    out
}

#[cfg(test)]
mod tests {
    use super::super::replay::{DecPointSummary, LayerSummary};
    use super::*;

    fn summary() -> DecPointSummary {
        DecPointSummary {
            batch: 8,
            endpoint: "walk-ffn".into(),
            endpoint_code: 0,
            wire_format: "f16".into(),
            wire_format_code: 1,
            wire_in: "f32".into(),
            wire_out: "f16".into(),
            served_wire_in: vec!["f32".into()],
            served_wire_out: vec!["f16".into()],
            dispatch_mode: "batch".into(),
            dispatch_mode_code: 1,
            steps: 16,
            step_ms_mean: 12.5,
            step_ms_p50: 12.0,
            step_ms_p99: 20.0,
            tok_s: 640.0,
            payload_bytes_tok: 21504.0,
            payload_bytes_tok_in: 21000.0,
            payload_bytes_tok_out: 504.0,
            weight_bytes_tok: Some(2.0e9),
            weight_bytes_tok_naive: None,
            weight_bytes_tok_union: None,
            movement_ratio: Some(1.1e-5),
            experts_union_frac: None,
            server_ms_p50: Some(9.0),
            server_ms_p99: Some(15.0),
            queue_ms: 0.0,
            encode_us_p50: 120.0,
            client_decode_us_p50: 80.0,
            serve_us_p50: Some(9000.0),
            serve_us_p99: Some(15000.0),
            transmit_us_p50: Some(3000.0),
            transmit_us_p99: Some(5000.0),
            transmit_us_clamped: 0,
            per_layer: vec![LayerSummary {
                layer: 3,
                client_ms_p50: 0.4,
                client_ms_p99: 0.9,
            }],
        }
    }

    #[test]
    fn pulse_line_has_numeric_step_and_schema_keys() {
        let line = pulse_line(7, &summary(), Some(0.05), Some(10.0), false);
        assert_eq!(line["step"], 7);
        assert_eq!(line["dec/batch"], 8);
        assert_eq!(line["dec/endpoint"], "walk-ffn");
        assert_eq!(line["dec/endpoint_code"], 0);
        assert_eq!(line["dec/wire_format"], "f16");
        assert_eq!(line["dec/wire_format_code"], 1);
        // Direction split: plain arms record their real request direction
        // (f32 — the historical request wire) next to the negotiated return.
        assert_eq!(line["dec/wire_in"], "f32");
        assert_eq!(line["dec/wire_out"], "f16");
        assert_eq!(line["dec/dispatch_mode_code"], 1);
        assert!(line["dec/step_ms_p50"].is_number());
        assert!(line["dec/step_ms_p99"].is_number());
        assert!(line["dec/tok_s"].is_number());
        assert!(line["dec/payload_bytes_tok"].is_number());
        assert_eq!(line["dec/payload_bytes_tok_in"], 21000.0);
        assert_eq!(line["dec/payload_bytes_tok_out"], 504.0);
        assert!(line["dec/weight_bytes_tok"].is_number());
        assert!(line["dec/movement_ratio"].is_number());
        assert!(line["dec/server_ms_p50"].is_number());
        assert!(line["net/rtt_ms"].is_number());
        assert!(line["net/gbps"].is_number());
        // Routed-only keys absent on a dense point.
        assert!(line.get("dec/weight_bytes_tok_naive").is_none());
        assert!(line.get("dec/weight_bytes_tok_union").is_none());
        assert!(line.get("dec/experts_union_frac").is_none());
        // Per-layer keys only under the flag.
        assert!(line.get("dec/layer3_ms_p50").is_none());
        // Two-scoreboard keys (dec-funnel §3 DEC-1A).
        assert_eq!(line["dec/queue_ms"], 0.0);
        assert_eq!(line["dec/encode_us_p50"], 120.0);
        assert_eq!(line["dec/client_decode_us_p50"], 80.0);
        assert_eq!(line["dec/serve_us_p50"], 9000.0);
        assert_eq!(line["dec/serve_us_p99"], 15000.0);
        assert_eq!(line["dec/transmit_us_p50"], 3000.0);
    }

    #[test]
    fn pulse_line_omits_serve_and_transmit_without_timing_data() {
        // Pre-extension server on a trailer endpoint: serve/transmit keys
        // absent; the client-measured decomposition keys stay.
        let mut s = summary();
        s.serve_us_p50 = None;
        s.serve_us_p99 = None;
        s.transmit_us_p50 = None;
        s.transmit_us_p99 = None;
        let line = pulse_line(0, &s, None, None, false);
        assert!(line.get("dec/serve_us_p50").is_none());
        assert!(line.get("dec/serve_us_p99").is_none());
        assert!(line.get("dec/transmit_us_p50").is_none());
        assert!(line["dec/queue_ms"].is_number());
        assert!(line["dec/encode_us_p50"].is_number());
        assert!(line["dec/client_decode_us_p50"].is_number());
    }

    #[test]
    fn pulse_line_routed_point_emits_naive_union_and_frac() {
        let mut s = summary();
        s.endpoint = "experts-ml-q8k".into();
        s.endpoint_code = 3;
        s.weight_bytes_tok = None;
        s.weight_bytes_tok_naive = Some(1.6e9);
        s.weight_bytes_tok_union = Some(4.0e8);
        s.movement_ratio = Some(2.2e-5);
        s.experts_union_frac = Some(0.25);
        s.server_ms_p50 = None;
        s.server_ms_p99 = None;
        let line = pulse_line(2, &s, None, None, false);
        assert_eq!(line["dec/endpoint"], "experts-ml-q8k");
        assert_eq!(line["dec/endpoint_code"], 3);
        assert!(line.get("dec/weight_bytes_tok").is_none());
        assert!(line["dec/weight_bytes_tok_naive"].is_number());
        assert!(line["dec/weight_bytes_tok_union"].is_number());
        assert!(line["dec/movement_ratio"].is_number());
        assert_eq!(line["dec/experts_union_frac"], 0.25);
        assert!(line.get("dec/server_ms_p50").is_none());
    }

    #[test]
    fn pulse_line_pair_point_emits_combined_label_and_direction_keys() {
        // Asymmetric pair arm: dec/wire_format keeps the combined label
        // for continuity; dec/wire_in and dec/wire_out carry the split.
        let mut s = summary();
        s.wire_format = "f16/i8".into();
        s.wire_format_code = 112;
        s.wire_in = "f16".into();
        s.wire_out = "i8".into();
        s.served_wire_in = vec!["f16".into()];
        s.served_wire_out = vec!["i8".into()];
        let line = pulse_line(4, &s, None, None, false);
        assert_eq!(line["dec/wire_format"], "f16/i8");
        assert_eq!(line["dec/wire_format_code"], 112);
        assert_eq!(line["dec/wire_in"], "f16");
        assert_eq!(line["dec/wire_out"], "i8");
        assert!(line["dec/payload_bytes_tok_in"].is_number());
        assert!(line["dec/payload_bytes_tok_out"].is_number());
    }

    #[test]
    fn pulse_line_optional_fields_omitted_and_per_layer_gated() {
        let mut s = summary();
        s.server_ms_p50 = None;
        s.server_ms_p99 = None;
        s.weight_bytes_tok = None;
        s.movement_ratio = None;
        let line = pulse_line(0, &s, None, None, true);
        assert!(line.get("dec/weight_bytes_tok").is_none());
        assert!(line.get("dec/movement_ratio").is_none());
        assert!(line.get("dec/server_ms_p50").is_none());
        assert!(line.get("net/rtt_ms").is_none());
        assert!(line["dec/layer3_ms_p50"].is_number());
        assert!(line["dec/layer3_ms_p99"].is_number());
    }

    #[test]
    fn to_jsonl_one_compact_line_each() {
        let lines = vec![
            pulse_line(0, &summary(), None, None, false),
            pulse_line(1, &summary(), None, None, false),
        ];
        let jsonl = to_jsonl(&lines);
        let rows: Vec<&str> = jsonl.trim_end().split('\n').collect();
        assert_eq!(rows.len(), 2);
        for (i, row) in rows.iter().enumerate() {
            let v: serde_json::Value = serde_json::from_str(row).unwrap();
            assert_eq!(v["step"], i);
        }
        assert!(jsonl.ends_with('\n'));
    }
}
