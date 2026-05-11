//! `SELECT * FROM FEATURES` — list features (one row per slot)
//! optionally filtered by layer / feature id / top token.

use crate::ast::{Condition, Value};
use crate::error::LqlError;
use crate::executor::Session;

use super::format::{also_display, banner, format_also, FEATURES_DEFAULT_LIMIT};

impl Session {
    pub(crate) fn exec_select_features(
        &self,
        conditions: &[Condition],
        limit: Option<u32>,
    ) -> Result<Vec<String>, LqlError> {
        let (_path, config, patched) = self.require_vindex()?;
        let classifier = self.relation_classifier();

        let layer_filter = conditions
            .iter()
            .find(|c| c.field == "layer")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let feature_filter = conditions
            .iter()
            .find(|c| c.field == "feature")
            .and_then(|c| match c.value {
                Value::Integer(n) if n >= 0 => Some(n as usize),
                _ => None,
            });
        let token_filter = conditions
            .iter()
            .find(|c| c.field == "token" || c.field == "entity")
            .and_then(|c| match &c.value {
                Value::String(s) => Some(s.as_str()),
                _ => None,
            });

        // Default LIMIT picking:
        //   - feature_filter set  → show that feature at every layer
        //   - layer_filter set    → show every feature at that layer
        //   - neither             → page-size default (FEATURES_DEFAULT_LIMIT)
        let default_limit = if feature_filter.is_some() {
            config.num_layers
        } else if layer_filter.is_some() {
            config.intermediate_size
        } else {
            FEATURES_DEFAULT_LIMIT
        };
        let limit = limit.unwrap_or(default_limit as u32) as usize;

        let scan_layers: Vec<usize> = if let Some(l) = layer_filter {
            vec![l]
        } else {
            (0..config.num_layers).collect()
        };

        let mut out = Vec::new();
        out.push(format!(
            "{:<8} {:<8} {:<16} {:<28} {:<14} {:>8}",
            "Layer", "Feature", "Token", "Also", "Relation", "Score"
        ));
        out.push(banner(86));

        let mut count = 0;
        'outer: for layer in &scan_layers {
            let nf = patched.num_features(*layer);
            for feat in 0..nf {
                if count >= limit {
                    break 'outer;
                }
                if let Some(ff) = feature_filter {
                    if feat != ff {
                        continue;
                    }
                }
                if let Some(meta) = patched.feature_meta(*layer, feat) {
                    if let Some(tf) = token_filter {
                        if meta.top_token.to_lowercase() != tf.to_lowercase() {
                            continue;
                        }
                    }
                    let also = also_display(&format_also(&meta.top_k));
                    let rel = classifier
                        .and_then(|rc| rc.label_for_feature(*layer, feat))
                        .unwrap_or("");
                    out.push(format!(
                        "L{:<7} F{:<7} {:16} {:28} {:14} {:>8.4}",
                        layer, feat, meta.top_token, also, rel, meta.c_score
                    ));
                    count += 1;
                }
            }
        }

        if count == 0 {
            out.push("  (no matching features)".into());
        }

        Ok(out)
    }
}
