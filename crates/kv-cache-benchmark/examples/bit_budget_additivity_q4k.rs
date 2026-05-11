//! Exp37 q4k slot-bit additivity runner.
//!
//! Scores the object slot for each row in the Exp37 design matrix using exact
//! target log-probabilities from the low-memory q4k walk path, then computes
//! pairwise additivity interactions.

#[cfg(feature = "real-model")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    runner::run()
}

#[cfg(not(feature = "real-model"))]
fn main() {
    eprintln!("This example requires the 'real-model' feature.");
    std::process::exit(1);
}

#[cfg(feature = "real-model")]
mod runner {
    use std::collections::HashMap;
    use std::fs::File;
    use std::io::{BufRead, BufReader, Write};
    use std::path::PathBuf;

    use larql_inference::vindex::{predict_q4k_hidden_with_ffn, WalkFfn};
    use larql_inference::{encode_prompt, hidden_to_raw_logits, open_inference_vindex};
    use larql_vindex::{load_model_weights_q4k, load_vindex_tokenizer};
    use serde::{Deserialize, Serialize};
    use serde_json::json;

    #[derive(Debug)]
    struct Args {
        vindex: PathBuf,
        design: PathBuf,
        out_json: PathBuf,
        scored_csv: PathBuf,
        interactions_csv: PathBuf,
        top_k: usize,
        feature_top_k: usize,
    }

    #[derive(Clone, Debug, Deserialize, Serialize)]
    struct Cell {
        source_id: String,
        relation: String,
        cell: String,
        axes: String,
        template: String,
        subject: String,
        object: String,
        text: String,
        object_span_start: usize,
        object_span_end: usize,
    }

    #[derive(Clone, Debug, Serialize)]
    struct ScoredCell {
        #[serde(flatten)]
        cell: Cell,
        prefix: String,
        slot_bits_total: f64,
        slot_bits_per_token: f64,
        object_n_tokens: usize,
        clipped_tokens: usize,
        token_bits: Vec<f64>,
        token_probs: Vec<f64>,
        token_ids: Vec<u32>,
    }

    #[derive(Clone, Debug, Serialize)]
    struct Interaction {
        source_id: String,
        axis_a: String,
        axis_b: String,
        joint_cell: String,
        slot_bits_delta_a: f64,
        slot_bits_delta_b: f64,
        slot_bits_observed_joint_delta: f64,
        slot_bits_predicted_joint_delta: f64,
        slot_bits_interaction_bits: f64,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = parse_args();
        std::fs::create_dir_all(args.out_json.parent().unwrap())?;
        std::fs::create_dir_all(args.scored_csv.parent().unwrap())?;
        std::fs::create_dir_all(args.interactions_csv.parent().unwrap())?;

        let cells = load_design(&args.design)?;
        println!("Loading q4k vindex {}", args.vindex.display());
        let mut cb = larql_vindex::SilentLoadCallbacks;
        let mut weights = load_model_weights_q4k(&args.vindex, &mut cb)?;
        let tokenizer = load_vindex_tokenizer(&args.vindex)?;
        let index = open_inference_vindex(&args.vindex)?;

        let mut scored = Vec::new();
        for (idx, cell) in cells.iter().enumerate() {
            scored.push(score_cell(
                &mut weights,
                &tokenizer,
                &index,
                cell,
                args.top_k,
                args.feature_top_k,
            )?);
            if (idx + 1) % 10 == 0 {
                println!("scored {}/{}", idx + 1, cells.len());
            }
        }
        let interactions = compute_interactions(&scored);

        std::fs::write(
            &args.out_json,
            serde_json::to_string_pretty(&json!({
                "experiment": "37_bit_budget_additivity",
                "path": "q4k",
                "scoring": "exact_target_logprob",
                "vindex": args.vindex,
                "top_k_predictions": args.top_k,
                "feature_top_k": args.feature_top_k,
                "n_cells": scored.len(),
                "cells": scored,
                "interactions": interactions,
            }))?,
        )?;
        write_scored_csv(&args.scored_csv, &scored)?;
        write_interactions_csv(&args.interactions_csv, &interactions)?;
        println!("wrote {}", args.out_json.display());
        println!("wrote {}", args.scored_csv.display());
        println!("wrote {}", args.interactions_csv.display());
        Ok(())
    }

    fn parse_args() -> Args {
        let mut args = Args {
            vindex: PathBuf::from("output/gemma3-4b-q4k-v2.vindex"),
            design: PathBuf::from("~/chris-source/chris-experiments/shannon/37_bit_budget_additivity/results/design_matrix.csv"),
            out_json: PathBuf::from(
                "~/chris-source/chris-experiments/shannon/37_bit_budget_additivity/results/q4k_scored_cells.json",
            ),
            scored_csv: PathBuf::from(
                "~/chris-source/chris-experiments/shannon/37_bit_budget_additivity/results/q4k_scored_cells.csv",
            ),
            interactions_csv: PathBuf::from(
                "~/chris-source/chris-experiments/shannon/37_bit_budget_additivity/results/q4k_interactions.csv",
            ),
            top_k: 2048,
            feature_top_k: 2048,
        };
        let raw: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < raw.len() {
            match raw[i].as_str() {
                "--vindex" => {
                    i += 1;
                    args.vindex = PathBuf::from(&raw[i]);
                }
                "--design" => {
                    i += 1;
                    args.design = PathBuf::from(&raw[i]);
                }
                "--out-json" => {
                    i += 1;
                    args.out_json = PathBuf::from(&raw[i]);
                }
                "--scored-csv" => {
                    i += 1;
                    args.scored_csv = PathBuf::from(&raw[i]);
                }
                "--interactions-csv" => {
                    i += 1;
                    args.interactions_csv = PathBuf::from(&raw[i]);
                }
                "--top-k" => {
                    i += 1;
                    args.top_k = raw[i].parse().expect("--top-k must be usize");
                }
                "--feature-top-k" => {
                    i += 1;
                    args.feature_top_k = raw[i].parse().expect("--feature-top-k must be usize");
                }
                other => {
                    eprintln!("unknown arg: {other}");
                    std::process::exit(2);
                }
            }
            i += 1;
        }
        args
    }

    fn load_design(path: &PathBuf) -> Result<Vec<Cell>, Box<dyn std::error::Error>> {
        let file = File::open(path)?;
        let reader = BufReader::new(file);
        let mut lines = reader.lines();
        let header = lines.next().ok_or("empty design csv")??;
        let headers: Vec<&str> = header.split(',').collect();
        let mut out = Vec::new();
        for line in lines {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            let values: Vec<&str> = line.split(',').collect();
            if values.len() != headers.len() {
                return Err(format!("unsupported csv row with commas: {line}").into());
            }
            let mut row = HashMap::new();
            for (key, value) in headers.iter().zip(values.iter()) {
                row.insert(*key, *value);
            }
            out.push(Cell {
                source_id: get(&row, "source_id")?.to_string(),
                relation: get(&row, "relation")?.to_string(),
                cell: get(&row, "cell")?.to_string(),
                axes: get(&row, "axes")?.to_string(),
                template: get(&row, "template")?.to_string(),
                subject: get(&row, "subject")?.to_string(),
                object: get(&row, "object")?.to_string(),
                text: get(&row, "text")?.to_string(),
                object_span_start: get(&row, "object_span_start")?.parse()?,
                object_span_end: get(&row, "object_span_end")?.parse()?,
            });
        }
        Ok(out)
    }

    fn get<'a>(
        row: &'a HashMap<&str, &str>,
        key: &str,
    ) -> Result<&'a str, Box<dyn std::error::Error>> {
        row.get(key)
            .copied()
            .ok_or_else(|| format!("missing csv field {key}").into())
    }

    fn score_cell(
        weights: &mut larql_models::ModelWeights,
        tokenizer: &tokenizers::Tokenizer,
        index: &larql_vindex::VectorIndex,
        cell: &Cell,
        _top_k: usize,
        feature_top_k: usize,
    ) -> Result<ScoredCell, Box<dyn std::error::Error>> {
        let prefix = cell.text[..cell.object_span_start].to_string();
        let mut context_ids = encode_prompt(tokenizer, &*weights.arch, &prefix)?;
        let object_surface = if prefix.ends_with(char::is_whitespace) {
            cell.object.clone()
        } else {
            format!(" {}", cell.object)
        };
        let object_ids = tokenizer
            .encode(object_surface.as_str(), false)
            .map_err(|e| format!("tokenize object {:?}: {e}", cell.object))?
            .get_ids()
            .to_vec();
        let mut token_bits = Vec::new();
        let mut token_probs = Vec::new();
        let clipped = 0usize;
        for &target_id in &object_ids {
            let prob = exact_target_prob(
                weights,
                index,
                &context_ids,
                target_id as usize,
                feature_top_k,
            );
            token_probs.push(prob);
            token_bits.push(-prob.log2());
            context_ids.push(target_id);
        }
        let total = token_bits.iter().sum::<f64>();
        Ok(ScoredCell {
            cell: cell.clone(),
            prefix,
            slot_bits_total: total,
            slot_bits_per_token: total / object_ids.len().max(1) as f64,
            object_n_tokens: object_ids.len(),
            clipped_tokens: clipped,
            token_bits,
            token_probs,
            token_ids: object_ids,
        })
    }

    fn exact_target_prob(
        weights: &mut larql_models::ModelWeights,
        index: &larql_vindex::VectorIndex,
        token_ids: &[u32],
        target_id: usize,
        feature_top_k: usize,
    ) -> f64 {
        let weights_ref: &larql_models::ModelWeights =
            unsafe { &*(weights as *const larql_models::ModelWeights) };
        let walk_ffn = WalkFfn::new(weights_ref, index, feature_top_k);
        let h = predict_q4k_hidden_with_ffn(weights, token_ids, index, &walk_ffn);
        let seq_len = h.shape()[0];
        let h_last = h.slice(ndarray::s![seq_len - 1..seq_len, ..]).to_owned();
        let logits = hidden_to_raw_logits(weights, &h_last);
        let target = logits[target_id] as f64;
        let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
        let exp_sum: f64 = logits.iter().map(|&l| ((l as f64) - max_logit).exp()).sum();
        let logsumexp = max_logit + exp_sum.ln();
        (target - logsumexp).exp().max(f64::MIN_POSITIVE)
    }

    fn compute_interactions(scored: &[ScoredCell]) -> Vec<Interaction> {
        let mut by_source: HashMap<String, HashMap<String, &ScoredCell>> = HashMap::new();
        for row in scored {
            by_source
                .entry(row.cell.source_id.clone())
                .or_default()
                .insert(row.cell.cell.clone(), row);
        }
        let pairs = [
            ("syntax", "fact", "syntax_fact"),
            ("syntax", "style", "syntax_style"),
            ("fact", "style", "fact_style"),
        ];
        let mut out = Vec::new();
        for (source_id, cells) in by_source {
            let Some(base) = cells.get("base") else {
                continue;
            };
            for (axis_a, axis_b, joint) in pairs {
                let (Some(a), Some(b), Some(ab)) =
                    (cells.get(axis_a), cells.get(axis_b), cells.get(joint))
                else {
                    continue;
                };
                let delta_a = a.slot_bits_total - base.slot_bits_total;
                let delta_b = b.slot_bits_total - base.slot_bits_total;
                let observed = ab.slot_bits_total - base.slot_bits_total;
                let predicted = delta_a + delta_b;
                out.push(Interaction {
                    source_id: source_id.clone(),
                    axis_a: axis_a.to_string(),
                    axis_b: axis_b.to_string(),
                    joint_cell: joint.to_string(),
                    slot_bits_delta_a: delta_a,
                    slot_bits_delta_b: delta_b,
                    slot_bits_observed_joint_delta: observed,
                    slot_bits_predicted_joint_delta: predicted,
                    slot_bits_interaction_bits: observed - predicted,
                });
            }
        }
        out.sort_by(|a, b| {
            (&a.source_id, &a.axis_a, &a.axis_b).cmp(&(&b.source_id, &b.axis_a, &b.axis_b))
        });
        out
    }

    fn write_scored_csv(
        path: &PathBuf,
        rows: &[ScoredCell],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut f = File::create(path)?;
        writeln!(
            f,
            "source_id,relation,cell,axes,subject,object,prefix,slot_bits_total,slot_bits_per_token,object_n_tokens,clipped_tokens"
        )?;
        for row in rows {
            writeln!(
                f,
                "{},{},{},{},{},{},{},{:.6},{:.6},{},{}",
                row.cell.source_id,
                row.cell.relation,
                row.cell.cell,
                row.cell.axes,
                row.cell.subject,
                row.cell.object,
                row.prefix,
                row.slot_bits_total,
                row.slot_bits_per_token,
                row.object_n_tokens,
                row.clipped_tokens
            )?;
        }
        Ok(())
    }

    fn write_interactions_csv(
        path: &PathBuf,
        rows: &[Interaction],
    ) -> Result<(), Box<dyn std::error::Error>> {
        let mut f = File::create(path)?;
        writeln!(
            f,
            "source_id,axis_a,axis_b,joint_cell,slot_bits_delta_a,slot_bits_delta_b,slot_bits_observed_joint_delta,slot_bits_predicted_joint_delta,slot_bits_interaction_bits"
        )?;
        for row in rows {
            writeln!(
                f,
                "{},{},{},{},{:.6},{:.6},{:.6},{:.6},{:.6}",
                row.source_id,
                row.axis_a,
                row.axis_b,
                row.joint_cell,
                row.slot_bits_delta_a,
                row.slot_bits_delta_b,
                row.slot_bits_observed_joint_delta,
                row.slot_bits_predicted_joint_delta,
                row.slot_bits_interaction_bits
            )?;
        }
        Ok(())
    }
}
