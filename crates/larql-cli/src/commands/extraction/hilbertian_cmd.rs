use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{
    canonical::{
        complex_structure_split_half, head_block, head_hilbertian_residual, kv_head_for_query,
        HeadHilbertianInfo, HilbertianMeta,
    },
    format::{attn_load::load_attention_qk, filenames::HILBERTIAN_META_JSON, load::load_vindex_config},
};

#[derive(Args)]
pub struct HilbertianArgs {
    /// Path to the .vindex directory to analyse.
    vindex: PathBuf,
}

pub fn run(args: HilbertianArgs) -> Result<(), Box<dyn std::error::Error>> {
    let dir = &args.vindex;
    let t0 = Instant::now();
    println!("Hilbertian residual for vindex at {}", dir.display());

    let config = load_vindex_config(dir)?;
    let mc = config
        .model_config
        .as_ref()
        .ok_or("hilbertian: index.json has no model_config (need head_dim / head counts)")?;
    let (num_q, num_kv, head_dim, hidden) =
        (mc.num_q_heads, mc.num_kv_heads, mc.head_dim, config.hidden_size);
    println!(
        "  model: {} ({}), {} layers, hidden={hidden}, q_heads={num_q}, kv_heads={num_kv}, head_dim={head_dim}",
        config.model, config.family, config.num_layers
    );

    let j = complex_structure_split_half(head_dim);

    print!("  loading attention Q/K ... ");
    let qk = load_attention_qk(dir, config.num_layers)?;
    println!("{} layers ({:.1}ms)", qk.len(), t0.elapsed().as_secs_f64() * 1000.0);

    let mut heads: Vec<HeadHilbertianInfo> = Vec::with_capacity(config.num_layers * num_q);
    for (layer, (wq, wk)) in qk.iter().enumerate() {
        if wq.nrows() < num_q * head_dim || wk.nrows() < num_kv * head_dim {
            return Err(format!(
                "layer {layer}: attention weights too small (wq {} rows, wk {} rows; \
                 need {} and {}) — config/weights mismatch",
                wq.nrows(),
                wk.nrows(),
                num_q * head_dim,
                num_kv * head_dim
            )
            .into());
        }
        let wq64 = wq.mapv(|v| v as f64);
        let wk64 = wk.mapv(|v| v as f64);
        for h in 0..num_q {
            let g = kv_head_for_query(h, num_q, num_kv);
            let wq_h = head_block(&wq64, h, head_dim);
            let wk_g = head_block(&wk64, g, head_dim);
            let residual = head_hilbertian_residual(&wq_h, &wk_g, &j);
            heads.push(HeadHilbertianInfo { layer, query_head: h, kv_head: g, residual });
        }
    }

    let meta = HilbertianMeta {
        version: 1,
        model: config.model.clone(),
        hidden_size: hidden,
        head_dim,
        num_q_heads: num_q,
        num_kv_heads: num_kv,
        complex_structure: "split_half".into(),
        heads,
    };

    let out = dir.join(HILBERTIAN_META_JSON);
    std::fs::write(&out, serde_json::to_string_pretty(&meta)?)?;

    let rs: Vec<f64> = meta.heads.iter().map(|h| h.residual).collect();
    let mean = rs.iter().sum::<f64>() / rs.len() as f64;
    let min = rs.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = rs.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  residual over {} heads: mean {:.4}, min {:.4}, max {:.4}",
        rs.len(),
        mean,
        min,
        max
    );
    println!("  wrote {}", out.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    Ok(())
}
