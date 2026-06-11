use std::path::PathBuf;
use std::time::Instant;

use clap::Args;
use larql_vindex::{
    canonical::{
        classify_layer_regime, compute_onshell_mask, compute_whitening, estimate_covariance,
        CanonicalMeta, LayerCanonicalInfo,
    },
    format::{
        down_meta::read_cscores_binary,
        filenames::CANONICAL_META_JSON,
        load::{load_vindex_config, load_vindex_embeddings},
    },
};

#[derive(Args)]
pub struct CanonicalizeArgs {
    /// Path to the .vindex directory to canonicalize.
    vindex: PathBuf,

    /// Override the on-shell fraction (default 0.15 = top 15% by c_score).
    #[arg(long, default_value = "0.15")]
    onshell_fraction: f32,

    /// Override the covariance sample size (default 4096 embedding rows).
    #[arg(long, default_value = "4096")]
    covariance_samples: usize,
}

pub fn run(args: CanonicalizeArgs) -> Result<(), Box<dyn std::error::Error>> {
    let vindex_dir = &args.vindex;
    let onshell_fraction = args.onshell_fraction;
    let covariance_samples = args.covariance_samples;

    println!("Canonicalizing vindex at {}", vindex_dir.display());

    // ── 1. Load index.json ──────────────────────────────────────────────
    let t0 = Instant::now();
    let config = load_vindex_config(vindex_dir)?;
    println!(
        "  model: {} ({}), {} layers, hidden={}, vocab={}",
        config.model, config.family, config.num_layers, config.hidden_size, config.vocab_size
    );

    // ── 2. Load embeddings.bin ──────────────────────────────────────────
    print!("  loading embeddings.bin ... ");
    let (embed, embed_scale) = load_vindex_embeddings(vindex_dir)?;
    println!(
        "{}×{} ({:.1}ms)",
        embed.shape()[0],
        embed.shape()[1],
        t0.elapsed().as_secs_f64() * 1000.0
    );

    // ── 3. Estimate covariance G ────────────────────────────────────────
    let t1 = Instant::now();
    print!("  estimating G ({covariance_samples} samples) ... ");
    let g = estimate_covariance(&embed, embed_scale, covariance_samples);
    println!("{:.1}ms", t1.elapsed().as_secs_f64() * 1000.0);

    // ── 4. Cholesky whitening ───────────────────────────────────────────
    let t2 = Instant::now();
    print!("  Cholesky decomposition ... ");
    let whitening = compute_whitening(&g).map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;
    println!("{:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);

    // ── 5. Read c_scores from down_meta.bin ────────────────────────────
    let t3 = Instant::now();
    print!("  reading c_scores from down_meta.bin ... ");
    let cscores_per_layer = read_cscores_binary(vindex_dir)?;
    println!(
        "{} layers, {:.1}ms",
        cscores_per_layer.len(),
        t3.elapsed().as_secs_f64() * 1000.0
    );

    // ── 6. Per-layer: regime + on-shell ────────────────────────────────
    let mut layers_info: Vec<LayerCanonicalInfo> = Vec::with_capacity(config.num_layers);
    for layer in 0..config.num_layers {
        let cscores = cscores_per_layer
            .get(layer)
            .map(|v| v.as_slice())
            .unwrap_or(&[]);
        let (regime, mean_density) = classify_layer_regime(cscores);
        let mask = compute_onshell_mask(cscores, onshell_fraction);
        let on_shell_count = mask.iter().filter(|&&b| b).count();
        layers_info.push(LayerCanonicalInfo {
            layer,
            regime,
            on_shell_count,
            total_features: cscores.len(),
            mean_density,
        });
    }

    // ── 7. Build and write canonical_meta.json ─────────────────────────
    let meta = CanonicalMeta {
        version: 1,
        model: config.model.clone(),
        family: config.family.clone(),
        num_layers: config.num_layers,
        hidden_size: config.hidden_size,
        covariance_sample_size: covariance_samples,
        embed_scale,
        cholesky_l_packed: whitening.l_packed,
        layers: layers_info,
    };

    let out_path = vindex_dir.join(CANONICAL_META_JSON);
    let json = serde_json::to_string_pretty(&meta)?;
    std::fs::write(&out_path, &json)?;

    let total_on_shell: usize = meta.layers.iter().map(|l| l.on_shell_count).sum();
    let total_features: usize = meta.layers.iter().map(|l| l.total_features).sum();
    let pct = if total_features > 0 {
        100.0 * total_on_shell as f32 / total_features as f32
    } else {
        0.0
    };

    println!("  on-shell features: {total_on_shell}/{total_features} ({pct:.1}%)");
    println!("  wrote {}", out_path.display());
    println!("  total: {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);

    Ok(())
}
