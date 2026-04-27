//! `larql slice <SRC> -o <DST> --parts a,b,c` — carve a subset of a vindex.
//!
//! Pure file-I/O subcommand. Copies a filtered set of files from an existing
//! vindex directory to a new one, rewriting `index.json` so `extract_level`
//! and `has_model_weights` reflect what's actually present. No re-download,
//! no re-extract from the source model — operates only on the built
//! artifact.
//!
//! Useful for building multiple deployment variants from a single extract:
//!
//!   * **client**   — attention + embed + norms + tokenizer (laptop; pairs
//!                    with `larql run --ffn URL`)
//!   * **server**   — gate vectors + FFN + down_meta (FFN-service host;
//!                    pairs with `larql serve --ffn-only`)
//!   * **browse**   — gate + embed + down_meta (DESCRIBE/WALK only, no
//!                    forward pass)
//!   * **router**   — index + tokenizer + router_weights (ADR-0003 MoE
//!                    router; dense vindexes don't have router_weights.bin
//!                    so this preset errors out for dense models)
//!
//! The three dense presets (`client`, `server`, `browse`) work on every
//! vindex this repo produces. See `docs/adr/0006-q4k-remote-ffn.md` for the
//! dense-remote topology these presets were cut to serve.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

use clap::Args;

use crate::commands::primary::cache;

// ─── Parts catalogue ─────────────────────────────────────────────────────
//
// Each `Part` maps to one or more filename patterns. The `index.json` +
// tokenizer come along implicitly so the output is always a loadable
// vindex; everything else is opt-in.

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Part {
    Embed,
    Norms,
    Attn,
    Gate,
    DownMeta,
    Ffn,
    LmHead,
    Router,
    Tokenizer,
    Manifest,
    Labels,
    Readme,
}

impl Part {
    fn parse(s: &str) -> Option<Self> {
        match s.trim().to_ascii_lowercase().as_str() {
            "embed" | "embeddings" => Some(Self::Embed),
            "norms" | "norm" => Some(Self::Norms),
            "attn" | "attention" => Some(Self::Attn),
            "gate" | "gate_vectors" | "gates" => Some(Self::Gate),
            "down_meta" | "meta" => Some(Self::DownMeta),
            "ffn" | "interleaved" | "up_down" => Some(Self::Ffn),
            "lm_head" | "lmhead" => Some(Self::LmHead),
            "router" | "router_weights" => Some(Self::Router),
            "tokenizer" | "tok" => Some(Self::Tokenizer),
            "manifest" | "weight_manifest" => Some(Self::Manifest),
            "labels" | "clusters" => Some(Self::Labels),
            "readme" => Some(Self::Readme),
            _ => None,
        }
    }

    /// Files matched by this part. Patterns are matched case-sensitively
    /// against each basename in the source directory. Prefix matches on
    /// `attn_weights_` etc. pick up quantisation variants (q4, q4k, q8).
    fn matches(self, filename: &str) -> bool {
        match self {
            Self::Embed => filename == "embeddings.bin",
            Self::Norms => filename == "norms.bin",
            Self::Attn => filename.starts_with("attn_weights"),
            Self::Gate => {
                filename == "gate_vectors.bin" || filename.starts_with("gate_vectors_")
            }
            Self::DownMeta => filename == "down_meta.bin" || filename == "down_meta.jsonl",
            Self::Ffn => {
                filename.starts_with("interleaved")
                    || filename == "up_weights.bin"
                    || filename == "down_weights.bin"
                    || filename == "up_features.bin"
                    || filename == "down_features.bin"
            }
            Self::LmHead => filename.starts_with("lm_head"),
            Self::Router => filename == "router_weights.bin",
            Self::Tokenizer => filename == "tokenizer.json",
            Self::Manifest => filename == "weight_manifest.json",
            Self::Labels => {
                filename == "feature_labels.json"
                    || filename == "feature_clusters.jsonl"
                    || filename == "relation_clusters.json"
            }
            Self::Readme => filename == "README.md",
        }
    }
}

/// Preset part-sets. Expansion is deterministic; `--parts` overrides take
/// precedence when both are passed.
pub fn preset_parts(preset: &str) -> Result<BTreeSet<Part>, String> {
    use Part::*;
    // Note: `embed` + `norms` appear in the server preset because
    // `load_model_weights_q4k` unconditionally opens `embeddings.bin` at
    // load time and pulls norms from `weight_manifest.json`. The server
    // doesn't run attention, but it still needs embed + norms to
    // instantiate a ModelWeights struct for the walk-ffn handler.
    let set: &[Part] = match preset.to_ascii_lowercase().as_str() {
        // Default 2-tier client (holds the embedding table locally).
        // Pairs with `larql run --ffn URL`.
        "client" => &[Embed, Norms, Attn, Tokenizer, Manifest, Labels],
        // 3-tier client (ADR-0008). Attention only — embeddings +
        // tokenizer are delegated to a remote embed server, FFN to the
        // remote FFN server. Smallest client footprint (~1 GB on 4B).
        // Pairs with `larql run --embed URL --ffn URL` (embed-URL flag
        // lands with the embed-server work).
        "attn" | "attention" => &[Norms, Attn, Manifest, Labels],
        // Embed-server slice. Pairs with `larql serve --embed-only`
        // (ADR-0008). No attention, no FFN — just the embedding table
        // + tokenizer. Memory-bound service; one server can fan out to
        // many attention workers.
        "embed" | "embed-server" => &[Embed, Tokenizer, Labels],
        "server" | "ffn" | "ffn-service" => {
            &[Embed, Norms, Gate, DownMeta, Ffn, Tokenizer, Manifest, Labels]
        }
        "browse" => &[Embed, Gate, DownMeta, Tokenizer, Labels, Readme],
        "router" => &[Router, Tokenizer, Manifest, Labels, Readme],
        "all" => &[
            Embed, Norms, Attn, Gate, DownMeta, Ffn, LmHead, Router, Tokenizer,
            Manifest, Labels, Readme,
        ],
        other => {
            return Err(format!(
                "unknown preset '{other}'. Expected: client, attn, embed, server, browse, router, all"
            ));
        }
    };
    Ok(set.iter().copied().collect())
}

// ─── CLI ─────────────────────────────────────────────────────────────────

#[derive(Args)]
pub struct SliceArgs {
    /// Source vindex: directory, `hf://owner/name`, `owner/name`, or cache shorthand.
    pub source: String,

    /// Destination directory. Must not exist unless `--force`.
    #[arg(short = 'o', long)]
    pub output: PathBuf,

    /// Comma-separated parts to include.
    ///
    /// Valid names: `embed`, `norms`, `attn`, `gate`, `down_meta`, `ffn`,
    /// `lm_head`, `router`, `tokenizer`, `manifest`, `labels`, `readme`.
    /// `index.json` is always copied.
    ///
    /// Mutually compatible with `--preset` (the union is taken).
    #[arg(long, value_delimiter = ',')]
    pub parts: Vec<String>,

    /// Preset that expands to a part list:
    ///   * `client`  — attn + embed + norms + tokenizer (2-tier; pairs with `larql run --ffn URL`)
    ///   * `attn`    — attn + norms only (3-tier; pairs with `larql run --embed URL --ffn URL`, ADR-0008)
    ///   * `embed`   — embed + tokenizer (embed-server slice; pairs with `larql serve --embed-only`)
    ///   * `server`  — gate + ffn + down_meta + embed + norms + tokenizer (pairs with `larql serve --ffn-only`)
    ///   * `browse`  — gate + embed + down_meta (no forward pass)
    ///   * `router`  — router_weights + tokenizer (MoE router; dense models error out)
    ///   * `all`     — every part (full vindex, useful for `--force` clones)
    #[arg(long)]
    pub preset: Option<String>,

    /// Overwrite `--output` if it already exists.
    #[arg(long)]
    pub force: bool,

    /// Preview what would be copied without writing anything.
    #[arg(long)]
    pub dry_run: bool,
}

/// Outcome of a slice operation — what got copied, skipped, and how the
/// destination `index.json` was rewritten. Returned by the testable core
/// so integration tests can assert behaviour without parsing stdout.
#[derive(Debug)]
pub struct SliceOutcome {
    pub source: PathBuf,
    pub destination: PathBuf,
    pub parts: BTreeSet<Part>,
    /// (basename, byte count). Sorted by name. In dry-run mode these are
    /// the files that *would* be copied.
    pub copied: Vec<(String, u64)>,
    pub skipped: Vec<String>,
    pub source_level: larql_vindex::ExtractLevel,
    pub new_level: larql_vindex::ExtractLevel,
    pub new_has_weights: bool,
    pub total_bytes: u64,
    pub dry_run: bool,
}

/// Library-callable slice. Doesn't print or touch the global cache — all
/// resolution is the caller's responsibility. `run` wraps this with CLI
/// prints and `cache::resolve_model` lookup.
pub fn slice_vindex(
    src: &Path,
    dst: &Path,
    parts: BTreeSet<Part>,
    force: bool,
    dry_run: bool,
) -> Result<SliceOutcome, Box<dyn std::error::Error>> {
    if !src.is_dir() {
        return Err(format!("source vindex not a directory: {}", src.display()).into());
    }
    if !src.join("index.json").exists() {
        return Err(format!(
            "source vindex missing index.json: {}",
            src.display()
        )
        .into());
    }
    if parts.is_empty() {
        return Err("no parts selected".into());
    }
    if dst.exists() && !force {
        return Err(format!(
            "output path exists: {} (pass --force to overwrite)",
            dst.display()
        )
        .into());
    }
    if dst == src {
        return Err("--output must differ from source vindex".into());
    }

    // Enumerate source files.
    let mut copied: Vec<(String, u64)> = Vec::new();
    let mut copy_paths: Vec<PathBuf> = Vec::new();
    let mut skipped: Vec<String> = Vec::new();
    for entry in std::fs::read_dir(src)? {
        let entry = entry?;
        let meta = entry.metadata()?;
        if !meta.is_file() {
            continue;
        }
        let name_os = entry.file_name();
        let name = match name_os.to_str() {
            Some(s) => s.to_string(),
            None => continue,
        };
        let kept = name == "index.json" || parts.iter().any(|p| p.matches(&name));
        if kept {
            copy_paths.push(entry.path());
            copied.push((name, meta.len()));
        } else {
            skipped.push(name);
        }
    }
    copied.sort_by(|a, b| a.0.cmp(&b.0));
    copy_paths.sort_by(|a, b| a.file_name().cmp(&b.file_name()));
    skipped.sort();
    let total_bytes = copied.iter().map(|(_, n)| *n).sum();

    // Compute rewritten config fields.
    // `has_model_weights` is true whenever attention OR FFN compute weights
    // are present — either is enough to justify the q4k loader opening
    // norms + PLE tensors through weight_manifest.json. Setting it to
    // `false` on a client slice (attn-only) would make `larql run` refuse
    // to load with "vindex does not contain model weights".
    let cfg = larql_vindex::load_vindex_config(src)?;
    let new_level = effective_level(&parts, cfg.extract_level);
    let new_has_weights = parts.contains(&Part::Ffn) || parts.contains(&Part::Attn);

    let outcome = SliceOutcome {
        source: src.to_path_buf(),
        destination: dst.to_path_buf(),
        parts,
        copied,
        skipped,
        source_level: cfg.extract_level,
        new_level,
        new_has_weights,
        total_bytes,
        dry_run,
    };

    if dry_run {
        return Ok(outcome);
    }

    // Write output.
    if dst.exists() && force {
        std::fs::remove_dir_all(dst)?;
    }
    std::fs::create_dir_all(dst)?;

    for src_path in &copy_paths {
        let name = src_path.file_name().unwrap();
        let dst_path = dst.join(name);
        if name == std::ffi::OsStr::new("index.json") {
            let mut new_cfg = cfg.clone();
            new_cfg.extract_level = new_level;
            new_cfg.has_model_weights = new_has_weights;
            let json = serde_json::to_string_pretty(&new_cfg)?;
            std::fs::write(&dst_path, json)?;
        } else {
            std::fs::copy(src_path, &dst_path)?;
        }
    }

    Ok(outcome)
}

pub fn run(args: SliceArgs) -> Result<(), Box<dyn std::error::Error>> {
    // 1. Resolve source through the cache shorthand.
    let src = cache::resolve_model(&args.source)?;

    // 2. Build requested part set (parts ∪ preset expansion).
    let mut wanted: BTreeSet<Part> = BTreeSet::new();
    if let Some(ref p) = args.preset {
        wanted.extend(preset_parts(p)?);
    }
    for raw in &args.parts {
        match Part::parse(raw) {
            Some(p) => {
                wanted.insert(p);
            }
            None => return Err(format!(
                "unknown part '{raw}'. Run `larql slice --help` for valid names."
            )
            .into()),
        }
    }
    if wanted.is_empty() {
        return Err(
            "no parts selected. Pass `--parts a,b,c` or `--preset client|server|browse|router`."
                .into(),
        );
    }

    // 3. Delegate to the testable core.
    let outcome = slice_vindex(&src, &args.output, wanted, args.force, args.dry_run)?;

    // 4. Report what happened.
    println!("Source:         {}", outcome.source.display());
    println!("Destination:    {}", outcome.destination.display());
    println!("Preset:         {}", args.preset.as_deref().unwrap_or("—"));
    let names: Vec<&'static str> = outcome.parts.iter().map(part_name).collect();
    println!("Parts:          {}", names.join(", "));
    println!(
        "Extract level:  {} → {}",
        outcome.source_level, outcome.new_level
    );
    println!(
        "FFN weights:    {}",
        if outcome.new_has_weights { "present" } else { "absent" }
    );

    println!(
        "\nCopying {} file(s) — {}:",
        outcome.copied.len(),
        human_size(outcome.total_bytes)
    );
    for (name, size) in &outcome.copied {
        println!("  {:<36} {:>12}", name, human_size(*size));
    }
    if !outcome.skipped.is_empty() {
        println!("\nSkipping {} file(s):", outcome.skipped.len());
        for name in &outcome.skipped {
            println!("  {name}");
        }
    }
    if outcome.dry_run {
        println!("\n(dry run — no files written)");
    } else {
        println!(
            "\nWrote {} — {}",
            outcome.destination.display(),
            human_size(outcome.total_bytes)
        );
    }
    Ok(())
}

fn effective_level(
    parts: &BTreeSet<Part>,
    source_level: larql_vindex::ExtractLevel,
) -> larql_vindex::ExtractLevel {
    use larql_vindex::ExtractLevel::*;
    // Bottom-up: each tier requires strictly more parts than the one below.
    let have_attn = parts.contains(&Part::Attn) && parts.contains(&Part::Norms);
    let have_ffn = parts.contains(&Part::Ffn);
    let have_lm_head = parts.contains(&Part::LmHead);
    let candidate = if have_attn && have_ffn && have_lm_head {
        All
    } else if have_attn && have_ffn {
        Inference
    } else if have_attn {
        Attention
    } else {
        Browse
    };
    // Never claim a higher level than the source.
    candidate.min(source_level)
}

fn part_name(p: &Part) -> &'static str {
    match p {
        Part::Embed => "embed",
        Part::Norms => "norms",
        Part::Attn => "attn",
        Part::Gate => "gate",
        Part::DownMeta => "down_meta",
        Part::Ffn => "ffn",
        Part::LmHead => "lm_head",
        Part::Router => "router",
        Part::Tokenizer => "tokenizer",
        Part::Manifest => "manifest",
        Part::Labels => "labels",
        Part::Readme => "readme",
    }
}

fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}

// ─── Tests ───────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn part_parse_aliases() {
        assert_eq!(Part::parse("attn"), Some(Part::Attn));
        assert_eq!(Part::parse("attention"), Some(Part::Attn));
        assert_eq!(Part::parse("Embeddings"), Some(Part::Embed));
        assert_eq!(Part::parse("unknown"), None);
    }

    #[test]
    fn attn_matches_quant_variants() {
        assert!(Part::Attn.matches("attn_weights.bin"));
        assert!(Part::Attn.matches("attn_weights_q4.bin"));
        assert!(Part::Attn.matches("attn_weights_q4k.bin"));
        assert!(Part::Attn.matches("attn_weights_q4k_manifest.json"));
        assert!(!Part::Attn.matches("gate_vectors.bin"));
    }

    #[test]
    fn ffn_matches_interleaved_and_hidden_major() {
        assert!(Part::Ffn.matches("interleaved.bin"));
        assert!(Part::Ffn.matches("interleaved_q4k.bin"));
        assert!(Part::Ffn.matches("up_weights.bin"));
        assert!(Part::Ffn.matches("down_features.bin"));
        // Gate vectors are their own part even though they share the FFN role.
        assert!(!Part::Ffn.matches("gate_vectors.bin"));
    }

    #[test]
    fn preset_client_is_attention_tier() {
        let parts = preset_parts("client").unwrap();
        assert!(parts.contains(&Part::Attn));
        assert!(parts.contains(&Part::Norms));
        assert!(parts.contains(&Part::Embed));
        assert!(parts.contains(&Part::Tokenizer));
        // Client slice must NOT carry FFN compute weights — defeats the point.
        assert!(!parts.contains(&Part::Ffn));
    }

    #[test]
    fn preset_server_carries_ffn_not_attention() {
        let parts = preset_parts("server").unwrap();
        assert!(parts.contains(&Part::Ffn));
        assert!(parts.contains(&Part::Gate));
        assert!(parts.contains(&Part::DownMeta));
        // FFN-service server runs no attention → skip attn weights.
        assert!(!parts.contains(&Part::Attn));
        // …but it still needs embed + norms: `load_model_weights_q4k`
        // unconditionally reads embeddings.bin and pulls norms from the
        // weight manifest. Omitting them crashes the server on startup
        // with "No such file or directory".
        assert!(parts.contains(&Part::Embed));
        assert!(parts.contains(&Part::Norms));
    }

    #[test]
    fn preset_unknown_errors() {
        assert!(preset_parts("xyz").is_err());
    }

    #[test]
    fn preset_attn_is_attention_without_embed() {
        // 3-tier client — attn + norms only. Embedding table is
        // delegated to an embed server per ADR-0008, so we specifically
        // must NOT include Part::Embed. Size win on 4B is ~2.7 GB.
        let parts = preset_parts("attn").unwrap();
        assert!(parts.contains(&Part::Attn));
        assert!(parts.contains(&Part::Norms));
        assert!(!parts.contains(&Part::Embed), "attn preset must drop embed");
        assert!(!parts.contains(&Part::Gate));
        assert!(!parts.contains(&Part::Ffn));
        assert!(!parts.contains(&Part::Tokenizer), "tokenizer lives with embed server");
    }

    #[test]
    fn preset_attn_alias_attention() {
        // `attention` is a spelling alias for `attn` — same part set.
        let a = preset_parts("attn").unwrap();
        let b = preset_parts("attention").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn preset_embed_carries_embed_and_tokenizer_only() {
        // Embed-server slice. The server from ADR-0008 needs the
        // embedding table + tokenizer; it doesn't run any compute so
        // attention, gate, and FFN all stay out.
        let parts = preset_parts("embed").unwrap();
        assert!(parts.contains(&Part::Embed));
        assert!(parts.contains(&Part::Tokenizer));
        assert!(!parts.contains(&Part::Attn));
        assert!(!parts.contains(&Part::Gate));
        assert!(!parts.contains(&Part::Ffn));
        assert!(!parts.contains(&Part::Norms), "embed server doesn't run attention — no norms");
    }

    #[test]
    fn preset_embed_alias_embed_server() {
        let a = preset_parts("embed").unwrap();
        let b = preset_parts("embed-server").unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn attn_plus_embed_equals_client_minus_manifests() {
        // Sanity: an `attn` slice + an `embed` slice cover the same
        // runtime bytes as the 2-tier `client` preset (modulo label
        // bookkeeping). Concatenating the two shouldn't miss any
        // deployment-critical part.
        let client = preset_parts("client").unwrap();
        let attn = preset_parts("attn").unwrap();
        let embed = preset_parts("embed").unwrap();
        let union: BTreeSet<Part> = attn.union(&embed).copied().collect();
        // Client includes: Attn, Norms, Embed, Tokenizer, Manifest, Labels.
        // attn ∪ embed includes: Attn, Norms, Manifest, Labels (attn) + Embed, Tokenizer, Labels (embed).
        // Both cover Attn+Norms+Embed+Tokenizer — the actual runtime bytes.
        for critical in [Part::Attn, Part::Norms, Part::Embed, Part::Tokenizer] {
            assert!(
                union.contains(&critical),
                "attn ∪ embed missing {critical:?}, which client has"
            );
            assert!(client.contains(&critical));
        }
    }

    #[test]
    fn effective_level_client_is_attention() {
        let parts: BTreeSet<Part> = [Part::Attn, Part::Norms, Part::Embed, Part::Tokenizer]
            .into_iter()
            .collect();
        let lvl = effective_level(&parts, larql_vindex::ExtractLevel::All);
        assert_eq!(lvl, larql_vindex::ExtractLevel::Attention);
    }

    #[test]
    fn effective_level_server_is_browse_without_attn() {
        // Server preset omits attn → effective level caps at Browse, even with FFN.
        let parts: BTreeSet<Part> = [Part::Gate, Part::Ffn, Part::DownMeta, Part::Tokenizer]
            .into_iter()
            .collect();
        let lvl = effective_level(&parts, larql_vindex::ExtractLevel::All);
        assert_eq!(lvl, larql_vindex::ExtractLevel::Browse);
    }

    #[test]
    fn effective_level_capped_by_source() {
        // Even a full parts set can't claim a higher tier than the source.
        let parts: BTreeSet<Part> = [
            Part::Attn, Part::Norms, Part::Embed, Part::Ffn, Part::Gate,
            Part::DownMeta, Part::LmHead, Part::Tokenizer,
        ]
        .into_iter()
        .collect();
        let lvl = effective_level(&parts, larql_vindex::ExtractLevel::Browse);
        assert_eq!(lvl, larql_vindex::ExtractLevel::Browse);
    }
}
