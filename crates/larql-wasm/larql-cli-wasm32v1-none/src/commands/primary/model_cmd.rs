//! `larql model pull` — download a HuggingFace *model* repo (safetensors +
//! tokenizer + config) into the local HF cache and print the snapshot
//! path. Companion to `larql pull` (which is vindex-only): bridges the gap
//! between "raw HF model on the hub" and `larql convert
//! safetensors-to-vindex`.
//!
//! Example:
//!   larql model pull Qwen/Qwen3-0.6B
//!   → ~/.cache/huggingface/hub/models--Qwen--Qwen3-0.6B/snapshots/<sha>/
//!
//! Then:
//!   larql convert safetensors-to-vindex \
//!     --input ~/.cache/.../snapshots/<sha>/ \
//!     --output ~/larql-vindex/qwen3-0.6b.vindex --f16

use clap::{Args, Subcommand};
use indicatif::{MultiProgress, ProgressBar, ProgressStyle};
#[allow(unused_imports)]
use alloc::{boxed::Box, string::{String, ToString}, vec::Vec, vec, format, borrow::{Cow, ToOwned}, rc::Rc, sync::Arc, collections::{BTreeMap, BTreeSet, VecDeque, BinaryHeap}};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use hashbrown::{HashMap, HashSet};
#[cfg(not(target_arch = "wasm32"))]
#[allow(unused_imports)]
use std::collections::{HashMap, HashSet};
#[cfg(target_arch = "wasm32")]
#[allow(unused_imports)]
use larql_wasm_math::FloatExt as _;
#[derive(Args)]
pub struct ModelArgs {
    #[command(subcommand)]
    command: ModelCommand,
}

#[derive(Subcommand)]
enum ModelCommand {
    /// Download a HuggingFace model repo (safetensors + tokenizer + config).
    Pull(ModelPullArgs),
}

#[derive(Args)]
pub struct ModelPullArgs {
    /// `hf://owner/name[@rev]` or `owner/name`.
    pub repo: String,

    /// Specific revision or branch.
    #[arg(long)]
    pub revision: Option<String>,
}

pub fn run(args: ModelArgs) -> Result<(), Box<dyn core::error::Error>> {
    match args.command {
        ModelCommand::Pull(p) => run_pull(p),
    }
}

fn run_pull(args: ModelPullArgs) -> Result<(), Box<dyn core::error::Error>> {
    let bare = args.repo.trim_start_matches("hf://");
    if !bare.contains('/') || bare.split('/').count() != 2 {
        return Err(format!(
            "model pull expects `hf://owner/name` or `owner/name`, got: {}",
            args.repo
        )
        .into());
    }
    let hf_path = match args.revision.as_deref() {
        Some(rev) => format!("hf://{bare}@{rev}"),
        None => format!("hf://{bare}"),
    };

    eprintln!("Pulling model {hf_path}...");

    let multi = MultiProgress::new();
    let style = ProgressStyle::with_template(
        "{spinner:.cyan} {msg:<48} [{bar:40.green/black}] {bytes:>10}/{total_bytes:<10} {bytes_per_sec}",
    )
    .unwrap()
    .progress_chars("=>-");

    let cached = larql_vindex::resolve_hf_model_with_progress(&hf_path, |filename| {
        let bar = multi.add(ProgressBar::new(0));
        bar.set_style(style.clone());
        bar.set_message(filename.to_string());
        BarProgress::new(bar)
    })?;

    eprintln!("\nDownloaded to: {}", cached.display());
    eprintln!("\nNext step:");
    eprintln!(
        "  larql convert safetensors-to-vindex --input {} --output <vindex-dir> --f16",
        cached.display()
    );

    Ok(())
}

/// `indicatif::ProgressBar` wrapper implementing hf-hub's `Progress`
/// trait. Same shape as the one in `pull_cmd.rs`, kept local to avoid a
/// cross-module dep on a private helper.
struct BarProgress {
    bar: ProgressBar,
}

impl BarProgress {
    fn new(bar: ProgressBar) -> Self {
        Self { bar }
    }
}

impl larql_vindex::DownloadProgress for BarProgress {
    fn init(&mut self, size: usize, _filename: &str) {
        self.bar.set_length(size as u64);
        self.bar.set_position(0);
    }

    fn update(&mut self, n: usize) {
        self.bar.inc(n as u64);
    }

    fn finish(&mut self) {
        self.bar.finish();
    }
}
