//! Workspace certification status table.
//!
//! Reads `[package.metadata.wasm-cert]` from each crate's `Cargo.toml` via
//! `cargo_metadata` and renders a Markdown table.  Writes to
//! `$GITHUB_STEP_SUMMARY` when set (GitHub Actions job summary).
//!
//! Metadata schema (`[package.metadata.wasm-cert]`):
//!   current-tier  = "0" | "1" | "2" | "3"
//!   target-tier   = "3"
//!   cov-threshold = 80
//! Legacy fallback: `claimed-level = N` is read if `current-tier` is absent.

use anyhow::Result;
use cargo_metadata::{Metadata, MetadataCommand};

use crate::rules::WasmTier;

/// Certification manifest read from `[package.metadata.wasm-cert]`.
#[derive(Debug, Default)]
pub struct WasmCert {
    pub current_tier: WasmTier,
    pub target_tier: WasmTier,
    pub cov_threshold: u8,
    pub notes: String,
}

impl WasmCert {
    pub fn from_metadata(meta: &serde_json::Map<String, serde_json::Value>) -> Option<Self> {
        let cert = meta.get("wasm-cert")?;
        let current_tier = cert
            .get("current-tier")
            .and_then(|v| v.as_str())
            .map(WasmTier::from_str)
            .or_else(|| {
                cert.get("claimed-level")
                    .and_then(|v| v.as_u64())
                    .map(|n| WasmTier::from_u8(n as u8))
            })
            .unwrap_or(WasmTier::Level0);
        let target_tier = cert
            .get("target-tier")
            .and_then(|v| v.as_str())
            .map(WasmTier::from_str)
            .unwrap_or(current_tier);
        Some(WasmCert {
            current_tier,
            target_tier,
            cov_threshold: cert
                .get("cov-threshold")
                .and_then(|v| v.as_u64())
                .unwrap_or(0) as u8,
            notes: cert
                .get("notes")
                .and_then(|v| v.as_str())
                .unwrap_or("")
                .to_owned(),
        })
    }
}

pub fn workspace_meta() -> Result<Metadata> {
    Ok(MetadataCommand::new().exec()?)
}

pub fn run(json: bool) -> Result<()> {
    let meta = workspace_meta()?;
    let mut rows: Vec<(String, WasmCert)> = vec![];

    for pkg in &meta.packages {
        if !meta.workspace_members.contains(&pkg.id) {
            continue;
        }
        let cert = pkg
            .metadata
            .as_object()
            .and_then(|m| WasmCert::from_metadata(m))
            .unwrap_or_default();
        rows.push((pkg.name.clone(), cert));
    }

    rows.sort_by(|a, b| a.0.cmp(&b.0));

    let output = if json {
        render_json(&rows)
    } else {
        render_markdown(&rows)
    };

    println!("{output}");

    if let Ok(summary_path) = std::env::var("GITHUB_STEP_SUMMARY") {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(summary_path)?;
        writeln!(f, "{output}")?;
    }

    Ok(())
}

fn render_markdown(rows: &[(String, WasmCert)]) -> String {
    let mut out = String::new();
    out.push_str("## wasm32 Certification Status\n\n");
    out.push_str("| Crate | Tier | Target | Cov% | Notes |\n");
    out.push_str("|-------|------|--------|------|-------|\n");
    for (name, cert) in rows {
        let tier_str = match cert.current_tier {
            WasmTier::Level0 => "0 ⚠ uncertified".to_owned(),
            t => t.to_string(),
        };
        let cov_str = if cert.cov_threshold > 0 {
            format!("≥{}%", cert.cov_threshold)
        } else {
            "—".to_owned()
        };
        out.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            name,
            tier_str,
            cert.target_tier,
            cov_str,
            cert.notes,
        ));
    }
    out
}

fn render_json(rows: &[(String, WasmCert)]) -> String {
    let arr: Vec<serde_json::Value> = rows
        .iter()
        .map(|(name, cert)| {
            serde_json::json!({
                "crate": name,
                "current_tier": cert.current_tier.to_string(),
                "target_tier": cert.target_tier.to_string(),
                "cov_threshold": cert.cov_threshold,
                "notes": cert.notes,
            })
        })
        .collect();
    serde_json::to_string_pretty(&arr).unwrap_or_default()
}
