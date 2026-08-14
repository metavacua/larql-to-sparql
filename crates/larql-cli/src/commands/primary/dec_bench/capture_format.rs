//! On-disk residual capture pool for the DEC replay loadgen.
//!
//! A capture directory holds two files:
//!   * `manifest.json` — [`CaptureManifest`]: model identity, shape, and
//!     per-prompt step counts.
//!   * `residuals.bin` — f32 LE, layout `[prompt][step][layer][hidden]`,
//!     `steps` = the minimum step count across prompts (ragged tails are
//!     truncated at write time so every `(step, layer)` cell has a row from
//!     every prompt).
//!
//! Rows are captured **pre-normed** (see `ResidualCaptureSink`), i.e. exactly
//! the bytes the f32 wire carries and exactly the input Q8K quantisation
//! consumes — so replay needs no model weights and the pool is portable
//! across hosts (Mac capture → x86 replay).
//!
//! ## Optional routed-experts sidecars (additive, manifest stays v1)
//!
//! A pool captured with `--routing` additionally holds:
//!   * `raw.bin`     — raw post-attention residuals, same
//!     `[prompt][step][layer][hidden]` f32-LE layout as `residuals.bin`
//!     (the routed f32 endpoints want this; the server applies
//!     pre_experts_norm itself).
//!   * `normed.bin`  — pre-experts-normed residuals, same layout (input to
//!     the routed q8k frame quantisation).
//!   * `routing.bin` — `[prompt][step][layer][top_k × (u32 expert_id LE +
//!     f32 weight LE)]`. Fixed `top_k` records; layers with no MoE routing
//!     hold `top_k` sentinel entries ([`ROUTING_SENTINEL_EXPERT`], 0.0).
//!     Zero-weight pairs are stripped at write time (they cost nothing
//!     server-side; accounting for them would overcount): real pairs are
//!     compacted to the front of the record and the tail is sentinel-padded,
//!     so readers take the prefix before the first sentinel.
//!
//! The manifest gains an *optional* `routing` block
//! (`{ top_k, has_raw, has_normed }`). Absent block = walk-ffn-only pool:
//! the shipped 330M pool keeps opening and replaying unchanged
//! (`CaptureManifest` tolerates unknown fields, so old binaries also open
//! routed pools and simply ignore the sidecars).

use serde::{Deserialize, Serialize};
use std::io::Write;
use std::path::Path;

/// Bump when the binary layout changes. `open` rejects other versions.
/// The routing sidecars are additive — version stays 1.
pub const CAPTURE_VERSION: u32 = 1;

const MANIFEST_FILE: &str = "manifest.json";
const RESIDUALS_FILE: &str = "residuals.bin";
const RAW_FILE: &str = "raw.bin";
const NORMED_FILE: &str = "normed.bin";
const ROUTING_FILE: &str = "routing.bin";

/// Expert-id sentinel in `routing.bin`: marks an unused slot in a fixed
/// `top_k`-record (non-MoE layer, or a slot freed by zero-weight stripping).
pub const ROUTING_SENTINEL_EXPERT: u32 = u32::MAX;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PromptMeta {
    pub id: usize,
    pub text: String,
    /// Steps this prompt actually produced before truncation to `steps`.
    pub steps_captured: usize,
}

/// Optional manifest block describing the routed-experts sidecars.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingManifest {
    /// Fixed record length of `routing.bin` (entries per `[prompt][step][layer]`
    /// cell). From the model arch's `num_experts_per_token` at capture time.
    pub top_k: usize,
    /// `raw.bin` present.
    pub has_raw: bool,
    /// `normed.bin` present.
    pub has_normed: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CaptureManifest {
    pub version: u32,
    /// Model path / id the pool was captured from.
    pub model: String,
    pub hidden_size: usize,
    pub num_layers: usize,
    /// Replayable steps: min over prompts of steps captured.
    pub steps: usize,
    /// Always `"f32-le"` at version 1.
    pub dtype: String,
    pub prompts: Vec<PromptMeta>,
    pub created_unix: u64,
    /// Routed-experts sidecar block. `None` = walk-ffn-only pool (the
    /// pre-routing format — old pools deserialise to `None`, and the block
    /// is omitted from JSON on write so non-routed pools stay byte-stable).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub routing: Option<RoutingManifest>,
}

impl CaptureManifest {
    /// Expected byte length for `residuals.bin` — and for `raw.bin` /
    /// `normed.bin`, which share the exact layout.
    pub fn expected_bytes(&self) -> u64 {
        self.prompts.len() as u64
            * self.steps as u64
            * self.num_layers as u64
            * self.hidden_size as u64
            * 4
    }

    /// Expected `routing.bin` byte length (`None` when the pool has no
    /// routing block). 8 bytes per entry: u32 expert id + f32 weight.
    pub fn expected_routing_bytes(&self) -> Option<u64> {
        self.routing.as_ref().map(|r| {
            self.prompts.len() as u64
                * self.steps as u64
                * self.num_layers as u64
                * r.top_k as u64
                * 8
        })
    }
}

/// Routed-experts capture input for [`CapturePool::write_with_routing`].
/// All planes use the same `[prompt][step][layer]` nesting as `per_prompt`;
/// `routing[p][s][l]` holds up to `top_k` `(expert_id, weight)` pairs
/// (empty for layers with no MoE routing).
pub struct RoutingCapture {
    pub top_k: usize,
    /// Raw post-attention residual rows (length = hidden each).
    pub raw: Vec<Vec<Vec<Vec<f32>>>>,
    /// Pre-experts-normed residual rows (length = hidden each).
    pub normed: Vec<Vec<Vec<Vec<f32>>>>,
    /// Per-layer routing pairs, final post-renorm post-scale weights.
    pub routing: Vec<Vec<Vec<Vec<(u32, f32)>>>>,
}

/// An open, validated capture pool.
pub struct CapturePool {
    pub manifest: CaptureManifest,
    /// Raw `residuals.bin` contents; decoded to f32 on demand in [`Self::rows`].
    data: Vec<u8>,
    // The routed planes and their accessors are consumed by the Phase C
    // routed replay arm (Endpoint seam); until it lands, unit tests are the
    // only readers, so the bin target sees them as dead code.
    /// `raw.bin` contents when the pool has routing with `has_raw`.
    #[allow(dead_code)]
    raw: Option<Vec<u8>>,
    /// `normed.bin` contents when the pool has routing with `has_normed`.
    #[allow(dead_code)]
    normed: Option<Vec<u8>>,
    /// Decoded `routing.bin` (fixed `top_k` records incl. sentinels).
    #[allow(dead_code)]
    routing: Option<Vec<(u32, f32)>>,
}

impl CapturePool {
    /// Write a pool from per-prompt captured steps.
    ///
    /// `per_prompt[p][step][layer]` is the pre-normed residual for prompt `p`.
    /// Prompts with more steps than the shortest are truncated so the binary
    /// layout is rectangular; the pre-truncation count is kept in
    /// [`PromptMeta::steps_captured`].
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn write(
        dir: &Path,
        model: &str,
        hidden_size: usize,
        num_layers: usize,
        prompt_texts: &[String],
        per_prompt: &[Vec<Vec<Vec<f32>>>],
        created_unix: u64,
    ) -> Result<CaptureManifest, String> {
        Self::write_with_routing(
            dir,
            model,
            hidden_size,
            num_layers,
            prompt_texts,
            per_prompt,
            None,
            created_unix,
        )
    }

    /// [`Self::write`] plus the optional routed-experts sidecars. With
    /// `routing: None` this is exactly the walk-ffn-only write path.
    #[allow(clippy::too_many_arguments)]
    pub fn write_with_routing(
        dir: &Path,
        model: &str,
        hidden_size: usize,
        num_layers: usize,
        prompt_texts: &[String],
        per_prompt: &[Vec<Vec<Vec<f32>>>],
        routing: Option<&RoutingCapture>,
        created_unix: u64,
    ) -> Result<CaptureManifest, String> {
        if per_prompt.is_empty() {
            return Err("capture pool: no prompts captured".into());
        }
        if per_prompt.len() != prompt_texts.len() {
            return Err(format!(
                "capture pool: {} prompt texts but {} capture sinks",
                prompt_texts.len(),
                per_prompt.len()
            ));
        }
        let steps = per_prompt.iter().map(|s| s.len()).min().unwrap_or(0);
        if steps == 0 {
            return Err("capture pool: a prompt produced zero decode steps".into());
        }
        for (p, prompt_steps) in per_prompt.iter().enumerate() {
            for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
                if layers.len() != num_layers {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} has {} layers, expected {num_layers}",
                        layers.len()
                    ));
                }
                for (l, row) in layers.iter().enumerate() {
                    if row.len() != hidden_size {
                        return Err(format!(
                            "capture pool: prompt {p} step {s} layer {l} has {} floats, \
                             expected hidden {hidden_size}",
                            row.len()
                        ));
                    }
                }
            }
        }
        if let Some(rc) = routing {
            validate_routing_capture(rc, per_prompt.len(), steps, num_layers, hidden_size)?;
        }

        let manifest = CaptureManifest {
            version: CAPTURE_VERSION,
            model: model.to_string(),
            hidden_size,
            num_layers,
            steps,
            dtype: "f32-le".into(),
            prompts: prompt_texts
                .iter()
                .enumerate()
                .map(|(id, text)| PromptMeta {
                    id,
                    text: text.clone(),
                    steps_captured: per_prompt[id].len(),
                })
                .collect(),
            created_unix,
            routing: routing.map(|rc| RoutingManifest {
                top_k: rc.top_k,
                has_raw: true,
                has_normed: true,
            }),
        };

        std::fs::create_dir_all(dir).map_err(|e| format!("capture pool: mkdir: {e}"))?;
        write_plane(&dir.join(RESIDUALS_FILE), per_prompt, steps)?;
        if let Some(rc) = routing {
            write_plane(&dir.join(RAW_FILE), &rc.raw, steps)?;
            write_plane(&dir.join(NORMED_FILE), &rc.normed, steps)?;
            write_routing_file(&dir.join(ROUTING_FILE), rc, steps)?;
        }

        let manifest_json = serde_json::to_string_pretty(&manifest)
            .map_err(|e| format!("capture pool: manifest serialize: {e}"))?;
        std::fs::write(dir.join(MANIFEST_FILE), manifest_json)
            .map_err(|e| format!("capture pool: write manifest: {e}"))?;
        Ok(manifest)
    }

    /// Open and validate a pool directory.
    pub fn open(dir: &Path) -> Result<Self, String> {
        let manifest_path = dir.join(MANIFEST_FILE);
        let manifest_json = std::fs::read_to_string(&manifest_path)
            .map_err(|e| format!("capture pool: read {}: {e}", manifest_path.display()))?;
        let manifest: CaptureManifest = serde_json::from_str(&manifest_json)
            .map_err(|e| format!("capture pool: parse manifest: {e}"))?;
        if manifest.version != CAPTURE_VERSION {
            return Err(format!(
                "capture pool: version {} unsupported (expected {CAPTURE_VERSION})",
                manifest.version
            ));
        }
        let read_exact_len = |file: &str, expect: u64| -> Result<Vec<u8>, String> {
            let path = dir.join(file);
            let data = std::fs::read(&path)
                .map_err(|e| format!("capture pool: read {}: {e}", path.display()))?;
            if data.len() as u64 != expect {
                return Err(format!(
                    "capture pool: {file} is {} bytes, manifest expects {expect}",
                    data.len(),
                ));
            }
            Ok(data)
        };
        let data = read_exact_len(RESIDUALS_FILE, manifest.expected_bytes())?;

        let (raw, normed, routing) = match &manifest.routing {
            None => (None, None, None),
            Some(rb) => {
                if rb.top_k == 0 {
                    return Err("capture pool: routing block has top_k 0".into());
                }
                let raw = if rb.has_raw {
                    Some(read_exact_len(RAW_FILE, manifest.expected_bytes())?)
                } else {
                    None
                };
                let normed = if rb.has_normed {
                    Some(read_exact_len(NORMED_FILE, manifest.expected_bytes())?)
                } else {
                    None
                };
                let expect = manifest
                    .expected_routing_bytes()
                    .expect("routing block present");
                let bytes = read_exact_len(ROUTING_FILE, expect)?;
                let decoded: Vec<(u32, f32)> = bytes
                    .chunks_exact(8)
                    .map(|c| {
                        (
                            u32::from_le_bytes(c[..4].try_into().unwrap()),
                            f32::from_le_bytes(c[4..].try_into().unwrap()),
                        )
                    })
                    .collect();
                (raw, normed, Some(decoded))
            }
        };

        Ok(Self {
            manifest,
            data,
            raw,
            normed,
            routing,
        })
    }

    /// Number of prompts in the pool — the maximum replay batch size.
    pub fn num_prompts(&self) -> usize {
        self.manifest.prompts.len()
    }

    /// Whether the pool carries the routed-experts sidecars.
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn has_routing(&self) -> bool {
        self.routing.is_some()
    }

    /// Fixed `routing.bin` record length (`None` for walk-ffn-only pools).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn routing_top_k(&self) -> Option<usize> {
        self.manifest.routing.as_ref().map(|r| r.top_k)
    }

    /// Bounds-check a `(batch, step, layer)` cell request.
    fn check_cell(&self, batch: usize, step: usize, layer: usize) -> Result<(), String> {
        let m = &self.manifest;
        if batch == 0 || batch > m.prompts.len() {
            return Err(format!(
                "capture pool: batch {batch} out of range (pool has {} prompts)",
                m.prompts.len()
            ));
        }
        if step >= m.steps {
            return Err(format!(
                "capture pool: step {step} out of range (pool has {} steps)",
                m.steps
            ));
        }
        if layer >= m.num_layers {
            return Err(format!(
                "capture pool: layer {layer} out of range (pool has {} layers)",
                m.num_layers
            ));
        }
        Ok(())
    }

    /// Decode a `batch × hidden` row block for `(step, layer)` out of one
    /// `[prompt][step][layer][hidden]` plane.
    fn plane_rows(
        &self,
        plane: &[u8],
        batch: usize,
        step: usize,
        layer: usize,
    ) -> Result<Vec<f32>, String> {
        self.check_cell(batch, step, layer)?;
        let m = &self.manifest;
        let row_bytes = m.hidden_size * 4;
        let mut out = Vec::with_capacity(batch * m.hidden_size);
        for prompt in 0..batch {
            let row_idx = (prompt * m.steps + step) * m.num_layers + layer;
            let start = row_idx * row_bytes;
            out.extend(
                plane[start..start + row_bytes]
                    .chunks_exact(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );
        }
        Ok(out)
    }

    /// Build a `batch × hidden` contiguous row block for `(step, layer)`:
    /// row `i` is prompt `i`'s pre-normed residual at that step and layer.
    /// Distinct prompts per row keep MoE routing union realistic.
    pub fn rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        self.plane_rows(&self.data, batch, step, layer)
    }

    /// [`Self::rows`]-shaped block of **raw post-attention** residuals from
    /// `raw.bin`. Errors on walk-ffn-only pools (or `has_raw: false`).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn raw_rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        let plane = self
            .raw
            .as_ref()
            .ok_or("capture pool: raw.bin not present (walk-ffn-only pool)")?;
        self.plane_rows(plane, batch, step, layer)
    }

    /// [`Self::rows`]-shaped block of **pre-experts-normed** residuals from
    /// `normed.bin`. Errors on walk-ffn-only pools (or `has_normed: false`).
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn normed_rows(&self, batch: usize, step: usize, layer: usize) -> Result<Vec<f32>, String> {
        let plane = self
            .normed
            .as_ref()
            .ok_or("capture pool: normed.bin not present (walk-ffn-only pool)")?;
        self.plane_rows(plane, batch, step, layer)
    }

    /// Routing pairs for one `(prompt, step, layer)` cell:
    /// `Ok(Some(pairs))` — the non-sentinel `(expert_id, weight)` prefix
    /// (zero-weight pairs were stripped at write time), `Ok(None)` — layer
    /// carried no MoE routing (all-sentinel record), `Err` — walk-ffn-only
    /// pool or out-of-range cell.
    #[allow(dead_code)] // Phase C routed-replay consumer pending; tests pin it
    pub fn routing(
        &self,
        prompt: usize,
        step: usize,
        layer: usize,
    ) -> Result<Option<&[(u32, f32)]>, String> {
        let entries = self
            .routing
            .as_ref()
            .ok_or("capture pool: routing.bin not present (walk-ffn-only pool)")?;
        if prompt >= self.manifest.prompts.len() {
            return Err(format!(
                "capture pool: prompt {prompt} out of range (pool has {} prompts)",
                self.manifest.prompts.len()
            ));
        }
        // Reuse the shared step/layer bounds checks (batch=1 is always valid).
        self.check_cell(1, step, layer)?;
        let m = &self.manifest;
        let k = self
            .manifest
            .routing
            .as_ref()
            .expect("routing data implies manifest block")
            .top_k;
        let base = ((prompt * m.steps + step) * m.num_layers + layer) * k;
        let record = &entries[base..base + k];
        let n = record
            .iter()
            .position(|&(id, _)| id == ROUTING_SENTINEL_EXPERT)
            .unwrap_or(k);
        Ok(if n == 0 { None } else { Some(&record[..n]) })
    }
}

/// Validate one routed sidecar plane set against the pool shape.
fn validate_routing_capture(
    rc: &RoutingCapture,
    num_prompts: usize,
    steps: usize,
    num_layers: usize,
    hidden_size: usize,
) -> Result<(), String> {
    if rc.top_k == 0 {
        return Err("capture pool: routing capture has top_k 0".into());
    }
    for (name, plane) in [("raw", &rc.raw), ("normed", &rc.normed)] {
        if plane.len() != num_prompts {
            return Err(format!(
                "capture pool: routing {name} plane has {} prompts, expected {num_prompts}",
                plane.len()
            ));
        }
        for (p, prompt_steps) in plane.iter().enumerate() {
            if prompt_steps.len() < steps {
                return Err(format!(
                    "capture pool: routing {name} plane prompt {p} has {} steps, needs ≥ {steps}",
                    prompt_steps.len()
                ));
            }
            for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
                if layers.len() != num_layers {
                    return Err(format!(
                        "capture pool: routing {name} plane prompt {p} step {s} has {} layers, \
                         expected {num_layers}",
                        layers.len()
                    ));
                }
                for (l, row) in layers.iter().enumerate() {
                    if row.len() != hidden_size {
                        return Err(format!(
                            "capture pool: routing {name} plane prompt {p} step {s} layer {l} \
                             has {} floats, expected hidden {hidden_size}",
                            row.len()
                        ));
                    }
                }
            }
        }
    }
    if rc.routing.len() != num_prompts {
        return Err(format!(
            "capture pool: routing pairs have {} prompts, expected {num_prompts}",
            rc.routing.len()
        ));
    }
    for (p, prompt_steps) in rc.routing.iter().enumerate() {
        if prompt_steps.len() < steps {
            return Err(format!(
                "capture pool: routing pairs prompt {p} has {} steps, needs ≥ {steps}",
                prompt_steps.len()
            ));
        }
        for (s, layers) in prompt_steps.iter().take(steps).enumerate() {
            if layers.len() != num_layers {
                return Err(format!(
                    "capture pool: routing pairs prompt {p} step {s} has {} layers, \
                     expected {num_layers}",
                    layers.len()
                ));
            }
            for (l, pairs) in layers.iter().enumerate() {
                if pairs.len() > rc.top_k {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} layer {l} has {} routing pairs, \
                         top_k is {}",
                        pairs.len(),
                        rc.top_k
                    ));
                }
                if pairs.iter().any(|&(id, _)| id == ROUTING_SENTINEL_EXPERT) {
                    return Err(format!(
                        "capture pool: prompt {p} step {s} layer {l} uses reserved expert id \
                         {ROUTING_SENTINEL_EXPERT}"
                    ));
                }
            }
        }
    }
    Ok(())
}

/// Write one `[prompt][step][layer][hidden]` f32-LE plane, truncated to
/// `steps` per prompt. Shared by `residuals.bin` / `raw.bin` / `normed.bin`.
fn write_plane(path: &Path, per_prompt: &[Vec<Vec<Vec<f32>>>], steps: usize) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("capture pool: create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    for prompt_steps in per_prompt {
        for layers in prompt_steps.iter().take(steps) {
            for row in layers {
                for &v in row {
                    w.write_all(&v.to_le_bytes())
                        .map_err(|e| format!("capture pool: write: {e}"))?;
                }
            }
        }
    }
    w.flush().map_err(|e| format!("capture pool: flush: {e}"))
}

/// Write `routing.bin`: fixed `top_k` records; zero-weight pairs stripped
/// (compacted out) and the record sentinel-padded to length.
fn write_routing_file(path: &Path, rc: &RoutingCapture, steps: usize) -> Result<(), String> {
    let f = std::fs::File::create(path)
        .map_err(|e| format!("capture pool: create {}: {e}", path.display()))?;
    let mut w = std::io::BufWriter::new(f);
    let mut put = |id: u32, weight: f32| -> Result<(), String> {
        w.write_all(&id.to_le_bytes())
            .and_then(|()| w.write_all(&weight.to_le_bytes()))
            .map_err(|e| format!("capture pool: write: {e}"))
    };
    for prompt_steps in &rc.routing {
        for layers in prompt_steps.iter().take(steps) {
            for pairs in layers {
                let mut written = 0usize;
                for &(id, weight) in pairs {
                    if weight != 0.0 {
                        put(id, weight)?;
                        written += 1;
                    }
                }
                for _ in written..rc.top_k {
                    put(ROUTING_SENTINEL_EXPERT, 0.0)?;
                }
            }
        }
    }
    w.flush().map_err(|e| format!("capture pool: flush: {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn synthetic_capture(
        prompts: usize,
        steps: usize,
        layers: usize,
        hidden: usize,
    ) -> Vec<Vec<Vec<Vec<f32>>>> {
        (0..prompts)
            .map(|p| {
                (0..steps)
                    .map(|s| {
                        (0..layers)
                            .map(|l| {
                                (0..hidden)
                                    .map(|h| (p * 1000 + s * 100 + l * 10 + h) as f32)
                                    .collect()
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect()
    }

    fn write_synthetic(dir: &Path, prompts: usize, steps: usize, layers: usize, hidden: usize) {
        let cap = synthetic_capture(prompts, steps, layers, hidden);
        let texts: Vec<String> = (0..prompts).map(|i| format!("prompt {i}")).collect();
        CapturePool::write(dir, "test-model", hidden, layers, &texts, &cap, 12345).unwrap();
    }

    /// Routing fixture: layer 0 of every (prompt, step) routes to experts
    /// (p+s) % 3 and 3 with weights 0.75/0.25; layer 1+ is non-MoE (empty).
    fn synthetic_routing(
        prompts: usize,
        steps: usize,
        layers: usize,
        hidden: usize,
        top_k: usize,
    ) -> RoutingCapture {
        let raw = synthetic_capture(prompts, steps, layers, hidden);
        let normed: Vec<Vec<Vec<Vec<f32>>>> = raw
            .iter()
            .map(|p| {
                p.iter()
                    .map(|s| {
                        s.iter()
                            .map(|l| l.iter().map(|v| v * 0.5).collect())
                            .collect()
                    })
                    .collect()
            })
            .collect();
        let routing: Vec<Vec<Vec<Vec<(u32, f32)>>>> = (0..prompts)
            .map(|p| {
                (0..steps)
                    .map(|s| {
                        (0..layers)
                            .map(|l| {
                                if l == 0 {
                                    vec![(((p + s) % 3) as u32, 0.75f32), (3u32, 0.25f32)]
                                } else {
                                    Vec::new()
                                }
                            })
                            .collect()
                    })
                    .collect()
            })
            .collect();
        RoutingCapture {
            top_k,
            raw,
            normed,
            routing,
        }
    }

    fn write_synthetic_routed(
        dir: &Path,
        prompts: usize,
        steps: usize,
        layers: usize,
        hidden: usize,
        top_k: usize,
    ) -> CaptureManifest {
        let cap = synthetic_capture(prompts, steps, layers, hidden);
        let rc = synthetic_routing(prompts, steps, layers, hidden, top_k);
        let texts: Vec<String> = (0..prompts).map(|i| format!("prompt {i}")).collect();
        CapturePool::write_with_routing(
            dir,
            "test-model",
            hidden,
            layers,
            &texts,
            &cap,
            Some(&rc),
            7,
        )
        .unwrap()
    }

    #[test]
    fn manifest_round_trips_through_serde() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "m".into(),
            hidden_size: 8,
            num_layers: 2,
            steps: 3,
            dtype: "f32-le".into(),
            prompts: vec![PromptMeta {
                id: 0,
                text: "p".into(),
                steps_captured: 3,
            }],
            created_unix: 99,
            routing: None,
        };
        let json = serde_json::to_string(&m).unwrap();
        // Walk-ffn-only manifests must not grow a routing key — the shipped
        // 330M pool's manifest stays byte-stable in shape.
        assert!(!json.contains("routing"));
        let back: CaptureManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.version, m.version);
        assert_eq!(back.hidden_size, 8);
        assert_eq!(back.expected_bytes(), 3 * 2 * 8 * 4);
        assert!(back.routing.is_none());
        assert!(back.expected_routing_bytes().is_none());
    }

    #[test]
    fn manifest_routing_block_round_trips_and_sizes() {
        let m = CaptureManifest {
            version: CAPTURE_VERSION,
            model: "m".into(),
            hidden_size: 8,
            num_layers: 2,
            steps: 3,
            dtype: "f32-le".into(),
            prompts: vec![PromptMeta {
                id: 0,
                text: "p".into(),
                steps_captured: 3,
            }],
            created_unix: 99,
            routing: Some(RoutingManifest {
                top_k: 4,
                has_raw: true,
                has_normed: true,
            }),
        };
        let json = serde_json::to_string(&m).unwrap();
        let back: CaptureManifest = serde_json::from_str(&json).unwrap();
        let rb = back.routing.clone().expect("routing block survives serde");
        assert_eq!(rb.top_k, 4);
        assert!(rb.has_raw && rb.has_normed);
        assert_eq!(back.expected_routing_bytes(), Some(3 * 2 * 4 * 8));
    }

    #[test]
    fn write_open_rows_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 2, 2, 2, 4);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert_eq!(pool.num_prompts(), 2);

        // Row i of a batch must be prompt i at the same (step, layer).
        let rows = pool.rows(2, 1, 1).unwrap();
        assert_eq!(rows.len(), 2 * 4);
        let expect_p0: Vec<f32> = (0..4).map(|h| (100 + 10 + h) as f32).collect();
        let expect_p1: Vec<f32> = (0..4).map(|h| (1000 + 100 + 10 + h) as f32).collect();
        assert_eq!(&rows[..4], &expect_p0[..]);
        assert_eq!(&rows[4..], &expect_p1[..]);
    }

    #[test]
    fn walk_ffn_only_pool_has_no_routing_and_sidecar_accessors_err() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 2, 2, 2, 4);
        // No sidecar files on disk.
        assert!(!dir.path().join("raw.bin").exists());
        assert!(!dir.path().join("normed.bin").exists());
        assert!(!dir.path().join("routing.bin").exists());
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(!pool.has_routing());
        assert_eq!(pool.routing_top_k(), None);
        assert!(pool.rows(2, 0, 0).is_ok(), "dense replay path unaffected");
        assert!(pool.raw_rows(1, 0, 0).is_err());
        assert!(pool.normed_rows(1, 0, 0).is_err());
        assert!(pool.routing(0, 0, 0).is_err());
    }

    #[test]
    fn routed_write_open_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let manifest = write_synthetic_routed(dir.path(), 2, 2, 2, 4, 2);
        let rb = manifest.routing.as_ref().expect("routing block written");
        assert_eq!(rb.top_k, 2);
        assert!(rb.has_raw && rb.has_normed);

        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.has_routing());
        assert_eq!(pool.routing_top_k(), Some(2));

        // residuals.bin unchanged by the sidecars.
        let rows = pool.rows(2, 1, 1).unwrap();
        assert_eq!(rows.len(), 2 * 4);

        // raw plane = synthetic_capture values; normed plane = raw × 0.5.
        let raw = pool.raw_rows(2, 1, 1).unwrap();
        let expect_p0: Vec<f32> = (0..4).map(|h| (100 + 10 + h) as f32).collect();
        assert_eq!(&raw[..4], &expect_p0[..]);
        let normed = pool.normed_rows(2, 1, 1).unwrap();
        for (n, r) in normed.iter().zip(raw.iter()) {
            assert_eq!(*n, r * 0.5);
        }

        // Layer 0 routed: prompt 1 step 1 → expert (1+1)%3=2 @0.75, 3 @0.25.
        let pairs = pool.routing(1, 1, 0).unwrap().expect("MoE layer routes");
        assert_eq!(pairs, &[(2u32, 0.75f32), (3u32, 0.25f32)]);
        // Layer 1 non-MoE: all-sentinel record → None.
        assert_eq!(pool.routing(1, 1, 1).unwrap(), None);
    }

    #[test]
    fn routing_bounds_checks() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_routed(dir.path(), 2, 2, 2, 4, 2);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.routing(2, 0, 0).is_err(), "prompt out of range");
        assert!(pool.routing(0, 2, 0).is_err(), "step out of range");
        assert!(pool.routing(0, 0, 2).is_err(), "layer out of range");
        assert!(pool.raw_rows(3, 0, 0).is_err(), "batch > prompts");
        assert!(pool.normed_rows(1, 0, 2).is_err(), "layer out of range");
    }

    #[test]
    fn zero_weight_pairs_stripped_to_sentinel_at_write() {
        let dir = tempfile::tempdir().unwrap();
        let cap = synthetic_capture(1, 1, 1, 2);
        let mut rc = synthetic_routing(1, 1, 1, 2, 3);
        // One real pair sandwiched between zero-weight pairs.
        rc.routing[0][0][0] = vec![(5u32, 0.0f32), (7u32, 1.0f32), (9u32, 0.0f32)];
        let texts = vec!["a".into()];
        CapturePool::write_with_routing(dir.path(), "m", 2, 1, &texts, &cap, Some(&rc), 0).unwrap();

        // On-disk record: real pair compacted first, then sentinel padding.
        let bytes = std::fs::read(dir.path().join("routing.bin")).unwrap();
        assert_eq!(bytes.len(), 3 * 8, "fixed top_k record layout");
        assert_eq!(u32::from_le_bytes(bytes[0..4].try_into().unwrap()), 7);
        assert_eq!(f32::from_le_bytes(bytes[4..8].try_into().unwrap()), 1.0);
        assert_eq!(
            u32::from_le_bytes(bytes[8..12].try_into().unwrap()),
            ROUTING_SENTINEL_EXPERT
        );
        assert_eq!(
            u32::from_le_bytes(bytes[16..20].try_into().unwrap()),
            ROUTING_SENTINEL_EXPERT
        );

        let pool = CapturePool::open(dir.path()).unwrap();
        assert_eq!(
            pool.routing(0, 0, 0).unwrap(),
            Some(&[(7u32, 1.0f32)][..]),
            "reader sees only the surviving pair"
        );
    }

    #[test]
    fn routed_write_truncates_ragged_prompts_with_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let mut cap = synthetic_capture(2, 3, 1, 2);
        cap[1].truncate(1); // prompt 1 stopped early (EOS)
        let mut rc = synthetic_routing(2, 3, 1, 2, 2);
        rc.raw[1].truncate(1);
        rc.normed[1].truncate(1);
        rc.routing[1].truncate(1);
        let texts = vec!["a".into(), "b".into()];
        let m = CapturePool::write_with_routing(dir.path(), "m", 2, 1, &texts, &cap, Some(&rc), 0)
            .unwrap();
        assert_eq!(m.steps, 1);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.raw_rows(2, 0, 0).is_ok());
        assert!(pool.raw_rows(2, 1, 0).is_err(), "step beyond min must fail");
        assert!(pool.routing(0, 0, 0).unwrap().is_some());
    }

    #[test]
    fn routed_write_rejects_bad_routing_shapes() {
        let dir = tempfile::tempdir().unwrap();
        let cap = synthetic_capture(2, 2, 2, 4);
        let texts = vec!["a".into(), "b".into()];
        let write = |rc: &RoutingCapture| {
            CapturePool::write_with_routing(dir.path(), "m", 4, 2, &texts, &cap, Some(rc), 0)
        };

        // top_k 0.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.top_k = 0;
        assert!(write(&rc).is_err());

        // More pairs than top_k.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.routing[0][0][0] = vec![(0, 0.5), (1, 0.3), (2, 0.2)];
        assert!(write(&rc).is_err());

        // Reserved sentinel expert id in the input.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.routing[0][0][0] = vec![(ROUTING_SENTINEL_EXPERT, 0.5)];
        assert!(write(&rc).is_err());

        // Wrong prompt count on a plane.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.raw.pop();
        assert!(write(&rc).is_err());
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.routing.pop();
        assert!(write(&rc).is_err());

        // Too few steps on a plane.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.normed[0].truncate(1);
        assert!(write(&rc).is_err());
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.routing[0].truncate(1);
        assert!(write(&rc).is_err());

        // Wrong layer count / hidden size on a plane row.
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.raw[0][0].pop();
        assert!(write(&rc).is_err());
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.normed[0][0][0].pop();
        assert!(write(&rc).is_err());
        let mut rc = synthetic_routing(2, 2, 2, 4, 2);
        rc.routing[0][0].pop();
        assert!(write(&rc).is_err());
    }

    #[test]
    fn write_truncates_ragged_prompts_to_min_steps() {
        let dir = tempfile::tempdir().unwrap();
        let mut cap = synthetic_capture(2, 3, 1, 2);
        cap[1].truncate(1); // prompt 1 stopped early (EOS)
        let texts = vec!["a".into(), "b".into()];
        let m = CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).unwrap();
        assert_eq!(m.steps, 1);
        assert_eq!(m.prompts[0].steps_captured, 3);
        assert_eq!(m.prompts[1].steps_captured, 1);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.rows(2, 0, 0).is_ok());
        assert!(pool.rows(2, 1, 0).is_err(), "step beyond min must fail");
    }

    #[test]
    fn write_rejects_empty_and_mismatched_inputs() {
        let dir = tempfile::tempdir().unwrap();
        let texts = vec!["a".into()];
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &[], 0).is_err());

        // Wrong layer count.
        let cap = synthetic_capture(1, 1, 2, 2);
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

        // Wrong hidden size.
        let cap = synthetic_capture(1, 1, 1, 3);
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());

        // Zero steps.
        let cap: Vec<Vec<Vec<Vec<f32>>>> = vec![vec![]];
        assert!(CapturePool::write(dir.path(), "m", 2, 1, &texts, &cap, 0).is_err());
    }

    #[test]
    fn open_rejects_truncated_bin_and_bad_version() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 1, 1, 1, 4);

        // Truncate residuals.bin → open must fail on size mismatch.
        let bin = dir.path().join("residuals.bin");
        let data = std::fs::read(&bin).unwrap();
        std::fs::write(&bin, &data[..data.len() - 4]).unwrap();
        assert!(CapturePool::open(dir.path()).is_err());
        std::fs::write(&bin, &data).unwrap();
        assert!(CapturePool::open(dir.path()).is_ok());

        // Corrupt the version → open must fail.
        let mf = dir.path().join("manifest.json");
        let json = std::fs::read_to_string(&mf).unwrap();
        std::fs::write(&mf, json.replace("\"version\": 1", "\"version\": 999")).unwrap();
        assert!(CapturePool::open(dir.path()).is_err());
    }

    #[test]
    fn open_rejects_truncated_or_missing_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_routed(dir.path(), 1, 1, 2, 4, 2);

        for file in ["raw.bin", "normed.bin", "routing.bin"] {
            let path = dir.path().join(file);
            let data = std::fs::read(&path).unwrap();
            // Truncated sidecar → exact-byte-length validation must reject.
            std::fs::write(&path, &data[..data.len() - 4]).unwrap();
            assert!(
                CapturePool::open(dir.path()).is_err(),
                "truncated {file} must be rejected"
            );
            // Missing sidecar (manifest says present) → rejected too.
            std::fs::remove_file(&path).unwrap();
            assert!(
                CapturePool::open(dir.path()).is_err(),
                "missing {file} must be rejected"
            );
            std::fs::write(&path, &data).unwrap();
            assert!(CapturePool::open(dir.path()).is_ok());
        }

        // top_k 0 in the manifest block is impossible from write — corrupt it.
        let mf = dir.path().join("manifest.json");
        let json = std::fs::read_to_string(&mf).unwrap();
        std::fs::write(&mf, json.replace("\"top_k\": 2", "\"top_k\": 0")).unwrap();
        assert!(CapturePool::open(dir.path()).is_err());
    }

    #[test]
    fn open_tolerates_absent_raw_or_normed_when_flagged_off() {
        // has_raw/has_normed false → those planes are optional; routing.bin
        // still required and served.
        let dir = tempfile::tempdir().unwrap();
        write_synthetic_routed(dir.path(), 1, 1, 1, 2, 2);
        std::fs::remove_file(dir.path().join("raw.bin")).unwrap();
        std::fs::remove_file(dir.path().join("normed.bin")).unwrap();
        let mf = dir.path().join("manifest.json");
        let json = std::fs::read_to_string(&mf).unwrap();
        let json = json
            .replace("\"has_raw\": true", "\"has_raw\": false")
            .replace("\"has_normed\": true", "\"has_normed\": false");
        std::fs::write(&mf, json).unwrap();

        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.has_routing());
        assert!(pool.raw_rows(1, 0, 0).is_err());
        assert!(pool.normed_rows(1, 0, 0).is_err());
        assert!(pool.routing(0, 0, 0).unwrap().is_some());
    }

    #[test]
    fn rows_bounds_checks() {
        let dir = tempfile::tempdir().unwrap();
        write_synthetic(dir.path(), 2, 2, 2, 2);
        let pool = CapturePool::open(dir.path()).unwrap();
        assert!(pool.rows(0, 0, 0).is_err());
        assert!(pool.rows(3, 0, 0).is_err(), "batch > prompts");
        assert!(pool.rows(1, 2, 0).is_err(), "step out of range");
        assert!(pool.rows(1, 0, 2).is_err(), "layer out of range");
    }
}
