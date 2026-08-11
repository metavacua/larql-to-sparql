//! Portable Shannon-scoring math core, split out of [`super::shannon_cmd`]
//! (round-1 wasm32 gating survey for `commands/primary`) so it keeps
//! compiling for `wasm32v1-none` even though `shannon_cmd` itself is
//! native-only (clap Args, `std::fs`/`std::time::Instant`/`indicatif`,
//! `tokenizers::Tokenizer`-by-value CLI driver code).
//!
//! Everything here is pure numeric/byte-level logic: logit-lens
//! projection, bits/KL accounting, and the model-driven arithmetic coder
//! (`BitWriter`/`BitReader`/`ArithmeticEncoder`/`ArithmeticDecoder` +
//! `ShannonFile`/`VindexShannonBlock` framing). `shannon_cmd.rs` re-exports
//! this module wholesale (`pub(crate) use super::shannon_math::*;`) so its
//! own native drivers see these names exactly as before the split, and
//! `shannon_trace/dump.rs`'s native `run_layer_dump` keeps reaching
//! `forward_hidden_all_layers` via `shannon_cmd::forward_hidden_all_layers`
//! unchanged.
//!
//! No `#[forbid(unsafe_code)]` here (matches `shannon_cmd.rs`'s own
//! posture, see `commands/primary/mod.rs`): `logits_for_row` uses
//! `ndarray::s![...]`, whose macro expansion hides its own internal
//! `unsafe` block that `forbid` cannot see through (pattern 18,
//! CI-confirmed via workflow run 31464601274).

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;
// `format!` isn't in `alloc_prelude` (only value types are) and this
// crate's `extern crate alloc;` (main.rs) isn't `#[macro_use]`, unlike
// every other already-gated crate in this workspace -- import the macro
// explicitly here rather than depend on that (see report: flagged as a
// crate-root gap out of this group's scope).
#[cfg(target_arch = "wasm32")]
use alloc::format;

use crate::collections::HashMap;
use larql_inference::attention::SharedKV;
use larql_inference::ffn::{expert_weights_resolvable, ExpertWeightFfn, FfnBackend, WeightFfn};
use larql_inference::forward::{apply_norm, dot_proj};
use larql_inference::ModelWeights;
use ndarray::{s, Array2};

// ── Constants shared with the native driver in `shannon_cmd.rs` ─────────

pub(crate) const LN_2: f64 = core::f64::consts::LN_2;
pub(crate) const DEFAULT_CONTEXT: usize = 512;
pub(crate) const DEFAULT_STRIDE: usize = 256;

// Arithmetic coding must rebuild the exact same integer frequency table when
// decoding. The vindex/Metal path is fast but can produce tiny cross-run float
// drift, so keep this comfortably above Gemma's 262K vocab without making the
// table hypersensitive to low-order logit differences.
pub(crate) const FREQ_TOTAL: u32 = 1 << 19;
pub(crate) const CODE_BITS: u32 = 32;
pub(crate) const TOP_VALUE: u64 = (1u64 << CODE_BITS) - 1;
pub(crate) const FIRST_QTR: u64 = TOP_VALUE / 4 + 1;
pub(crate) const HALF: u64 = FIRST_QTR * 2;
pub(crate) const THIRD_QTR: u64 = FIRST_QTR * 3;
pub(crate) const VINDEX_BLOCK_TARGET_TOKENS: usize = 512;

pub(crate) fn validate_window(context: usize, stride: usize) -> Result<(), Box<dyn core::error::Error>> {
    if context < 2 {
        return Err("--context must be at least 2 for scoring".into());
    }
    if stride == 0 {
        return Err("--stride must be at least 1".into());
    }
    if stride >= context {
        return Err("--stride must be smaller than --context so every target has a prefix".into());
    }
    Ok(())
}

pub(crate) fn ensure_token_prefix(
    prefix: &[u32],
    full: &[u32],
) -> Result<(), Box<dyn core::error::Error>> {
    if full.len() < prefix.len() || full[..prefix.len()] != *prefix {
        return Err(
            "answer did not tokenize as a suffix of prefix+answer; add explicit boundary whitespace"
                .into(),
        );
    }
    Ok(())
}

#[derive(Debug, Default)]
pub(crate) struct ScoreSummary {
    pub(crate) total_bits: f64,
    pub(crate) token_bits: Vec<f64>,
}

impl ScoreSummary {
    pub(crate) fn bits_per_token(&self) -> f64 {
        self.total_bits / self.token_bits.len().max(1) as f64
    }
}

/// Pick the FFN backend that matches this model's architecture.
///
/// The scorer used to hardcode [`WeightFfn`], which resolves the *dense*
/// `ffn_{gate,up,down}_key`. A mixture-of-experts model has no such tensors,
/// so scoring one panicked with a misleading "this is a `--compact` vindex"
/// hint — the tensors were not missing, they never existed for that
/// architecture. That made `shannon verify` unusable on every MoE model, which
/// is the entire model class the K3 ladder is built on (GPT-OSS, Kimi Linear,
/// K3). See `docs/k3-funnel.md` §4.7.
///
/// `expert_weights_resolvable` gates on the weights actually being present, so
/// a MoE architecture whose experts are packed rather than per-expert f32 still
/// falls through to the dense path and fails with its own error rather than
/// half-resolving here.
pub(crate) fn score_ffn(weights: &ModelWeights) -> Box<dyn FfnBackend + '_> {
    if weights.arch.is_moe() && expert_weights_resolvable(weights, 0) {
        Box::new(ExpertWeightFfn { weights })
    } else {
        Box::new(WeightFfn { weights })
    }
}

pub(crate) fn forward_hidden(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Array2<f32>, Box<dyn core::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = score_ffn(weights);
    let mut h = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h, token_ids);
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::default();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &*ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
    }
    Ok(h)
}

pub(crate) fn forward_hidden_all_layers(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<Array2<f32>>, Box<dyn core::error::Error>> {
    if token_ids.is_empty() {
        return Err("empty token window".into());
    }
    let ffn = score_ffn(weights);
    let h0 = larql_inference::forward::embed_tokens_pub(weights, token_ids);
    let ple_inputs =
        larql_inference::forward::ple::precompute_per_layer_inputs(weights, &h0, token_ids);
    let mut captures: Vec<Array2<f32>> = Vec::with_capacity(weights.num_layers + 1);
    captures.push(h0.clone());
    let mut h = h0;
    let mut kv_cache: HashMap<usize, SharedKV> = HashMap::default();
    for layer in 0..weights.num_layers {
        let shared_kv = weights
            .arch
            .kv_shared_source_layer(layer)
            .and_then(|src| kv_cache.get(&src));
        if let Some((h_new, _, kv_out)) = larql_inference::forward::run_layer_with_ffn(
            larql_inference::WeightsView::dense(weights),
            &h,
            layer,
            &*ffn,
            false,
            ple_inputs.get(layer),
            shared_kv,
        ) {
            h = h_new;
            if let Some(kv) = kv_out {
                kv_cache.insert(layer, kv);
            }
        }
        captures.push(h.clone());
    }
    Ok(captures)
}

pub(crate) fn final_norm(weights: &ModelWeights, h: &Array2<f32>) -> Array2<f32> {
    apply_norm(
        weights,
        h,
        weights.arch.final_norm_key(),
        weights.arch.norm_weight_offset(),
    )
}

pub(crate) fn logits_for_last_token(
    weights: &ModelWeights,
    token_ids: &[u32],
) -> Result<Vec<f32>, Box<dyn core::error::Error>> {
    let hidden = forward_hidden(weights, token_ids)?;
    let hidden = final_norm(weights, &hidden);
    logits_for_row(weights, &hidden, hidden.shape()[0] - 1)
}

pub(crate) fn logits_for_row(
    weights: &ModelWeights,
    final_hidden: &Array2<f32>,
    row_idx: usize,
) -> Result<Vec<f32>, Box<dyn core::error::Error>> {
    if row_idx >= final_hidden.shape()[0] {
        return Err("logit row out of range".into());
    }
    let row = final_hidden.slice(s![row_idx..row_idx + 1, ..]);
    let raw = dot_proj(&row, &weights.lm_head);
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    Ok(raw
        .row(0)
        .iter()
        .map(|&v| {
            let mut logit = v * inv_scale;
            if let Some(cap) = final_softcap {
                logit = (logit / cap).tanh() * cap;
            }
            logit
        })
        .collect())
}

pub(crate) fn bits_for_target(logits: &[f32], target: u32) -> Result<f64, Box<dyn core::error::Error>> {
    let target = target as usize;
    if target >= logits.len() {
        return Err(format!("target token {target} out of vocab").into());
    }
    let max_logit = finite_max(logits)?;
    let exp_sum: f64 = logits
        .iter()
        .filter(|v| v.is_finite())
        .map(|&v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - logits[target] as f64) / LN_2)
}

pub(crate) fn bits_for_raw_row(
    weights: &ModelWeights,
    row: ndarray::ArrayView1<'_, f32>,
    target: u32,
) -> Result<f64, Box<dyn core::error::Error>> {
    let target = target as usize;
    if target >= row.len() {
        return Err(format!("target token {target} out of vocab").into());
    }

    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };

    let max_logit = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(transform)
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".to_string())?;

    let exp_sum: f64 = row
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((transform(v) - max_logit) as f64).exp())
        .sum();
    let target_logit = transform(row[target]);
    let logsumexp = max_logit as f64 + exp_sum.ln();
    Ok((logsumexp - target_logit as f64) / LN_2)
}

pub(crate) fn prob_for_target(logits: &[f32], target: u32) -> Result<f64, Box<dyn core::error::Error>> {
    Ok(2.0_f64.powf(-bits_for_target(logits, target)?))
}

/// Apply per-arch logit scaling/softcap and return natural-log probabilities
/// over the full vocabulary for one position. Length matches the input row.
pub(crate) fn compute_log_probs_row(
    weights: &ModelWeights,
    row: ndarray::ArrayView1<'_, f32>,
) -> Vec<f32> {
    let inv_scale = 1.0 / weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let transform = |v: f32| {
        if !v.is_finite() {
            return v;
        }
        let mut logit = v * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        logit
    };
    let scaled: Vec<f32> = row.iter().copied().map(transform).collect();
    let max_logit = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .map(|v| ((v - max_logit) as f64).exp())
        .sum();
    let logsumexp = (max_logit as f64) + exp_sum.ln();
    scaled
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v as f64) - logsumexp) as f32
            } else {
                f32::NEG_INFINITY
            }
        })
        .collect()
}

pub(crate) fn layer_label(idx: usize) -> String {
    if idx == 0 {
        "embed".to_string()
    } else {
        format!("L{:02}", idx - 1)
    }
}

pub(crate) fn finite_max(values: &[f32]) -> Result<f32, Box<dyn core::error::Error>> {
    values
        .iter()
        .copied()
        .filter(|v| v.is_finite())
        .fold(None, |acc: Option<f32>, v| {
            Some(acc.map_or(v, |m| m.max(v)))
        })
        .ok_or_else(|| "all logits were non-finite".into())
}

pub(crate) fn quantized_counts(logits: &[f32]) -> Result<Vec<u32>, Box<dyn core::error::Error>> {
    if logits.len() >= FREQ_TOTAL as usize {
        return Err("vocab is too large for arithmetic coder frequency total".into());
    }
    let max_logit = finite_max(logits)?;
    let exp_values: Vec<f64> = logits
        .iter()
        .map(|&v| {
            if v.is_finite() {
                ((v - max_logit) as f64).exp()
            } else {
                0.0
            }
        })
        .collect();
    let exp_sum: f64 = exp_values.iter().sum();
    if exp_sum <= 0.0 {
        return Err("invalid probability distribution".into());
    }
    let spare = FREQ_TOTAL as usize - logits.len();
    let mut max_idx = 0usize;
    let mut max_exp = f64::NEG_INFINITY;
    let mut sum = 0u32;
    let mut counts = Vec::with_capacity(logits.len());
    for (i, exp_v) in exp_values.iter().copied().enumerate() {
        if exp_v > max_exp {
            max_exp = exp_v;
            max_idx = i;
        }
        let count = 1 + (exp_v / exp_sum * spare as f64).floor() as u32;
        sum = sum.saturating_add(count);
        counts.push(count);
    }
    if sum > FREQ_TOTAL {
        return Err("frequency quantization overflowed".into());
    }
    counts[max_idx] += FREQ_TOTAL - sum;
    Ok(counts)
}

pub(crate) fn interval_for_symbol(
    counts: &[u32],
    symbol: u32,
) -> Result<(u32, u32), Box<dyn core::error::Error>> {
    let symbol = symbol as usize;
    if symbol >= counts.len() {
        return Err(format!("symbol {symbol} out of frequency table").into());
    }
    let low: u32 = counts[..symbol].iter().sum();
    let high = low + counts[symbol];
    Ok((low, high))
}

pub(crate) fn symbol_for_value(
    counts: &[u32],
    value: u32,
) -> Result<(u32, u32, u32), Box<dyn core::error::Error>> {
    let mut low = 0u32;
    for (symbol, &count) in counts.iter().enumerate() {
        let high = low + count;
        if value < high {
            return Ok((symbol as u32, low, high));
        }
        low = high;
    }
    Err("arithmetic decoder value outside frequency table".into())
}

pub(crate) struct BitWriter {
    bytes: Vec<u8>,
    current: u8,
    used: u8,
}

impl BitWriter {
    pub(crate) fn new() -> Self {
        Self {
            bytes: Vec::new(),
            current: 0,
            used: 0,
        }
    }

    pub(crate) fn write(&mut self, bit: bool) {
        self.current = (self.current << 1) | u8::from(bit);
        self.used += 1;
        if self.used == 8 {
            self.bytes.push(self.current);
            self.current = 0;
            self.used = 0;
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        if self.used > 0 {
            self.current <<= 8 - self.used;
            self.bytes.push(self.current);
        }
        self.bytes
    }
}

pub(crate) struct BitReader<'a> {
    bytes: &'a [u8],
    byte_idx: usize,
    bit_idx: u8,
}

impl<'a> BitReader<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        Self {
            bytes,
            byte_idx: 0,
            bit_idx: 0,
        }
    }

    pub(crate) fn read(&mut self) -> bool {
        if self.byte_idx >= self.bytes.len() {
            return false;
        }
        let bit = (self.bytes[self.byte_idx] & (0x80 >> self.bit_idx)) != 0;
        self.bit_idx += 1;
        if self.bit_idx == 8 {
            self.bit_idx = 0;
            self.byte_idx += 1;
        }
        bit
    }
}

pub(crate) struct ArithmeticEncoder {
    low: u64,
    high: u64,
    pending: u64,
    bits: BitWriter,
}

impl ArithmeticEncoder {
    pub(crate) fn new() -> Self {
        Self {
            low: 0,
            high: TOP_VALUE,
            pending: 0,
            bits: BitWriter::new(),
        }
    }

    pub(crate) fn encode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                self.output_bit_plus_follow(false);
            } else if self.low >= HALF {
                self.output_bit_plus_follow(true);
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.pending += 1;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
        }
    }

    pub(crate) fn finish(mut self) -> Vec<u8> {
        self.pending += 1;
        if self.low < FIRST_QTR {
            self.output_bit_plus_follow(false);
        } else {
            self.output_bit_plus_follow(true);
        }
        self.bits.finish()
    }

    fn output_bit_plus_follow(&mut self, bit: bool) {
        self.bits.write(bit);
        for _ in 0..self.pending {
            self.bits.write(!bit);
        }
        self.pending = 0;
    }
}

pub(crate) struct ArithmeticDecoder<'a> {
    low: u64,
    high: u64,
    code: u64,
    bits: BitReader<'a>,
}

impl<'a> ArithmeticDecoder<'a> {
    pub(crate) fn new(bytes: &'a [u8]) -> Self {
        let mut bits = BitReader::new(bytes);
        let mut code = 0u64;
        for _ in 0..CODE_BITS {
            code = code * 2 + u64::from(bits.read());
        }
        Self {
            low: 0,
            high: TOP_VALUE,
            code,
            bits,
        }
    }

    pub(crate) fn scaled_value(&self, total: u32) -> u32 {
        let range = self.high - self.low + 1;
        ((((self.code - self.low + 1) * total as u64 - 1) / range) as u32).min(total - 1)
    }

    pub(crate) fn decode(&mut self, cum_low: u32, cum_high: u32, total: u32) {
        let range = self.high - self.low + 1;
        self.high = self.low + (range * cum_high as u64) / total as u64 - 1;
        self.low += (range * cum_low as u64) / total as u64;

        loop {
            if self.high < HALF {
                // nothing
            } else if self.low >= HALF {
                self.code -= HALF;
                self.low -= HALF;
                self.high -= HALF;
            } else if self.low >= FIRST_QTR && self.high < THIRD_QTR {
                self.code -= FIRST_QTR;
                self.low -= FIRST_QTR;
                self.high -= FIRST_QTR;
            } else {
                break;
            }
            self.low *= 2;
            self.high = self.high * 2 + 1;
            self.code = self.code * 2 + u64::from(self.bits.read());
        }
    }
}

pub(crate) struct ShannonFile {
    pub(crate) context: u32,
    pub(crate) first_token: u32,
    pub(crate) target_tokens: u64,
    pub(crate) original_bytes: u64,
    pub(crate) payload: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct VindexShannonBlock {
    pub(crate) first_token: u32,
    pub(crate) target_tokens: u64,
    pub(crate) payload: Vec<u8>,
}

impl ShannonFile {
    pub(crate) fn to_bytes(&self) -> Vec<u8> {
        let mut out = Vec::with_capacity(36 + self.payload.len());
        out.extend_from_slice(b"LSC1");
        out.extend_from_slice(&self.context.to_le_bytes());
        out.extend_from_slice(&self.first_token.to_le_bytes());
        out.extend_from_slice(&self.target_tokens.to_le_bytes());
        out.extend_from_slice(&self.original_bytes.to_le_bytes());
        out.extend_from_slice(&(self.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.payload);
        out
    }

    pub(crate) fn from_bytes(bytes: &[u8]) -> Result<Self, Box<dyn core::error::Error>> {
        if bytes.len() < 36 || &bytes[..4] != b"LSC1" {
            return Err("not a LARQL Shannon compressed file".into());
        }
        let context = u32::from_le_bytes(bytes[4..8].try_into()?);
        let first_token = u32::from_le_bytes(bytes[8..12].try_into()?);
        let target_tokens = u64::from_le_bytes(bytes[12..20].try_into()?);
        let original_bytes = u64::from_le_bytes(bytes[20..28].try_into()?);
        let payload_len = u64::from_le_bytes(bytes[28..36].try_into()?) as usize;
        if bytes.len() != 36 + payload_len {
            return Err("compressed file payload length mismatch".into());
        }
        Ok(Self {
            context,
            first_token,
            target_tokens,
            original_bytes,
            payload: bytes[36..].to_vec(),
        })
    }
}

pub(crate) fn encode_vindex_blocks(blocks: &[VindexShannonBlock]) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(b"LSB1");
    out.extend_from_slice(&(blocks.len() as u32).to_le_bytes());
    for block in blocks {
        out.extend_from_slice(&block.first_token.to_le_bytes());
        out.extend_from_slice(&block.target_tokens.to_le_bytes());
        out.extend_from_slice(&(block.payload.len() as u64).to_le_bytes());
        out.extend_from_slice(&block.payload);
    }
    out
}

pub(crate) fn parse_vindex_blocks(
    bytes: &[u8],
) -> Result<Option<Vec<VindexShannonBlock>>, Box<dyn core::error::Error>> {
    if !bytes.starts_with(b"LSB1") {
        return Ok(None);
    }
    if bytes.len() < 8 {
        return Err("truncated vindex block payload".into());
    }
    let block_count = u32::from_le_bytes(bytes[4..8].try_into()?) as usize;
    let mut offset = 8usize;
    let mut blocks = Vec::with_capacity(block_count);
    for _ in 0..block_count {
        if bytes.len().saturating_sub(offset) < 20 {
            return Err("truncated vindex block header".into());
        }
        let first_token = u32::from_le_bytes(bytes[offset..offset + 4].try_into()?);
        offset += 4;
        let target_tokens = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?);
        offset += 8;
        let payload_len = u64::from_le_bytes(bytes[offset..offset + 8].try_into()?) as usize;
        offset += 8;
        if bytes.len().saturating_sub(offset) < payload_len {
            return Err("truncated vindex block payload".into());
        }
        blocks.push(VindexShannonBlock {
            first_token,
            target_tokens,
            payload: bytes[offset..offset + payload_len].to_vec(),
        });
        offset += payload_len;
    }
    if offset != bytes.len() {
        return Err("trailing bytes after vindex block payload".into());
    }
    Ok(Some(blocks))
}
