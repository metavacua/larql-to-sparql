//! Python bindings for ResidualTrace — the complete record of inference.

use pyo3::prelude::*;

use std::path::Path;

use larql_inference::trace as trace_mod;
use larql_inference::trace::TracePositions;
use larql_inference::ffn::WeightFfn;
use larql_inference::ModelWeights;
use larql_vindex::tokenizers;

/// Complete inference trace — the residual stream DAG.
#[pyclass(name = "ResidualTrace", unsendable)]
pub struct PyResidualTrace {
    pub(crate) inner: trace_mod::ResidualTrace,
    pub(crate) weights_ptr: *const ModelWeights,
    pub(crate) tokenizer_ptr: *const tokenizers::Tokenizer,
}

impl PyResidualTrace {
    fn weights(&self) -> &ModelWeights {
        unsafe { &*self.weights_ptr }
    }
    fn tokenizer(&self) -> &tokenizers::Tokenizer {
        unsafe { &*self.tokenizer_ptr }
    }
}

#[pymethods]
impl PyResidualTrace {
    #[getter]
    fn prompt(&self) -> &str { &self.inner.prompt }

    #[getter]
    fn tokens(&self) -> Vec<String> { self.inner.tokens.clone() }

    #[getter]
    fn n_layers(&self) -> usize { self.inner.n_layers }

    #[getter]
    fn hidden_size(&self) -> usize { self.inner.hidden_size }

    #[getter]
    fn n_nodes(&self) -> usize { self.inner.nodes.len() }

    /// Top-k predictions at (layer, position). Position defaults to last token.
    #[pyo3(signature = (layer, position=None, k=5))]
    fn top_k(&self, layer: i32, position: Option<usize>, k: usize) -> Vec<(String, f32)> {
        let pos = position.unwrap_or_else(|| self.inner.tokens.len() - 1);
        self.inner.top_k(self.weights(), self.tokenizer(), layer, pos, k)
    }

    /// Rank of a token at (layer, position).
    #[pyo3(signature = (token, layer, position=None))]
    fn rank_of(&self, token: &str, layer: i32, position: Option<usize>) -> u32 {
        let pos = position.unwrap_or_else(|| self.inner.tokens.len() - 1);
        let tok_id = match self.tokenizer().encode(format!(" {}", token), true) {
            Ok(enc) => *enc.get_ids().last().unwrap_or(&0),
            Err(_) => return u32::MAX,
        };
        let node = match self.inner.node(layer, pos) {
            Some(n) => n,
            None => return u32::MAX,
        };
        let logits = self.inner.vocab_project(self.weights(), &node.residual);
        let probs = softmax_f32(&logits);
        probs.iter().filter(|&&p| p > probs[tok_id as usize]).count() as u32 + 1
    }

    /// Track answer rank, probability, and attn/ffn contribution through all layers.
    fn answer_trajectory(&self, answer: &str) -> PyResult<Vec<PyAnswerWaypoint>> {
        let tok_id = self.tokenizer().encode(format!(" {}", answer), true)
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let id = *tok_id.get_ids().last().unwrap_or(&0);
        let traj = self.inner.answer_trajectory(self.weights(), id);
        Ok(traj.into_iter().map(|w| PyAnswerWaypoint { inner: w }).collect())
    }

    /// Compact per-layer summary: norms, top prediction, delta norms.
    fn summary(&self) -> Vec<PyLayerSummary> {
        let summaries = self.inner.layer_summaries(self.weights(), self.tokenizer());
        summaries.into_iter().map(|s| PyLayerSummary { inner: s }).collect()
    }

    /// Get residual vector at (layer, position) as a list of floats.
    #[pyo3(signature = (layer, position=None))]
    fn residual(&self, layer: i32, position: Option<usize>) -> Option<Vec<f32>> {
        let pos = position.unwrap_or_else(|| self.inner.tokens.len() - 1);
        self.inner.node(layer, pos).map(|n| n.residual.clone())
    }

    /// Get attention delta at (layer, position) as a list of floats.
    #[pyo3(signature = (layer, position=None))]
    fn attn_delta(&self, layer: i32, position: Option<usize>) -> Option<Vec<f32>> {
        let pos = position.unwrap_or_else(|| self.inner.tokens.len() - 1);
        self.inner.node(layer, pos).map(|n| n.attn_delta.clone())
    }

    /// Get FFN delta at (layer, position) as a list of floats.
    #[pyo3(signature = (layer, position=None))]
    fn ffn_delta(&self, layer: i32, position: Option<usize>) -> Option<Vec<f32>> {
        let pos = position.unwrap_or_else(|| self.inner.tokens.len() - 1);
        self.inner.node(layer, pos).map(|n| n.ffn_delta.clone())
    }

    /// Save the trace to an mmap-friendly binary file.
    ///
    /// The file is append-only and can be re-opened for reading with
    /// zero-copy mmap access. Each token chain is written contiguously.
    fn save(&self, path: &str) -> PyResult<usize> {
        let mut writer = trace_mod::TraceWriter::create(
            Path::new(path), self.inner.hidden_size, self.inner.n_layers,
        ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        let written = writer.write_trace(&self.inner)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        writer.finish()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;

        Ok(written)
    }

    fn __repr__(&self) -> String {
        format!(
            "ResidualTrace('{}', {} tokens, {} layers, {} nodes)",
            self.inner.prompt, self.inner.tokens.len(),
            self.inner.n_layers, self.inner.nodes.len()
        )
    }
}

// ── Mmap'd trace store ──

/// Read-only mmap'd trace store — zero-copy access to frozen token chains.
#[pyclass(name = "TraceStore", unsendable)]
pub struct PyTraceStore {
    inner: trace_mod::TraceStore,
}

#[pymethods]
impl PyTraceStore {
    /// Open a trace file for mmap'd reading.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = trace_mod::TraceStore::open(Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: store })
    }

    #[getter]
    fn n_tokens(&self) -> usize { self.inner.n_tokens() }

    #[getter]
    fn n_layers(&self) -> usize { self.inner.n_layers() }

    #[getter]
    fn hidden_size(&self) -> usize { self.inner.hidden_size() }

    /// Read a residual vector. Zero-copy from mmap.
    /// Layer 0 = embedding, 1..n_layers = transformer layers.
    fn residual(&self, token: usize, layer: usize) -> Option<Vec<f32>> {
        self.inner.residual(token, layer).map(|s| s.to_vec())
    }

    /// Read attention delta. Zero-copy from mmap.
    fn attn_delta(&self, token: usize, layer: usize) -> Option<Vec<f32>> {
        self.inner.attn_delta(token, layer).map(|s| s.to_vec())
    }

    /// Read FFN delta. Zero-copy from mmap.
    fn ffn_delta(&self, token: usize, layer: usize) -> Option<Vec<f32>> {
        self.inner.ffn_delta(token, layer).map(|s| s.to_vec())
    }

    /// File size in bytes.
    fn file_size(&self) -> usize {
        HEADER_SIZE + self.inner.n_tokens() * self.chain_size()
    }

    fn __repr__(&self) -> String {
        let mb = (HEADER_SIZE + self.inner.n_tokens() * self.chain_size()) as f64 / 1e6;
        format!(
            "TraceStore({} tokens, {} layers, {}D, {:.1} MB)",
            self.inner.n_tokens(), self.inner.n_layers(),
            self.inner.hidden_size(), mb,
        )
    }
}

impl PyTraceStore {
    fn chain_size(&self) -> usize {
        let waypoints = self.inner.n_layers() + 1;
        waypoints * 3 * self.inner.hidden_size() * 4
    }
}

const HEADER_SIZE: usize = 64;

// ── Boundary store ──

/// Mmap'd boundary residual store — compressed context for infinite sequences.
///
/// Stores one residual per window boundary (~10 KB per 200 tokens).
/// 370K tokens → ~18.5 MB instead of 56 GB KV cache.
#[pyclass(name = "BoundaryStore", unsendable)]
pub struct PyBoundaryStore {
    inner: trace_mod::BoundaryStore,
}

#[pymethods]
impl PyBoundaryStore {
    /// Open an existing boundary file.
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let store = trace_mod::BoundaryStore::open(Path::new(path))
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: store })
    }

    #[getter] fn n_boundaries(&self) -> usize { self.inner.n_boundaries() }
    #[getter] fn total_tokens(&self) -> usize { self.inner.total_tokens() }
    #[getter] fn hidden_size(&self) -> usize { self.inner.hidden_size() }
    #[getter] fn window_size(&self) -> usize { self.inner.window_size() }

    /// Read boundary residual i — zero-copy from mmap.
    fn residual(&self, i: usize) -> Option<Vec<f32>> {
        self.inner.residual(i).map(|s| s.to_vec())
    }

    /// Find which boundary contains a given token offset.
    fn boundary_for_token(&self, token: usize) -> Option<usize> {
        self.inner.boundary_for_token(token)
    }

    /// Get the token range (start, end) for boundary i.
    fn token_range(&self, i: usize) -> Option<(usize, usize)> {
        self.inner.token_range(i)
    }

    fn __repr__(&self) -> String {
        let data_kb = self.inner.data_size() as f64 / 1024.0;
        format!(
            "BoundaryStore({} boundaries, {} tokens, {:.0} KB data, window={})",
            self.inner.n_boundaries(), self.inner.total_tokens(),
            data_kb, self.inner.window_size(),
        )
    }
}

/// Writable boundary store.
#[pyclass(name = "BoundaryWriter", unsendable)]
pub struct PyBoundaryWriter {
    inner: Option<trace_mod::BoundaryWriter>,
}

#[pymethods]
impl PyBoundaryWriter {
    /// Create a new boundary store file.
    #[new]
    #[pyo3(signature = (path, hidden_size, window_size=200, max_boundaries=10000))]
    fn new(path: &str, hidden_size: usize, window_size: usize, max_boundaries: usize) -> PyResult<Self> {
        let writer = trace_mod::BoundaryWriter::create(
            Path::new(path), hidden_size, window_size, max_boundaries,
        ).map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(Self { inner: Some(writer) })
    }

    /// Append a boundary residual.
    fn append(&mut self, token_offset: usize, window_tokens: usize, residual: Vec<f32>) -> PyResult<()> {
        let writer = self.inner.as_mut()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already finished"))?;
        writer.append(token_offset, window_tokens, &residual)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))
    }

    #[getter]
    fn n_boundaries(&self) -> usize {
        self.inner.as_ref().map(|w| w.n_boundaries()).unwrap_or(0)
    }

    #[getter]
    fn total_tokens(&self) -> usize {
        self.inner.as_ref().map(|w| w.total_tokens()).unwrap_or(0)
    }

    /// Flush and finalize the file.
    fn finish(&mut self) -> PyResult<String> {
        let writer = self.inner.take()
            .ok_or_else(|| pyo3::exceptions::PyRuntimeError::new_err("writer already finished"))?;
        let path = writer.finish()
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        Ok(path.to_string_lossy().to_string())
    }
}

/// Capture a trace from a WalkModel (called from PyWalkModel.trace).
pub fn capture_trace(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    prompt: &str,
    positions: &str,
) -> PyResult<PyResidualTrace> {
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let pos = match positions {
        "all" => TracePositions::All,
        _ => TracePositions::Last,
    };

    let ffn = WeightFfn { weights };
    let mut trace = trace_mod::trace_residuals(weights, &token_ids, pos, false, &ffn);

    trace.prompt = prompt.to_string();
    trace.tokens = token_ids.iter()
        .map(|&id| tokenizer.decode(&[id], true).unwrap_or_else(|_| format!("t{}", id)))
        .collect();

    Ok(PyResidualTrace {
        inner: trace,
        weights_ptr: weights as *const ModelWeights,
        tokenizer_ptr: tokenizer as *const tokenizers::Tokenizer,
    })
}

// ── Answer waypoint ──

#[pyclass(name = "AnswerWaypoint")]
pub struct PyAnswerWaypoint {
    inner: trace_mod::AnswerWaypoint,
}

#[pymethods]
impl PyAnswerWaypoint {
    #[getter] fn layer(&self) -> i32 { self.inner.layer }
    #[getter] fn rank(&self) -> u32 { self.inner.rank }
    #[getter] fn prob(&self) -> f32 { self.inner.prob }
    #[getter] fn attn_logit(&self) -> f32 { self.inner.attn_logit }
    #[getter] fn ffn_logit(&self) -> f32 { self.inner.ffn_logit }
    #[getter] fn residual_norm(&self) -> f32 { self.inner.residual_norm }

    fn __repr__(&self) -> String {
        let l = if self.inner.layer == -1 { "emb".to_string() } else { format!("L{}", self.inner.layer) };
        format!(
            "AnswerWaypoint({}, rank={}, prob={:.3}, attn={:.1}, ffn={:.1})",
            l, self.inner.rank, self.inner.prob, self.inner.attn_logit, self.inner.ffn_logit
        )
    }
}

// ── Layer summary ──

#[pyclass(name = "LayerSummary")]
pub struct PyLayerSummary {
    inner: trace_mod::LayerSummary,
}

#[pymethods]
impl PyLayerSummary {
    #[getter] fn layer(&self) -> i32 { self.inner.layer }
    #[getter] fn residual_norm(&self) -> f32 { self.inner.residual_norm }
    #[getter] fn attn_delta_norm(&self) -> f32 { self.inner.attn_delta_norm }
    #[getter] fn ffn_delta_norm(&self) -> f32 { self.inner.ffn_delta_norm }
    #[getter] fn top1_token(&self) -> &str { &self.inner.top1_token }
    #[getter] fn top1_prob(&self) -> f32 { self.inner.top1_prob }

    fn __repr__(&self) -> String {
        let l = if self.inner.layer == -1 { "emb".to_string() } else { format!("L{}", self.inner.layer) };
        format!(
            "LayerSummary({}, top1='{}' p={:.3}, |attn|={:.0}, |ffn|={:.0})",
            l, self.inner.top1_token, self.inner.top1_prob,
            self.inner.attn_delta_norm, self.inner.ffn_delta_norm
        )
    }
}

fn softmax_f32(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = logits.iter().map(|&l| ((l - max) as f64).exp()).sum();
    logits.iter().map(|&l| (((l - max) as f64).exp() / exp_sum) as f32).collect()
}
