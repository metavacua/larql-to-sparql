//! Typed errors for the Minimal-Viable Vindex (MVV) reader and query kernel.
//!
//! Every MVV operation returns a defined value or one of these typed errors on
//! ALL inputs — never a panic and never UB. `MvvError` is the codomain of the
//! totality contract in `docs/minimal-vindex-spec.md`.
//!
//! Deliberately portable and dependency-light: a plain enum + a hand-written
//! `Display` (no `thiserror`, no `std`), so it compiles unchanged on
//! wasm32v1-none and contributes nothing to the gated surface.
use core::fmt;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MvvError {
    /// `index.json` did not parse as the minimal descriptor.
    BadIndexJson,
    /// `version` is not a supported MVV descriptor version.
    UnsupportedVersion(u32),
    /// `num_layers` disagrees with `layers.len()`.
    LayerCountMismatch { declared: usize, actual: usize },
    /// A layer's `length` != `num_features * hidden_size * bytes_per_float(dtype)`,
    /// or an arithmetic overflow was hit computing that product.
    LayerLengthMismatch { layer: usize, expected: usize, declared: usize },
    /// A layer's byte length is not a whole multiple of the dtype width.
    Misaligned { layer: usize },
    /// The blob does not exactly cover the declared layers (too short, or, for
    /// a minimal blob, trailing slack). `expected` = bytes the layers require.
    BlobLengthMismatch { expected: usize, actual: usize },
    /// Queried layer index is out of range.
    LayerOutOfRange { layer: usize, num_layers: usize },
    /// `residual.len()` != `hidden_size`.
    DimMismatch { expected: usize, actual: usize },
    /// A feature range `[start, end)` is invalid for the layer.
    FeatureRange { start: usize, end: usize, num_features: usize },
    /// An optional self-describing header disagrees with `index.json`
    /// (index.json is the source of truth).
    HeaderMismatch,
}

impl fmt::Display for MvvError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MvvError::BadIndexJson => write!(f, "index.json did not parse as a minimal vindex descriptor"),
            MvvError::UnsupportedVersion(v) => write!(f, "unsupported vindex descriptor version: {v}"),
            MvvError::LayerCountMismatch { declared, actual } =>
                write!(f, "num_layers={declared} but layers[] has {actual} entries"),
            MvvError::LayerLengthMismatch { layer, expected, declared } =>
                write!(f, "layer {layer}: declared length {declared} != expected {expected}"),
            MvvError::Misaligned { layer } =>
                write!(f, "layer {layer}: byte length is not a whole number of dtype elements"),
            MvvError::BlobLengthMismatch { expected, actual } =>
                write!(f, "blob length {actual} does not match the {expected} bytes the layers require"),
            MvvError::LayerOutOfRange { layer, num_layers } =>
                write!(f, "layer {layer} out of range (num_layers={num_layers})"),
            MvvError::DimMismatch { expected, actual } =>
                write!(f, "residual length {actual} != hidden_size {expected}"),
            MvvError::FeatureRange { start, end, num_features } =>
                write!(f, "feature range [{start},{end}) invalid for num_features={num_features}"),
            MvvError::HeaderMismatch =>
                write!(f, "self-describing header disagrees with index.json"),
        }
    }
}
