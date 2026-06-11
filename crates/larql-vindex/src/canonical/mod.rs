pub mod types;
pub use types::{CanonicalMeta, LayerCanonicalInfo, Regime};

pub mod covariance;
pub use covariance::estimate_covariance;

pub mod whitening;
pub use whitening::{compute_whitening, unpack_l, WhiteningData};

pub mod onshell;
pub use onshell::{compute_onshell_mask, onshell_stats};

pub mod regime;
pub use regime::classify_layer_regime;
