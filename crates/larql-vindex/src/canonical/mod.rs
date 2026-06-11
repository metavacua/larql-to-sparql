pub mod types;
pub use types::{CanonicalMeta, LayerCanonicalInfo, Regime};

pub mod covariance;
pub use covariance::estimate_covariance;

pub mod whitening;
pub use whitening::{compute_whitening, unpack_l, WhiteningData};
