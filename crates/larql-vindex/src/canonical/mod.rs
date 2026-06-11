pub mod types;
pub use types::{CanonicalMeta, LayerCanonicalInfo, Regime};

pub mod covariance;
pub use covariance::estimate_covariance;
