// SPDX-License-Identifier: AGPL-3.0-only
// Copyright (C) 2026 Ian Douglas Lawrence Norman McLean
pub mod attention;
pub mod complex;
pub mod hyperbolic;
#[cfg(not(target_arch = "wasm32"))]
pub mod vindex_loader;

#[cfg(feature = "gpu")]
pub mod gpu;

#[cfg(feature = "browser")]
pub mod bridge;

pub use attention::{AttnBackend, AttnInput, AttnOutput};
