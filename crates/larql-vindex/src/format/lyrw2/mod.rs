//! LYRW v2 — the VINDEX3 per-layer weight container (format spec §6).
//!
//! LYRW v2 describes **storage only**: banks, entries, region schemas, offsets
//! and formats. It carries no programme identity — the MoE manifest binds
//! `bank_id → programme`, and keeping that binding in exactly one place is why
//! "the binary says programme 4, the manifest says gpt-oss-expert-v1" cannot
//! happen.
//!
//! # File layout
//!
//! ```text
//! header                24 B
//! bank descriptors      num_banks    x 24 B
//! segment descriptors   num_segments x 12 B
//! region schemas        sum over banks of (region_schema_count x 20 B)
//! entry tables          per segment: entry_count x schemas x 16 B
//! -- padded to 64 B --
//! payload regions       each 64-B aligned
//! ```
//!
//! # Relationship to LYRW v1
//!
//! None, deliberately (spec §6.6). This module never opens a v1 layer file and
//! the v1 loader never opens a v2 one. The shared magic exists so each side
//! can *refuse* the other precisely instead of mis-parsing it.

pub mod bank;
pub mod browse_mode;
pub mod consts;
pub mod error;
pub mod header;
pub mod layout;
pub mod plan;
#[cfg(test)]
mod plan_tests;
pub mod read;
#[cfg(test)]
mod read_refusal_tests;
#[cfg(test)]
mod read_tests;
pub mod region_format;
pub mod region_role;
pub mod region_schema;
pub mod segment;
#[cfg(test)]
mod test_fixtures;
pub mod wire;
pub mod write;
#[cfg(test)]
mod write_tests;

pub use bank::{BankDescriptor, BankKind};
pub use browse_mode::BrowseMode;
pub use consts::{FORMAT_VERSION, MAGIC, PAIR_ID_UNPAIRED, REGION_ALIGNMENT};
pub use error::Lyrw2Error;
pub use header::Lyrw2Header;
pub use layout::Lyrw2Layout;
pub use plan::Lyrw2Plan;
pub use read::{Lyrw2Reader, ResolvedRegion};
pub use region_format::{Packing, RegionFormat};
pub use region_role::RegionRole;
pub use region_schema::RegionSchema;
pub use segment::SegmentDescriptor;
pub use write::{Lyrw2Writer, RegionCursor};
