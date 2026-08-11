//! Container generation detection and dispatch (format spec §12.1).
//!
//! One larql binary supports both vindex generations, indefinitely, for reading
//! and serving. `index.json`'s `version` field is the **sole** discriminator —
//! no filename sniffing, no directory-shape heuristics:
//!
//! | `index.json.version` | generation | layer format |
//! | -------------------- | ---------- | ------------ |
//! | 1                    | VINDEX2    | LYRW `format_version` 1 |
//! | 2                    | VINDEX2    | LYRW `format_version` 1 |
//! | 3                    | VINDEX3    | LYRW `format_version` 2 |
//!
//! The generation is named for its *current* `index.json.version`. An earlier
//! draft called the shipped generation "VINDEX1" while its version was already
//! 2, which put a permanent off-by-one between the name and the discriminator
//! — the single most likely way to mis-detect a directory. Both were renamed
//! so the two agree.
//!
//! **The mapping is not a bijection below 2.** `index.json.version` 1 is a
//! *legacy schema of the same shipped generation*, not a pre-generation
//! artifact: such indexes exist in the wild and the loader reads them by
//! filling absent fields with defaults. Refusing them would break VINDEX2
//! compatibility, which is the one thing dual-generation support exists to
//! protect. This was caught by the E0 preservation matrix, which is why that
//! matrix runs on every commit.
//!
//! The layer format keeps its own sequence and is deliberately *not* aligned to
//! either: LYRW is a different artifact with a different lifetime, and its own
//! numbering was already correct. A VINDEX3 container holds LYRW v2 files.
//!
//! Detection fails closed. An unknown version names the version found and the
//! versions this binary supports, before any weight byte is read — a loader
//! that guesses produces a served model with wrong weights, not an error.

#[cfg(not(target_arch = "wasm32"))]
use std::path::Path;

use crate::format::filenames::INDEX_JSON;
use crate::VindexError;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// An `index.json` schema revision.
///
/// A newtype so no API can accept a bare `u32` and call it a generation. The
/// regression that motivated this compiled perfectly while conflating the two.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IndexSchemaVersion(pub u32);

impl IndexSchemaVersion {
    pub const fn get(self) -> u32 {
        self.0
    }
}

impl core::fmt::Display for IndexSchemaVersion {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// Oldest `index.json` schema still recognised as the shipped generation.
///
/// Schema 1 predates several fields and loads with defaults. It is the same
/// container generation, not an older one.
pub const V2_MIN_SCHEMA: u32 = 1;
/// What a fresh extraction of the shipped generation writes.
pub const V2_CURRENT_SCHEMA: u32 = 2;
pub const V3_MIN_SCHEMA: u32 = 3;
pub const V3_CURRENT_SCHEMA: u32 = 3;

/// Which container generation a directory holds.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ContainerGeneration {
    /// The shipped generation: `index.json` schemas 1-2, LYRW `format_version` 1.
    V2,
    /// The successor: `index.json` schema 3, LYRW `format_version` 2.
    V3,
}

impl ContainerGeneration {
    /// The schema a fresh extraction of this generation writes.
    pub const fn current_schema_version(self) -> IndexSchemaVersion {
        match self {
            Self::V2 => IndexSchemaVersion(V2_CURRENT_SCHEMA),
            Self::V3 => IndexSchemaVersion(V3_CURRENT_SCHEMA),
        }
    }

    /// Every schema this generation can read. Many-to-one, not a bijection.
    pub const fn supported_schema_versions(self) -> core::ops::RangeInclusive<u32> {
        match self {
            Self::V2 => V2_MIN_SCHEMA..=V2_CURRENT_SCHEMA,
            Self::V3 => V3_MIN_SCHEMA..=V3_CURRENT_SCHEMA,
        }
    }

    pub fn reads_schema(self, version: IndexSchemaVersion) -> bool {
        self.supported_schema_versions().contains(&version.get())
    }

    /// The LYRW `format_version` this generation's layer files carry.
    pub const fn lyrw_format_version(self) -> u32 {
        match self {
            Self::V2 => 1,
            Self::V3 => 2,
        }
    }

    /// Human-readable name used in diagnostics — "VINDEX2" / "VINDEX3".
    pub const fn name(self) -> &'static str {
        match self {
            Self::V2 => "VINDEX2",
            Self::V3 => "VINDEX3",
        }
    }

    /// Map a LYRW `format_version` to the generation that writes it.
    pub fn from_lyrw_format_version(found: u32) -> Result<Self, VindexError> {
        match found {
            1 => Ok(Self::V2),
            2 => Ok(Self::V3),
            other => Err(VindexError::UnknownContainerGeneration {
                found: other,
                supported: "1 (VINDEX2), 2 (VINDEX3)".into(),
            }),
        }
    }

    /// Refuse if this is not the generation the caller's path handles.
    pub fn require(self, required: Self) -> Result<(), VindexError> {
        if self == required {
            return Ok(());
        }
        Err(VindexError::WrongContainerGeneration {
            found: self.name(),
            required: required.name(),
        })
    }
}

/// Every generation this binary implements, oldest first.
pub const ALL_GENERATIONS: [ContainerGeneration; 2] =
    [ContainerGeneration::V2, ContainerGeneration::V3];

/// Map a schema revision to its owning container generation.
///
/// `index.json.version` remains the **sole** dispatch input — no filename
/// sniffing, no directory-shape heuristics — but it is a schema discriminator,
/// not a generation identifier. The loader maps supported schema revisions to
/// the generation that owns them, and that mapping is many-to-one.
pub fn generation_for_schema(
    version: IndexSchemaVersion,
) -> Result<ContainerGeneration, VindexError> {
    ALL_GENERATIONS
        .into_iter()
        .find(|g| g.reads_schema(version))
        .ok_or_else(|| VindexError::UnknownContainerGeneration {
            found: version.get(),
            supported: supported_schema_summary(),
        })
}

/// "1-2 (VINDEX2), 3 (VINDEX3)" — every schema this binary reads.
pub fn supported_schema_summary() -> String {
    ALL_GENERATIONS
        .into_iter()
        .map(|g| {
            let r = g.supported_schema_versions();
            if r.start() == r.end() {
                format!("{} ({})", r.start(), g.name())
            } else {
                format!("{}-{} ({})", r.start(), r.end(), g.name())
            }
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Read `index.json` and report which generation the directory holds.
///
/// Deliberately parses only the `version` field: a VINDEX3 `index.json` carries
/// keys the VINDEX2 config struct does not model, and full deserialisation
/// would fail on shape before it could report the far more useful "wrong
/// generation".
#[cfg(not(target_arch = "wasm32"))]
pub fn detect_generation(dir: &Path) -> Result<ContainerGeneration, VindexError> {
    let path = dir.join(INDEX_JSON);
    let text = std::fs::read_to_string(&path)?;
    let probe: VersionProbe =
        serde_json::from_str(&text).map_err(|e| VindexError::Parse(e.to_string()))?;
    match probe.version {
        Some(v) => generation_for_schema(IndexSchemaVersion(v)),
        None => Err(VindexError::UnknownContainerGeneration {
            found: 0,
            supported: format!(
                "{}; index.json declared no version field",
                supported_schema_summary()
            ),
        }),
    }
}

/// Minimal view over `index.json` — the version field and nothing else.
#[derive(serde::Deserialize)]
struct VersionProbe {
    version: Option<u32>,
}
