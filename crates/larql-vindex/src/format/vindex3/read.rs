//! Open a VINDEX3 container and reach its regions.
//!
//! The other half of the seam. [`write`](super::write) makes a directory that
//! declares itself VINDEX3; this makes one usable — which is what turns
//! "the runtime works on VINDEX2 operands" into "VINDEX3 works".
//!
//! Everything here is *resolution*, not execution: the container yields byte
//! slices and the shapes they were declared with, and the runtime binds them.
//! Keeping the two apart is what lets the same executor serve operands from
//! either generation without knowing which it got.

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use super::index::Vindex3Index;
use super::write::segment_path;
use crate::format::filenames::INDEX_JSON;
use crate::format::generation::{detect_generation, ContainerGeneration};
use crate::format::lyrw2::read::Lyrw2Reader;
use crate::format::moe_manifest::{MoeLayer, MoeManifest};
use crate::VindexError;

/// Wrap an IO failure with what was being read, keeping the original kind so a
/// caller can still branch on NotFound vs PermissionDenied.
fn contextual_io(what: &str, e: std::io::Error) -> VindexError {
    VindexError::Io(std::io::Error::new(e.kind(), format!("{what}: {e}")))
}

/// An opened VINDEX3 container: root authority, programme manifest, and the
/// segment bytes held resident for the reader's lifetime.
pub struct Vindex3Container {
    root: PathBuf,
    index: Vindex3Index,
    manifest: MoeManifest,
    /// Segment key → raw LYRW v2 file bytes.
    segments: HashMap<String, Vec<u8>>,
}

impl Vindex3Container {
    /// Open `root`, refusing anything that is not a VINDEX3 container.
    ///
    /// The generation is checked *first*, so a VINDEX2 directory is refused by
    /// name rather than producing a confusing parse failure three fields into
    /// a schema it was never written against (spec §12.1).
    pub fn open(root: &Path) -> Result<Self, VindexError> {
        match detect_generation(root)? {
            ContainerGeneration::V3 => {}
            // The crate already has a variant for exactly this refusal, and
            // reusing it keeps the message identical wherever the boundary is
            // crossed — §12.1 asks for a precise refusal, not a bespoke one.
            ContainerGeneration::V2 => {
                return Err(VindexError::WrongContainerGeneration {
                    found: "VINDEX2",
                    required: "VINDEX3",
                })
            }
        }

        let raw = std::fs::read_to_string(root.join(INDEX_JSON))?;
        let index: Vindex3Index = serde_json::from_str(&raw)
            .map_err(|e| VindexError::Parse(format!("parse VINDEX3 index.json: {e}")))?;

        let manifest_raw = std::fs::read_to_string(root.join(&index.moe_manifest))
            .map_err(|e| contextual_io(&index.moe_manifest, e))?;
        // `MoeManifest::parse` both deserialises and refuses a manifest whose
        // defects would only surface at bind time — unknown programme, duplicate
        // layer, malformed router.
        let manifest = MoeManifest::parse(&manifest_raw)?;

        let mut segments = HashMap::new();
        for key in index.segments.keys() {
            let path = segment_path(root, key);
            let bytes =
                std::fs::read(&path).map_err(|e| contextual_io(&format!("segment {key}"), e))?;
            segments.insert(key.clone(), bytes);
        }

        Ok(Self {
            root: root.to_path_buf(),
            index,
            manifest,
            segments,
        })
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    pub fn index(&self) -> &Vindex3Index {
        &self.index
    }

    pub fn manifest(&self) -> &MoeManifest {
        &self.manifest
    }

    /// The manifest entry for `layer`, or `None` if that layer is dense.
    pub fn layer(&self, layer: u32) -> Option<&MoeLayer> {
        self.manifest.layer(layer)
    }

    /// A parsed reader over the segment a bank names.
    ///
    /// Fails naming the key rather than the filename: the key is what the
    /// index declares and what a diagnosis should send someone to look at.
    pub fn segment(&self, key: &str) -> Result<Lyrw2Reader<'_>, VindexError> {
        let bytes = self.segments.get(key).ok_or_else(|| {
            VindexError::Parse(format!(
                "segment '{key}' is not declared by index.json; declared: {:?}",
                self.index.segments.keys().collect::<Vec<_>>()
            ))
        })?;
        Lyrw2Reader::parse(bytes).map_err(|e| VindexError::Parse(format!("segment '{key}': {e}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::format::vindex3::test_support::{
        fixture_a_spec, FIXTURE_A_EXPERTS, FIXTURE_A_LAYER, FIXTURE_A_SEGMENT_KEY,
    };
    use crate::format::vindex3::write::write_container;
    use tempfile::tempdir;

    fn opened() -> (tempfile::TempDir, Vindex3Container) {
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write");
        let c = Vindex3Container::open(dir.path()).expect("open");
        (dir, c)
    }

    #[test]
    fn a_written_container_opens_and_carries_its_manifest() {
        let (_dir, c) = opened();
        let layer = c.layer(FIXTURE_A_LAYER).expect("layer 0 is a MoE layer");
        assert_eq!(layer.routed_bank.experts, FIXTURE_A_EXPERTS);
        assert!(
            layer.routed_bank.resolve_programme().is_some(),
            "the declared programme must resolve, or nothing can bind it"
        );
    }

    #[test]
    fn the_declared_segment_parses_as_lyrw_v2() {
        // Round-trip through the real writer and the real reader — the two
        // halves that had never met before this module existed.
        let (_dir, c) = opened();
        let seg = c.segment(FIXTURE_A_SEGMENT_KEY).expect("segment parses");
        assert_eq!(seg.banks().len(), 1);
        assert_eq!(seg.banks()[0].num_entries, FIXTURE_A_EXPERTS);
    }

    #[test]
    fn an_undeclared_segment_is_refused_naming_what_is_declared() {
        let (_dir, c) = opened();
        let err = c.segment("routed/layer_999").unwrap_err();
        let msg = format!("{err}");
        assert!(msg.contains("routed/layer_999"), "{msg}");
        assert!(
            msg.contains(FIXTURE_A_SEGMENT_KEY),
            "must name what IS there: {msg}"
        );
    }

    #[test]
    fn an_opened_container_reports_where_it_came_from_and_what_it_declares() {
        let (dir, c) = opened();
        assert_eq!(c.root(), dir.path());
        assert_eq!(
            c.index().version,
            crate::format::generation::V3_CURRENT_SCHEMA
        );
        assert!(c
            .index()
            .declares_profile(crate::format::vindex3::PROFILE_EXACT));
        assert_eq!(c.manifest().layers.len(), 1);
    }

    #[test]
    fn a_dense_layer_has_no_manifest_entry() {
        // Layers absent from the manifest are dense — that is how an arbitrary
        // dense/MoE schedule is expressed, so asking for one must answer None
        // rather than erroring.
        let (_dir, c) = opened();
        assert!(c.layer(FIXTURE_A_LAYER + 99).is_none());
    }

    #[test]
    fn a_missing_manifest_file_is_refused_naming_it() {
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write");
        std::fs::remove_file(dir.path().join(crate::format::vindex3::MOE_MANIFEST_JSON)).unwrap();
        let msg = match Vindex3Container::open(dir.path()) {
            Ok(_) => panic!("a container without its manifest must not open"),
            Err(e) => format!("{e}"),
        };
        assert!(msg.contains("moe_manifest.json"), "{msg}");
    }

    #[test]
    fn a_declared_segment_that_is_absent_is_refused_naming_the_key() {
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write");
        std::fs::remove_file(crate::format::vindex3::segment_path(
            dir.path(),
            FIXTURE_A_SEGMENT_KEY,
        ))
        .unwrap();
        let msg = match Vindex3Container::open(dir.path()) {
            Ok(_) => panic!("a container missing a declared segment must not open"),
            Err(e) => format!("{e}"),
        };
        assert!(msg.contains(FIXTURE_A_SEGMENT_KEY), "{msg}");
    }

    #[test]
    fn a_malformed_index_is_refused_as_a_parse_failure() {
        let dir = tempdir().unwrap();
        write_container(dir.path(), &fixture_a_spec()).expect("write");
        // Keep `version: 3` so detection still routes here, then break the rest.
        std::fs::write(dir.path().join(INDEX_JSON), r#"{"version":3,"model":42}"#).unwrap();
        assert!(Vindex3Container::open(dir.path()).is_err());
    }

    #[test]
    fn the_v3_loader_refuses_a_v2_container_by_name() {
        // The generation boundary, from the loader's side. §12.1 requires a
        // precise refusal, never a parse error and never a conversion.
        let dir = tempdir().unwrap();
        std::fs::write(
            dir.path().join(INDEX_JSON),
            serde_json::json!({
                "version": 2, "model": "m", "family": "llama", "num_layers": 1,
                "hidden_size": 8, "intermediate_size": 8, "vocab_size": 4,
                "embed_scale": 1.0, "layers": [], "down_top_k": 0
            })
            .to_string(),
        )
        .unwrap();
        // `Vindex3Container` holds segment bytes and deliberately has no
        // Debug, so match rather than `unwrap_err`.
        let msg = match Vindex3Container::open(dir.path()) {
            Ok(_) => panic!("the VINDEX3 loader must refuse a VINDEX2 container"),
            Err(e) => format!("{e}"),
        };
        assert!(msg.contains("VINDEX2"), "{msg}");
        assert!(msg.contains("VINDEX3 loader"), "{msg}");
    }
}
