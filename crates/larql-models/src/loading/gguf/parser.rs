//! GGUF file parsing — GgufFile::open/open_single, shard filename parsing, sibling discovery.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;
use std::collections::HashMap;
use std::io::{BufReader, Seek};
use std::path::Path;

use crate::detect::ModelError;

use super::constants::GGUF_MAGIC;
use super::reader::{read_string, read_u32, read_u64, read_value};
use super::types::{GgufFile, GgufTensorInfo, ShardInfo};

/// Parse a multi-shard GGUF filename of the form
/// `<prefix>-<NNNNN>-of-<NNNNN>.gguf` (canonical llama.cpp split layout)
/// and return `(prefix_without_dashes, this_shard_idx_0based, total_shards)`.
///
/// Returns `None` for filenames that don't match the pattern (i.e. single
/// files); the caller treats those as single-shard GGUFs.
pub(crate) fn parse_shard_filename(path: &Path) -> Option<(String, usize, usize)> {
    let name = path.file_name()?.to_str()?;
    let stem = name.strip_suffix(".gguf")?;
    // Tail must be `<prefix>-NNNNN-of-NNNNN` with matching widths.
    // Rightmost run of digits = "NNNNN" (total shard count).
    let count_start = stem
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    if count_start >= stem.len() {
        return None; // no trailing digits at all
    }
    let count_str = &stem[count_start..];
    let before_count = &stem[..count_start]; // "<prefix>-NNNNN-of-"
    let before_of = before_count.strip_suffix("-of-")?;
    // Then second rightmost digits run = "NNNNN" (this shard's 1-based index).
    let idx_start = before_of
        .rfind(|c: char| !c.is_ascii_digit())
        .map(|i| i + 1)
        .unwrap_or(0);
    if idx_start >= before_of.len() {
        return None;
    }
    let idx_str = &before_of[idx_start..];
    let prefix = before_of[..idx_start].strip_suffix('-')?;

    let this_idx_1based: usize = idx_str.parse().ok()?;
    let total: usize = count_str.parse().ok()?;
    if this_idx_1based == 0 || this_idx_1based > total {
        return None;
    }
    // Width must match across the two numbers (llama.cpp convention).
    if idx_str.len() != count_str.len() {
        return None;
    }
    Some((prefix.to_string(), this_idx_1based - 1, total))
}

/// Discover the full set of sibling shards making up a multi-shard GGUF.
/// `path` is one shard the user pointed at; the returned vec is ordered by
/// shard index (shard 1 first → shard N last) and is guaranteed to be of
/// length `expected_total`.
pub(crate) fn discover_shard_siblings(
    parent: &Path,
    path: &Path,
    expected_total: usize,
) -> Result<Vec<std::path::PathBuf>, ModelError> {
    let (prefix, _, total_from_name) = parse_shard_filename(path).ok_or_else(|| {
        ModelError::Parse(format!(
            "multi-shard GGUF without canonical -NNNNN-of-NNNNN filename: {}",
            path.display()
        ))
    })?;
    if expected_total != total_from_name {
        return Err(ModelError::Parse(format!(
            "shard total mismatch: split.count={expected_total} but filename says of-{total_from_name}",
        )));
    }
    // Detect the widths used in the filename so we reconstruct sibling
    // names byte-for-byte (00001 vs 001).
    let name_str = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
    let total_width = name_str
        .strip_suffix(".gguf")
        .and_then(|s| s.rsplit("-of-").next())
        .map(|n| n.len())
        .unwrap_or(5);
    let width = name_str
        .strip_suffix(".gguf")
        .and_then(|s| {
            s.strip_suffix(&format!(
                "-of-{expected_total:0>tot_w$}",
                tot_w = total_width
            ))
        })
        .and_then(|s| s.rsplit('-').next())
        .map(|n| n.len())
        .unwrap_or(total_width);

    let mut paths = Vec::with_capacity(expected_total);
    for i in 1..=expected_total {
        let fname = format!(
            "{prefix}-{i:0>idx_width$}-of-{total:0>tot_width$}.gguf",
            prefix = prefix,
            i = i,
            idx_width = width,
            total = expected_total,
            tot_width = total_width,
        );
        let p = parent.join(&fname);
        if !p.exists() {
            return Err(ModelError::Parse(format!(
                "multi-shard GGUF missing expected sibling: {} (looking for shard {} of {})",
                p.display(),
                i,
                expected_total,
            )));
        }
        paths.push(p);
    }
    Ok(paths)
}

impl GgufFile {
    /// Parse a GGUF file header and tensor info (does not read tensor data yet).
    ///
    /// Detects multi-shard splits by checking the `split.count` GGUF metadata
    /// key on the file you point at; when `split.count > 1` (or the filename
    /// matches the canonical `*-NNNNN-of-NNNNN.gguf` pattern), sibling shards
    /// in the same directory are also discovered and their tensor infos are
    /// merged into the returned `GgufFile`. Tensors carry a `shard_idx`
    /// internally so [`Self::load_tensors_filtered`] reads each from the
    /// right shard.
    pub fn open(path: &Path) -> Result<Self, ModelError> {
        let mut gguf = Self::open_single(path)?;

        // Multi-shard detection: prefer the explicit `split.*` metadata
        // emitted by llama-gguf-split, fall back to the filename pattern
        // (some splitters skip the metadata).
        let split_count = gguf
            .metadata
            .get("split.count")
            .and_then(|v| v.as_u32())
            .unwrap_or(0);
        let pattern_count = parse_shard_filename(path).map(|(_, _, total)| total);
        let total_shards = match (split_count, pattern_count) {
            (n, _) if n > 1 => n as usize,
            (_, Some(n)) if n > 1 => n,
            _ => return Ok(gguf), // single-file
        };

        // We need every shard in the split — find them all.
        let parent = path.parent().ok_or_else(|| {
            ModelError::Parse(format!("GGUF path has no parent: {}", path.display()))
        })?;
        let shard_paths = discover_shard_siblings(parent, path, total_shards)?;
        debug_assert_eq!(shard_paths.len(), total_shards);

        // The first entry is the shard we already loaded (whichever the
        // caller pointed at). Rewrite `gguf` to be anchored at shard 0 and
        // then accumulate the remaining shards' tensor infos.
        let this_idx = shard_paths.iter().position(|p| p == path).ok_or_else(|| {
            ModelError::Parse(format!(
                "passed shard {} not found in discovered set",
                path.display()
            ))
        })?;
        let mut shards: Vec<ShardInfo> = Vec::with_capacity(total_shards);
        let mut combined_infos: Vec<GgufTensorInfo> = Vec::new();
        for (idx, shard_path) in shard_paths.iter().enumerate() {
            if idx == this_idx {
                shards.push(ShardInfo {
                    path: path.to_path_buf(),
                    data_offset: gguf.data_offset,
                });
                for info in &gguf.tensor_infos {
                    combined_infos.push(GgufTensorInfo {
                        name: info.name.clone(),
                        n_dims: info.n_dims,
                        dims: info.dims.clone(),
                        tensor_type: info.tensor_type,
                        offset: info.offset,
                        shard_idx: idx,
                    });
                }
            } else {
                let other = Self::open_single(shard_path)?;
                shards.push(ShardInfo {
                    path: shard_path.clone(),
                    data_offset: other.data_offset,
                });
                for mut info in other.tensor_infos {
                    info.shard_idx = idx;
                    combined_infos.push(info);
                }
            }
        }

        // Sanity check: total tensor count should match split.tensors.count
        // when that key is emitted (llama-gguf-split always writes it).
        if let Some(expected) = gguf
            .metadata
            .get("split.tensors.count")
            .and_then(|v| v.as_u32())
        {
            if combined_infos.len() != expected as usize {
                return Err(ModelError::Parse(format!(
                    "multi-shard tensor count mismatch: combined {} shards yielded \
                     {} tensors, but split.tensors.count = {}",
                    total_shards,
                    combined_infos.len(),
                    expected
                )));
            }
        }

        gguf.tensor_infos = combined_infos;
        gguf.shards = shards;
        // `gguf.path` / `gguf.data_offset` keep pointing at the
        // user-supplied shard for back-compat with diagnostics; the
        // multi-shard loader uses `shards[info.shard_idx]` internally.
        Ok(gguf)
    }

    /// Open a single GGUF file without multi-shard discovery. Used as the
    /// per-shard primitive by [`Self::open`].
    fn open_single(path: &Path) -> Result<Self, ModelError> {
        let file = std::fs::File::open(path)?;
        let mut r = BufReader::new(file);

        // Magic
        let magic = read_u32(&mut r)?;
        if magic != GGUF_MAGIC {
            return Err(ModelError::Parse(format!(
                "not a GGUF file (magic: 0x{:08X}, expected 0x{:08X})",
                magic, GGUF_MAGIC
            )));
        }

        // Version
        let version = read_u32(&mut r)?;
        if !(2..=3).contains(&version) {
            return Err(ModelError::Parse(format!(
                "unsupported GGUF version: {version}"
            )));
        }

        let n_tensors = read_u64(&mut r)? as usize;
        let n_metadata = read_u64(&mut r)? as usize;

        // Read metadata
        let mut metadata = HashMap::new();
        for _ in 0..n_metadata {
            let key = read_string(&mut r)?;
            let value = read_value(&mut r)?;
            metadata.insert(key, value);
        }

        // Read tensor infos
        let mut tensor_infos = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = read_string(&mut r)?;
            let n_dims = read_u32(&mut r)?;
            let mut dims = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dims.push(read_u64(&mut r)?);
            }
            let tensor_type = read_u32(&mut r)?;
            let offset = read_u64(&mut r)?;
            tensor_infos.push(GgufTensorInfo {
                name,
                n_dims,
                dims,
                tensor_type,
                offset,
                shard_idx: 0,
            });
        }

        // Data starts at next alignment boundary (32 bytes)
        let pos = r.stream_position().map_err(ModelError::Io)?;
        let alignment = 32u64;
        let data_offset = pos.div_ceil(alignment) * alignment;

        Ok(GgufFile {
            metadata,
            tensor_infos,
            data_offset,
            path: path.to_path_buf(),
            shards: vec![ShardInfo {
                path: path.to_path_buf(),
                data_offset,
            }],
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_shard_filename_canonical_layout() {
        let p = std::path::PathBuf::from("/x/Kimi-K2.6-UD-Q8_K_XL-00003-of-00014.gguf");
        let (prefix, idx, total) = parse_shard_filename(&p).unwrap();
        assert_eq!(prefix, "Kimi-K2.6-UD-Q8_K_XL");
        assert_eq!(idx, 2);
        assert_eq!(total, 14);
    }

    #[test]
    fn parse_shard_filename_rejects_single_file() {
        let p = std::path::PathBuf::from("/x/llama-3.1-8b-q4.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_unmatched_widths() {
        let p = std::path::PathBuf::from("/x/foo-00003-of-0014.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn parse_shard_filename_supports_3digit_split() {
        let p = std::path::PathBuf::from("/x/foo-001-of-003.gguf");
        let (prefix, idx, total) = parse_shard_filename(&p).unwrap();
        assert_eq!(prefix, "foo");
        assert_eq!(idx, 0);
        assert_eq!(total, 3);
    }

    #[test]
    fn parse_shard_filename_rejects_index_zero() {
        let p = std::path::PathBuf::from("/x/foo-00000-of-00003.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_index_exceeding_total() {
        let p = std::path::PathBuf::from("/x/foo-00004-of-00003.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_no_trailing_digits() {
        let p = std::path::PathBuf::from("/x/foo-abc.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn parse_shard_filename_rejects_all_digits_before_of() {
        // "00003-of-00003.gguf" — digits run to start, no prefix with '-'
        let p = std::path::PathBuf::from("/x/00003-of-00003.gguf");
        assert!(parse_shard_filename(&p).is_none());
    }

    #[test]
    fn discover_shard_siblings_rejects_total_mismatch() {
        let dir = tempfile::tempdir().unwrap();
        for i in 1..=3 {
            std::fs::File::create(dir.path().join(format!("m-{i:0>5}-of-00003.gguf"))).unwrap();
        }
        let first = dir.path().join("m-00001-of-00003.gguf");
        let err = discover_shard_siblings(dir.path(), &first, 5).unwrap_err();
        assert!(
            format!("{err}").contains("shard total mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn discover_shard_siblings_finds_all_in_order() {
        let dir = tempfile::tempdir().unwrap();
        for i in 1..=3 {
            std::fs::File::create(dir.path().join(format!("model-{i:0>5}-of-00003.gguf"))).unwrap();
        }
        let middle = dir.path().join("model-00002-of-00003.gguf");
        let paths = discover_shard_siblings(dir.path(), &middle, 3).unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths[0].ends_with("model-00001-of-00003.gguf"));
        assert!(paths[1].ends_with("model-00002-of-00003.gguf"));
        assert!(paths[2].ends_with("model-00003-of-00003.gguf"));
    }

    #[test]
    fn discover_shard_siblings_finds_3digit_splits() {
        let dir = tempfile::tempdir().unwrap();
        for i in 1..=3 {
            std::fs::File::create(dir.path().join(format!("foo-{i:0>3}-of-003.gguf"))).unwrap();
        }
        let first = dir.path().join("foo-001-of-003.gguf");
        let paths = discover_shard_siblings(dir.path(), &first, 3).unwrap();
        assert_eq!(paths.len(), 3);
        assert!(paths[0].ends_with("foo-001-of-003.gguf"));
        assert!(paths[1].ends_with("foo-002-of-003.gguf"));
        assert!(paths[2].ends_with("foo-003-of-003.gguf"));
    }

    #[test]
    fn discover_shard_siblings_errors_when_one_missing() {
        let dir = tempfile::tempdir().unwrap();
        for i in [1usize, 3] {
            std::fs::File::create(dir.path().join(format!("m-{i:0>5}-of-00003.gguf"))).unwrap();
        }
        let first = dir.path().join("m-00001-of-00003.gguf");
        let err = discover_shard_siblings(dir.path(), &first, 3).unwrap_err();
        assert!(
            format!("{err}").contains("missing expected sibling"),
            "unexpected error: {err}"
        );
    }

    /// End-to-end multi-shard open: two real GGUF files with different
    /// tensors in each, joined via canonical `-NNNNN-of-00002.gguf` layout.
    /// Verifies discovery, shard_idx assignment, and per-shard tensor
    /// reads via `load_tensors`.
    #[test]
    fn open_multi_shard_combines_tensors_from_all_shards() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();

        let write_shard =
            |idx: usize, tensor_ids: &[usize], metas: &[(&str, u32)]| -> std::path::PathBuf {
                let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
                let mut file = std::fs::File::create(&path).unwrap();
                file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
                file.write_all(&3u32.to_le_bytes()).unwrap();
                file.write_all(&(tensor_ids.len() as u64).to_le_bytes())
                    .unwrap();
                file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();

                for (k, v) in metas {
                    let kb = k.as_bytes();
                    file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
                    file.write_all(kb).unwrap();
                    file.write_all(&4u32.to_le_bytes()).unwrap(); // u32 type tag
                    file.write_all(&v.to_le_bytes()).unwrap();
                }

                for (rel, &tid) in tensor_ids.iter().enumerate() {
                    let name = format!("blk.{tid}.ffn_down.weight");
                    let nb = name.as_bytes();
                    file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
                    file.write_all(nb).unwrap();
                    file.write_all(&2u32.to_le_bytes()).unwrap();
                    file.write_all(&2u64.to_le_bytes()).unwrap();
                    file.write_all(&2u64.to_le_bytes()).unwrap();
                    file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                        .unwrap();
                    let off = (rel as u64) * 16;
                    file.write_all(&off.to_le_bytes()).unwrap();
                }

                let pos = file.stream_position().unwrap();
                let aligned = pos.div_ceil(32) * 32;
                file.write_all(&vec![0u8; (aligned - pos) as usize])
                    .unwrap();

                for &tid in tensor_ids {
                    for off in 0..4 {
                        file.write_all(&((tid as f32) + 0.1 * off as f32).to_le_bytes())
                            .unwrap();
                    }
                }
                file.flush().unwrap();
                path
            };

        let p1 = write_shard(
            1,
            &[0, 1],
            &[
                ("split.no", 0),
                ("split.count", 2),
                ("split.tensors.count", 4),
            ],
        );
        let _p2 = write_shard(
            2,
            &[2, 3],
            &[
                ("split.no", 1),
                ("split.count", 2),
                ("split.tensors.count", 4),
            ],
        );

        let gguf = GgufFile::open(&p1).unwrap();
        assert_eq!(gguf.shards.len(), 2);
        assert_eq!(gguf.tensor_infos.len(), 4);
        for (i, info) in gguf.tensor_infos.iter().enumerate() {
            let expected = if i < 2 { 0 } else { 1 };
            assert_eq!(
                info.shard_idx, expected,
                "tensor {i} ({}) shard mismatch",
                info.name
            );
        }

        let (tensors, _) = gguf.load_tensors().unwrap();
        assert_eq!(tensors.len(), 4);
        for tid in 0..4 {
            let key = format!("layers.{tid}.mlp.down_proj.weight");
            let arr = tensors.get(&key).unwrap_or_else(|| panic!("missing {key}"));
            assert!(
                (arr[[0, 0]] - tid as f32).abs() < 1e-6,
                "tensor {tid} top-left {} != {tid}",
                arr[[0, 0]]
            );
        }
    }

    #[test]
    fn open_rejects_multi_shard_when_a_shard_file_is_missing() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path().join("m-00001-of-00002.gguf");
        let mut file = std::fs::File::create(&p).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap();
        let k = "split.count".as_bytes();
        file.write_all(&(k.len() as u64).to_le_bytes()).unwrap();
        file.write_all(k).unwrap();
        file.write_all(&4u32.to_le_bytes()).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let err = match GgufFile::open(&p) {
            Ok(_) => panic!("expected error for missing sibling shard"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("missing expected sibling"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn open_multi_shard_via_non_first_shard() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let write_shard = |idx: usize, tensor_ids: &[usize], metas: &[(&str, u32)]| {
            let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&3u32.to_le_bytes()).unwrap();
            file.write_all(&(tensor_ids.len() as u64).to_le_bytes())
                .unwrap();
            file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
            for (k, v) in metas {
                let kb = k.as_bytes();
                file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(kb).unwrap();
                file.write_all(&4u32.to_le_bytes()).unwrap();
                file.write_all(&v.to_le_bytes()).unwrap();
            }
            for (rel, &tid) in tensor_ids.iter().enumerate() {
                let name = format!("blk.{tid}.ffn_down.weight");
                let nb = name.as_bytes();
                file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(nb).unwrap();
                file.write_all(&2u32.to_le_bytes()).unwrap();
                file.write_all(&2u64.to_le_bytes()).unwrap();
                file.write_all(&2u64.to_le_bytes()).unwrap();
                file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                    .unwrap();
                let off = (rel as u64) * 16;
                file.write_all(&off.to_le_bytes()).unwrap();
            }
            let pos = file.stream_position().unwrap();
            let aligned = pos.div_ceil(32) * 32;
            file.write_all(&vec![0u8; (aligned - pos) as usize])
                .unwrap();
            for &tid in tensor_ids {
                for off in 0..4 {
                    file.write_all(&((tid as f32) + 0.1 * off as f32).to_le_bytes())
                        .unwrap();
                }
            }
            file.flush().unwrap();
            path
        };

        let _p1 = write_shard(1, &[0], &[("split.count", 2), ("split.tensors.count", 2)]);
        let p2 = write_shard(2, &[1], &[("split.count", 2), ("split.tensors.count", 2)]);

        let gguf = GgufFile::open(&p2).unwrap();
        assert_eq!(gguf.shards.len(), 2);
        assert_eq!(gguf.tensor_infos.len(), 2);
    }

    #[test]
    fn open_multi_shard_discovers_via_filename_when_split_count_absent() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let write_shard = |idx: usize, n_tensors: usize, metas: &[(&str, u32)]| {
            let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&3u32.to_le_bytes()).unwrap();
            file.write_all(&(n_tensors as u64).to_le_bytes()).unwrap();
            file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
            for (k, v) in metas {
                let kb = k.as_bytes();
                file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(kb).unwrap();
                file.write_all(&4u32.to_le_bytes()).unwrap();
                file.write_all(&v.to_le_bytes()).unwrap();
            }
            for i in 0..n_tensors {
                let name = format!("blk.{i}.ffn_down.weight");
                let nb = name.as_bytes();
                file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(nb).unwrap();
                file.write_all(&2u32.to_le_bytes()).unwrap();
                file.write_all(&1u64.to_le_bytes()).unwrap();
                file.write_all(&1u64.to_le_bytes()).unwrap();
                file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                    .unwrap();
                file.write_all(&((i as u64) * 4).to_le_bytes()).unwrap();
            }
            let pos = file.stream_position().unwrap();
            let aligned = pos.div_ceil(32) * 32;
            file.write_all(&vec![0u8; (aligned - pos) as usize])
                .unwrap();
            for i in 0..n_tensors {
                file.write_all(&(i as f32).to_le_bytes()).unwrap();
            }
            file.flush().unwrap();
            path
        };

        // No split.count metadata — open must detect via filename pattern
        let p1 = write_shard(1, 1, &[]);
        let _p2 = write_shard(2, 1, &[]);

        let gguf = GgufFile::open(&p1).unwrap();
        assert_eq!(gguf.shards.len(), 2);
        assert_eq!(gguf.tensor_infos.len(), 2);
    }

    #[test]
    fn open_multi_shard_rejects_tensor_count_mismatch() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let write_shard = |idx: usize, n_tensors: usize, metas: &[(&str, u32)]| {
            let path = dir.path().join(format!("m-{idx:0>5}-of-00002.gguf"));
            let mut file = std::fs::File::create(&path).unwrap();
            file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
            file.write_all(&3u32.to_le_bytes()).unwrap();
            file.write_all(&(n_tensors as u64).to_le_bytes()).unwrap();
            file.write_all(&(metas.len() as u64).to_le_bytes()).unwrap();
            for (k, v) in metas {
                let kb = k.as_bytes();
                file.write_all(&(kb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(kb).unwrap();
                file.write_all(&4u32.to_le_bytes()).unwrap();
                file.write_all(&v.to_le_bytes()).unwrap();
            }
            for i in 0..n_tensors {
                let name = format!("blk.{i}.ffn_down.weight");
                let nb = name.as_bytes();
                file.write_all(&(nb.len() as u64).to_le_bytes()).unwrap();
                file.write_all(nb).unwrap();
                file.write_all(&2u32.to_le_bytes()).unwrap();
                file.write_all(&1u64.to_le_bytes()).unwrap();
                file.write_all(&1u64.to_le_bytes()).unwrap();
                file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
                    .unwrap();
                file.write_all(&((i as u64) * 4).to_le_bytes()).unwrap();
            }
            let pos = file.stream_position().unwrap();
            let aligned = pos.div_ceil(32) * 32;
            file.write_all(&vec![0u8; (aligned - pos) as usize])
                .unwrap();
            for i in 0..n_tensors {
                file.write_all(&(i as f32).to_le_bytes()).unwrap();
            }
            file.flush().unwrap();
            path
        };

        // split.tensors.count says 99 but actual is 2
        let p1 = write_shard(1, 1, &[("split.count", 2), ("split.tensors.count", 99)]);
        let _p2 = write_shard(2, 1, &[("split.count", 2), ("split.tensors.count", 99)]);

        let err = match GgufFile::open(&p1) {
            Ok(_) => panic!("expected error for tensor count mismatch"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("tensor count mismatch"),
            "unexpected error: {err}"
        );
    }

    #[test]
    fn multi_shard_tensor_info_accessors() {
        use std::io::{Seek, Write};

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("m-00001-of-00001.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&3u32.to_le_bytes()).unwrap();
        file.write_all(&1u64.to_le_bytes()).unwrap(); // 1 tensor
        file.write_all(&0u64.to_le_bytes()).unwrap(); // 0 metadata
        let name = b"blk.0.ffn_down.weight";
        file.write_all(&(name.len() as u64).to_le_bytes()).unwrap();
        file.write_all(name).unwrap();
        file.write_all(&2u32.to_le_bytes()).unwrap(); // n_dims
        file.write_all(&3u64.to_le_bytes()).unwrap(); // dim0
        file.write_all(&4u64.to_le_bytes()).unwrap(); // dim1
        file.write_all(&crate::quant::ggml::TYPE_F32.to_le_bytes())
            .unwrap();
        file.write_all(&0u64.to_le_bytes()).unwrap(); // offset
        let pos = file.stream_position().unwrap();
        let aligned = pos.div_ceil(32) * 32;
        file.write_all(&vec![0u8; (aligned - pos) as usize])
            .unwrap();
        file.write_all(&[0u8; 3 * 4 * 4]).unwrap(); // 3x4 f32
        file.flush().unwrap();

        let gguf = GgufFile::open(&path).unwrap();
        let info = &gguf.tensor_infos[0];
        assert_eq!(info.name(), "blk.0.ffn_down.weight");
        assert_eq!(info.n_dims(), 2);
        assert_eq!(info.dims(), &[3, 4]);
        assert_eq!(info.tensor_type(), crate::quant::ggml::TYPE_F32);
        assert_eq!(info.offset(), 0);
        assert_eq!(info.shard_idx(), 0);
    }

    #[test]
    fn open_single_rejects_bad_magic() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("bad.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&0xDEADBEEFu32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let err = match GgufFile::open(&path) {
            Ok(_) => panic!("expected error for bad magic"),
            Err(e) => e,
        };
        assert!(format!("{err}").contains("not a GGUF file"), "{err}");
    }

    #[test]
    fn open_single_rejects_unsupported_version() {
        use std::io::Write;
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("v99.gguf");
        let mut file = std::fs::File::create(&path).unwrap();
        file.write_all(&GGUF_MAGIC.to_le_bytes()).unwrap();
        file.write_all(&99u32.to_le_bytes()).unwrap();
        file.flush().unwrap();

        let err = match GgufFile::open(&path) {
            Ok(_) => panic!("expected error for version 99"),
            Err(e) => e,
        };
        assert!(
            format!("{err}").contains("unsupported GGUF version"),
            "{err}"
        );
    }
}
