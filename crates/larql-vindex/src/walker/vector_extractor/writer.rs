//! Streaming NDJSON writer + completed-layer scan for resume.

use std::collections::HashSet;
use std::io::{BufRead, BufWriter, Write};
use std::path::Path;

use larql_models::{VectorFileHeader, VectorRecord};

use crate::error::VindexError;

/// Streaming NDJSON writer for vector records.
pub struct VectorWriter {
    writer: BufWriter<std::fs::File>,
    count: usize,
}

impl VectorWriter {
    /// Create a new writer, truncating any existing file.
    pub fn create(path: &Path) -> Result<Self, VindexError> {
        let file = std::fs::File::create(path)?;
        Ok(Self {
            writer: BufWriter::new(file),
            count: 0,
        })
    }

    /// Open an existing file for appending and count existing records.
    pub fn append(path: &Path) -> Result<(Self, usize), VindexError> {
        let existing = if path.exists() {
            let file = std::fs::File::open(path)?;
            let reader = std::io::BufReader::new(file);
            let total_lines = reader.lines().count();
            if total_lines > 0 {
                total_lines - 1
            } else {
                0
            }
        } else {
            0
        };

        let file = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;
        Ok((
            Self {
                writer: BufWriter::new(file),
                count: existing,
            },
            existing,
        ))
    }

    /// Write the metadata header as the first line.
    pub fn write_header(&mut self, header: &VectorFileHeader) -> Result<(), VindexError> {
        serde_json::to_writer(&mut self.writer, header)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        Ok(())
    }

    /// Write a single vector record as one NDJSON line.
    pub fn write_record(&mut self, record: &VectorRecord) -> Result<(), VindexError> {
        serde_json::to_writer(&mut self.writer, record)
            .map_err(|e| VindexError::Parse(e.to_string()))?;
        self.writer.write_all(b"\n")?;
        self.count += 1;
        Ok(())
    }

    pub fn flush(&mut self) -> Result<(), VindexError> {
        self.writer.flush()?;
        Ok(())
    }

    pub fn count(&self) -> usize {
        self.count
    }
}

/// Scan an existing NDJSON file for completed layer numbers.
pub fn scan_completed_layers(path: &Path) -> Result<HashSet<usize>, VindexError> {
    let mut layers = HashSet::new();
    if !path.exists() {
        return Ok(layers);
    }

    let file = std::fs::File::open(path)?;
    let reader = std::io::BufReader::new(file);

    for line in reader.lines() {
        let line = line?;
        if let Some(pos) = line.find("\"layer\":") {
            let rest = &line[pos + 8..];
            let num_str: String = rest.chars().take_while(|c| c.is_ascii_digit()).collect();
            if let Ok(layer) = num_str.parse::<usize>() {
                layers.insert(layer);
            }
        }
    }

    Ok(layers)
}

#[cfg(test)]
mod tests {
    use super::*;
    use larql_models::TopKEntry;

    fn sample_header() -> VectorFileHeader {
        VectorFileHeader {
            _header: true,
            component: "ffn_down".into(),
            model: "test/mock".into(),
            dimension: 4,
            extraction_date: "2026-05-09".into(),
        }
    }

    fn sample_record(layer: usize, feature: usize) -> VectorRecord {
        VectorRecord {
            id: format!("L{layer}_f{feature}"),
            layer,
            feature,
            vector: vec![0.1, 0.2, 0.3, 0.4],
            dim: 4,
            top_token: "the".into(),
            top_token_id: 0,
            c_score: 0.5,
            top_k: vec![TopKEntry {
                token: "the".into(),
                token_id: 0,
                logit: 1.0,
            }],
        }
    }

    #[test]
    fn vector_writer_create_writes_header_and_records() {
        let dir = std::env::temp_dir().join("larql_vex_writer_create");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.jsonl");
        let _ = std::fs::remove_file(&path);

        let mut w = VectorWriter::create(&path).unwrap();
        w.write_header(&sample_header()).unwrap();
        w.write_record(&sample_record(0, 0)).unwrap();
        w.write_record(&sample_record(0, 1)).unwrap();
        w.flush().unwrap();
        assert_eq!(w.count(), 2);

        let text = std::fs::read_to_string(&path).unwrap();
        let lines: Vec<&str> = text.lines().collect();
        assert_eq!(lines.len(), 3);
        assert!(lines[0].contains("\"_header\":true"));
        assert!(lines[1].contains("L0_f0"));
        assert!(lines[2].contains("L0_f1"));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn vector_writer_append_to_missing_file_starts_at_zero() {
        let dir = std::env::temp_dir().join("larql_vex_writer_append_missing");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.jsonl");
        let _ = std::fs::remove_file(&path);

        let (writer, existing) = VectorWriter::append(&path).unwrap();
        assert_eq!(existing, 0);
        assert_eq!(writer.count(), 0);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn vector_writer_append_counts_existing_records() {
        let dir = std::env::temp_dir().join("larql_vex_writer_append_count");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.jsonl");
        let _ = std::fs::remove_file(&path);

        {
            let mut w = VectorWriter::create(&path).unwrap();
            w.write_header(&sample_header()).unwrap();
            for f in 0..3 {
                w.write_record(&sample_record(0, f)).unwrap();
            }
            w.flush().unwrap();
        }

        let (mut w, existing) = VectorWriter::append(&path).unwrap();
        assert_eq!(existing, 3, "should subtract the header line");
        assert_eq!(w.count(), 3);
        w.write_record(&sample_record(0, 99)).unwrap();
        w.flush().unwrap();
        assert_eq!(w.count(), 4);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scan_completed_layers_missing_file_is_empty() {
        let path = std::env::temp_dir().join("larql_vex_scan_missing.jsonl");
        let _ = std::fs::remove_file(&path);
        let layers = scan_completed_layers(&path).unwrap();
        assert!(layers.is_empty());
    }

    #[test]
    fn scan_completed_layers_collects_unique_layer_numbers() {
        let dir = std::env::temp_dir().join("larql_vex_scan_collect");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.jsonl");
        let _ = std::fs::remove_file(&path);

        let mut w = VectorWriter::create(&path).unwrap();
        w.write_header(&sample_header()).unwrap();
        for layer in [0usize, 1, 1, 2, 0] {
            w.write_record(&sample_record(layer, 0)).unwrap();
        }
        w.flush().unwrap();

        let layers = scan_completed_layers(&path).unwrap();
        assert_eq!(layers.len(), 3);
        assert!(layers.contains(&0));
        assert!(layers.contains(&1));
        assert!(layers.contains(&2));

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn scan_completed_layers_ignores_malformed_lines() {
        let dir = std::env::temp_dir().join("larql_vex_scan_malformed");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("out.jsonl");
        std::fs::write(
            &path,
            "{\"_header\":true}\n\
             {\"layer\":7,\"feature\":0}\n\
             not-json-at-all\n\
             {\"feature\":1}\n\
             {\"layer\":42}\n",
        )
        .unwrap();

        let layers = scan_completed_layers(&path).unwrap();
        assert!(layers.contains(&7));
        assert!(layers.contains(&42));
        assert!(!layers.contains(&1));

        std::fs::remove_dir_all(&dir).ok();
    }
}
