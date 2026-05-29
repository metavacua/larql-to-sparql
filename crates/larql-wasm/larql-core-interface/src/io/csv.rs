//! CSV I/O for knowledge graphs.
//!
//! Format: subject,relation,object,confidence,source

use std::io::Write;
use std::path::Path;

use crate::core::edge::Edge;
use crate::core::enums::SourceType;
use crate::core::graph::{Graph, GraphError};

const CSV_HEADER: [&str; 5] = ["subject", "relation", "object", "confidence", "source"];

/// Load a graph from CSV. Expected columns: subject,relation,object,confidence,source
pub fn load_csv(path: impl AsRef<Path>) -> Result<Graph, GraphError> {
    let contents = std::fs::read_to_string(path)?;
    let mut graph = Graph::new();

    for (i, fields) in parse_csv_records(&contents)?.into_iter().enumerate() {
        if fields.iter().all(|f| f.trim().is_empty()) {
            continue;
        }
        if i == 0 && fields.first().is_some_and(|f| f.trim() == CSV_HEADER[0]) {
            continue;
        }

        if fields.len() < 3 {
            continue;
        }

        let subject = fields[0].as_str();
        let relation = fields[1].as_str();
        let object = fields[2].as_str();
        let confidence: f64 = fields
            .get(3)
            .and_then(|s| s.trim().parse().ok())
            .unwrap_or(1.0);
        let source = fields
            .get(4)
            .map(|s| parse_source(s.trim()))
            .unwrap_or(SourceType::Unknown);

        graph.add_edge(
            Edge::new(subject, relation, object)
                .with_confidence(confidence)
                .with_source(source),
        );
    }

    Ok(graph)
}

/// Save a graph to CSV.
pub fn save_csv(graph: &Graph, path: impl AsRef<Path>) -> Result<(), GraphError> {
    let mut file = std::fs::File::create(path)?;
    writeln!(file, "{}", CSV_HEADER.join(","))?;
    for edge in graph.edges() {
        write_csv_field(&mut file, &edge.subject)?;
        write!(file, ",")?;
        write_csv_field(&mut file, &edge.relation)?;
        write!(file, ",")?;
        write_csv_field(&mut file, &edge.object)?;
        write!(file, ",")?;
        write_csv_field(&mut file, &edge.confidence.to_string())?;
        write!(file, ",")?;
        write_csv_field(&mut file, edge.source.as_str())?;
        writeln!(file)?;
    }
    Ok(())
}

fn parse_source(s: &str) -> SourceType {
    match s {
        "parametric" => SourceType::Parametric,
        "document" => SourceType::Document,
        "installed" => SourceType::Installed,
        "wikidata" => SourceType::Wikidata,
        "manual" => SourceType::Manual,
        _ => SourceType::Unknown,
    }
}

fn write_csv_field(mut w: impl Write, field: &str) -> std::io::Result<()> {
    if field.contains(',') || field.contains('"') || field.contains('\n') || field.contains('\r') {
        write!(w, "\"")?;
        for ch in field.chars() {
            if ch == '"' {
                write!(w, "\"\"")?;
            } else {
                write!(w, "{ch}")?;
            }
        }
        write!(w, "\"")?;
    } else {
        write!(w, "{field}")?;
    }
    Ok(())
}

fn parse_csv_records(input: &str) -> Result<Vec<Vec<String>>, GraphError> {
    let mut records = Vec::new();
    let mut record = Vec::new();
    let mut field = String::new();
    let mut chars = input.chars().peekable();
    let mut in_quotes = false;

    while let Some(ch) = chars.next() {
        if in_quotes {
            match ch {
                '"' => {
                    if chars.peek() == Some(&'"') {
                        field.push('"');
                        chars.next();
                    } else {
                        in_quotes = false;
                    }
                }
                _ => field.push(ch),
            }
            continue;
        }

        match ch {
            '"' if field.is_empty() => in_quotes = true,
            '"' => {
                return Err(GraphError::Deserialize(
                    "unexpected quote in unquoted CSV field".to_string(),
                ));
            }
            ',' => {
                record.push(std::mem::take(&mut field));
            }
            '\n' => {
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            '\r' => {
                if chars.peek() == Some(&'\n') {
                    chars.next();
                }
                record.push(std::mem::take(&mut field));
                records.push(std::mem::take(&mut record));
            }
            _ => field.push(ch),
        }
    }

    if in_quotes {
        return Err(GraphError::Deserialize(
            "unterminated quoted CSV field".to_string(),
        ));
    }

    if !field.is_empty() || !record.is_empty() {
        record.push(field);
        records.push(record);
    }

    Ok(records)
}
