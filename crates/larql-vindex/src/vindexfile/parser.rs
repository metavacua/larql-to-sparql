//! Vindexfile parser — reads the declarative build spec.

use std::path::Path;

use crate::error::VindexError;

#[cfg(target_arch = "wasm32")]
use crate::alloc_prelude::*;

/// Parsed Vindexfile.
#[derive(Debug, Clone)]
pub struct Vindexfile {
    /// Top-level directives (shared across all stages).
    pub directives: Vec<VindexfileDirective>,
    /// Named stages (e.g. dev, prod, edge).
    pub stages: Vec<VindexfileStage>,
}

/// A named stage with its own directives.
#[derive(Debug, Clone)]
pub struct VindexfileStage {
    pub name: String,
    pub directives: Vec<VindexfileDirective>,
}

/// A single directive in a Vindexfile.
#[derive(Debug, Clone)]
pub enum VindexfileDirective {
    /// Base vindex to build from.
    From(String),
    /// Apply a patch file.
    Patch(String),
    /// Insert a fact inline.
    Insert {
        entity: String,
        relation: String,
        target: String,
    },
    /// Delete a fact inline.
    Delete {
        entity: String,
        relation: String,
        target: String,
    },
    /// Load probe labels.
    Labels(String),
    /// Expose extract levels (browse, inference, compile).
    Expose(Vec<String>),
}

/// Parse a Vindexfile from a file path.
pub fn parse_vindexfile(path: &Path) -> Result<Vindexfile, VindexError> {
    let content = std::fs::read_to_string(path)?;
    parse_vindexfile_str(&content)
}

/// Parse a Vindexfile from a string.
pub fn parse_vindexfile_str(input: &str) -> Result<Vindexfile, VindexError> {
    let mut directives = Vec::new();
    let mut stages = Vec::new();
    let mut current_stage: Option<VindexfileStage> = None;

    for (line_num, raw_line) in input.lines().enumerate() {
        let line = raw_line.trim();

        // Skip empty lines and comments
        if line.is_empty() || line.starts_with('#') {
            continue;
        }

        // Check for STAGE directive first (before parse_directive)
        if line.to_uppercase().starts_with("STAGE ") {
            if let Some(stage) = current_stage.take() {
                stages.push(stage);
            }
            let name = line[6..].trim().to_string();
            current_stage = Some(VindexfileStage {
                name,
                directives: Vec::new(),
            });
            continue;
        }

        let directive = parse_directive(line, line_num + 1)?;

        if let Some(ref mut stage) = current_stage {
            stage.directives.push(directive);
        } else {
            directives.push(directive);
        }
    }

    // Close last stage
    if let Some(stage) = current_stage {
        stages.push(stage);
    }

    // Validate: must have a FROM
    if !directives
        .iter()
        .any(|d| matches!(d, VindexfileDirective::From(_)))
    {
        return Err(VindexError::Parse(
            "Vindexfile must contain a FROM directive".into(),
        ));
    }

    Ok(Vindexfile { directives, stages })
}

fn parse_directive(line: &str, line_num: usize) -> Result<VindexfileDirective, VindexError> {
    let upper = line.to_uppercase();

    if upper.starts_with("FROM ") {
        let path = line[5..].trim().to_string();
        Ok(VindexfileDirective::From(path))
    } else if upper.starts_with("PATCH ") {
        let path = line[6..].trim().to_string();
        Ok(VindexfileDirective::Patch(path))
    } else if upper.starts_with("INSERT ") {
        parse_insert(&line[7..], line_num)
    } else if upper.starts_with("DELETE ") {
        parse_delete(&line[7..], line_num)
    } else if upper.starts_with("LABELS ") {
        let path = line[7..].trim().to_string();
        Ok(VindexfileDirective::Labels(path))
    } else if upper.starts_with("EXPOSE ") {
        let levels: Vec<String> = line[7..]
            .split_whitespace()
            .map(|s| s.to_lowercase())
            .collect();
        Ok(VindexfileDirective::Expose(levels))
    } else {
        Err(VindexError::Parse(format!(
            "Vindexfile line {}: unknown directive: {}",
            line_num, line
        )))
    }
}

/// Parse INSERT ("entity", "relation", "target")
fn parse_insert(rest: &str, line_num: usize) -> Result<VindexfileDirective, VindexError> {
    let parts = extract_triple(rest, line_num)?;
    Ok(VindexfileDirective::Insert {
        entity: parts.0,
        relation: parts.1,
        target: parts.2,
    })
}

/// The three keys a DELETE condition must bind, in canonical order.
const DELETE_CONDITION_KEYS: [&str; 3] = ["entity", "relation", "target"];

/// Parse DELETE entity = "x" AND relation = "y" AND target = "z"
fn parse_delete(rest: &str, line_num: usize) -> Result<VindexfileDirective, VindexError> {
    // Support both tuple form and condition form
    if rest.trim().starts_with('(') {
        let parts = extract_triple(rest, line_num)?;
        return Ok(VindexfileDirective::Delete {
            entity: parts.0,
            relation: parts.1,
            target: parts.2,
        });
    }

    // Parse key=value pairs: entity = "x" AND relation = "y" AND target = "z".
    // Every key must appear exactly once — a DELETE that silently accepted a
    // missing key would match (and delete) far more than the author wrote.
    let mut values: [Option<String>; 3] = [None, None, None];

    for part in rest.split(" AND ") {
        let part = part.trim();
        let Some((key, val)) = part.split_once('=') else {
            return Err(VindexError::Parse(format!(
                "Vindexfile line {line_num}: DELETE condition {part:?} is not key = \"value\""
            )));
        };
        let key = key.trim().to_lowercase();
        let val = unquote(val.trim());
        let Some(slot) = DELETE_CONDITION_KEYS.iter().position(|k| *k == key) else {
            return Err(VindexError::Parse(format!(
                "Vindexfile line {line_num}: unknown DELETE condition key {key:?} \
                 (expected {DELETE_CONDITION_KEYS:?})"
            )));
        };
        if values[slot].is_some() {
            return Err(VindexError::Parse(format!(
                "Vindexfile line {line_num}: duplicate DELETE condition key {key:?}"
            )));
        }
        values[slot] = Some(val);
    }

    let missing: Vec<&str> = DELETE_CONDITION_KEYS
        .iter()
        .zip(values.iter())
        .filter(|(_, v)| v.is_none())
        .map(|(k, _)| *k)
        .collect();
    if !missing.is_empty() {
        return Err(VindexError::Parse(format!(
            "Vindexfile line {line_num}: DELETE condition missing key(s) {missing:?}"
        )));
    }

    let [entity, relation, target] = values.map(|v| v.expect("all keys checked present"));
    Ok(VindexfileDirective::Delete {
        entity,
        relation,
        target,
    })
}

/// Extract a parenthesised triple: ("a", "b", "c").
///
/// Splitting is quote-aware — a comma inside a double-quoted value
/// (`("Acme, Inc", "hq", "London")`) does not separate fields.
fn extract_triple(s: &str, line_num: usize) -> Result<(String, String, String), VindexError> {
    let s = s.trim();
    let inner = s.trim_start_matches('(').trim_end_matches(')');

    let parts = split_quoted_csv(inner).ok_or_else(|| {
        VindexError::Parse(format!(
            "Vindexfile line {line_num}: unbalanced quotes in tuple"
        ))
    })?;
    if parts.len() != 3 {
        return Err(VindexError::Parse(format!(
            "Vindexfile line {}: expected 3 values in tuple, got {}",
            line_num,
            parts.len()
        )));
    }

    Ok((unquote(&parts[0]), unquote(&parts[1]), unquote(&parts[2])))
}

/// Split on commas that are *outside* double-quoted spans. Returns
/// `None` when the input ends inside an open quote (truncated value).
fn split_quoted_csv(inner: &str) -> Option<Vec<String>> {
    let mut parts = Vec::new();
    let mut current = String::new();
    let mut in_quotes = false;
    for c in inner.chars() {
        match c {
            '"' => {
                in_quotes = !in_quotes;
                current.push(c);
            }
            ',' if !in_quotes => {
                parts.push(std::mem::take(&mut current));
            }
            _ => current.push(c),
        }
    }
    if in_quotes {
        return None;
    }
    parts.push(current);
    Some(parts)
}

/// Trim whitespace, then strip one matched pair of surrounding double
/// quotes (an interior quote or comma survives intact).
fn unquote(s: &str) -> String {
    let s = s.trim();
    s.strip_prefix('"')
        .and_then(|rest| rest.strip_suffix('"'))
        .unwrap_or(s)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_minimal_vindexfile() {
        let input = r#"
# Base model
FROM ./base.vindex
PATCH ./patch1.vlp
"#;
        let vf = parse_vindexfile_str(input).unwrap();
        assert_eq!(vf.directives.len(), 2);
        assert!(matches!(&vf.directives[0], VindexfileDirective::From(p) if p == "./base.vindex"));
        assert!(matches!(&vf.directives[1], VindexfileDirective::Patch(p) if p == "./patch1.vlp"));
        assert!(vf.stages.is_empty());
    }

    #[test]
    fn parse_full_vindexfile() {
        let input = r#"
FROM hf://chrishayuk/gemma-3-4b-it-vindex
PATCH hf://medical-ai/drug-interactions@2.1.0
PATCH ./patches/company-facts.vlp
INSERT ("Acme Corp", "headquarters", "London")
INSERT ("Acme Corp", "ceo", "Jane Smith")
DELETE entity = "Acme Corp" AND relation = "competitor" AND target = "WrongCo"
LABELS hf://chrishayuk/gemma-3-4b-it-labels@latest
EXPOSE browse inference
"#;
        let vf = parse_vindexfile_str(input).unwrap();
        assert_eq!(vf.directives.len(), 8);

        // Check FROM
        assert!(
            matches!(&vf.directives[0], VindexfileDirective::From(p) if p.starts_with("hf://"))
        );

        // Check INSERT
        assert!(
            matches!(&vf.directives[3], VindexfileDirective::Insert { entity, .. } if entity == "Acme Corp")
        );

        // Check DELETE
        assert!(
            matches!(&vf.directives[5], VindexfileDirective::Delete { target, .. } if target == "WrongCo")
        );

        // Check EXPOSE
        if let VindexfileDirective::Expose(levels) = &vf.directives[7] {
            assert_eq!(levels, &["browse", "inference"]);
        } else {
            panic!("expected Expose");
        }
    }

    #[test]
    fn parse_stages() {
        let input = r#"
FROM ./base.vindex
PATCH ./shared-fix.vlp

STAGE dev
  PATCH ./experimental.vlp
  EXPOSE browse inference compile

STAGE prod
  PATCH ./production.vlp
  EXPOSE browse inference

STAGE edge
  EXPOSE browse
"#;
        let vf = parse_vindexfile_str(input).unwrap();
        assert_eq!(vf.directives.len(), 2); // FROM + shared PATCH
        assert_eq!(vf.stages.len(), 3);
        assert_eq!(vf.stages[0].name, "dev");
        assert_eq!(vf.stages[0].directives.len(), 2);
        assert_eq!(vf.stages[1].name, "prod");
        assert_eq!(vf.stages[1].directives.len(), 2);
        assert_eq!(vf.stages[2].name, "edge");
        assert_eq!(vf.stages[2].directives.len(), 1);
    }

    #[test]
    fn parse_delete_tuple_form() {
        let input = r#"
FROM ./base.vindex
DELETE ("Entity", "relation", "target")
"#;
        let vf = parse_vindexfile_str(input).unwrap();
        assert!(matches!(&vf.directives[1],
            VindexfileDirective::Delete { entity, relation, target }
            if entity == "Entity" && relation == "relation" && target == "target"
        ));
    }

    #[test]
    fn missing_from_is_error() {
        let input = r#"
PATCH ./patch.vlp
"#;
        assert!(parse_vindexfile_str(input).is_err());
    }

    // ── quoted-comma + condition-form hardening ─────────────────────

    /// A comma inside a quoted value is part of the value, not a field
    /// separator. Regression: `extract_triple` used to split on bare
    /// commas, so `"Acme, Inc"` mis-parsed as two fields.
    #[test]
    fn insert_preserves_quoted_comma() {
        let input = "FROM ./b\nINSERT (\"Acme, Inc\", \"hq\", \"London\")\n";
        let vf = parse_vindexfile_str(input).unwrap();
        assert!(matches!(&vf.directives[1],
            VindexfileDirective::Insert { entity, relation, target }
            if entity == "Acme, Inc" && relation == "hq" && target == "London"
        ));
    }

    #[test]
    fn insert_rejects_unbalanced_quotes() {
        let input = "FROM ./b\nINSERT (\"Acme, \"hq\", \"London\")\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        assert!(err.to_string().contains("unbalanced quotes"), "{err}");
    }

    #[test]
    fn insert_rejects_wrong_arity_even_with_quoted_commas() {
        let input = "FROM ./b\nINSERT (\"a\", \"b\")\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        assert!(err.to_string().contains("expected 3 values"), "{err}");
    }

    /// DELETE condition form must bind all three keys — a missing key
    /// used to be silently accepted as an empty match-anything string.
    #[test]
    fn delete_condition_missing_key_is_error() {
        let input = "FROM ./b\nDELETE entity = \"Acme\" AND relation = \"ceo\"\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("missing key"), "{msg}");
        assert!(msg.contains("target"), "{msg}");
    }

    #[test]
    fn delete_condition_unknown_key_is_error() {
        let input = "FROM ./b\nDELETE entity = \"A\" AND relation = \"r\" AND victim = \"t\"\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        assert!(
            err.to_string().contains("unknown DELETE condition key"),
            "{err}"
        );
    }

    #[test]
    fn delete_condition_duplicate_key_is_error() {
        let input = "FROM ./b\nDELETE entity = \"A\" AND entity = \"B\" AND target = \"t\"\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        assert!(
            err.to_string().contains("duplicate DELETE condition key"),
            "{err}"
        );
    }

    #[test]
    fn delete_condition_malformed_clause_is_error() {
        let input = "FROM ./b\nDELETE entity \"Acme\"\n";
        let err = parse_vindexfile_str(input).unwrap_err();
        assert!(err.to_string().contains("not key = \"value\""), "{err}");
    }

    /// Condition-form values keep quoted commas too.
    #[test]
    fn delete_condition_preserves_quoted_comma_value() {
        let input = "FROM ./b\nDELETE entity = \"Acme, Inc\" AND relation = \"hq\" AND target = \"London\"\n";
        let vf = parse_vindexfile_str(input).unwrap();
        assert!(matches!(&vf.directives[1],
            VindexfileDirective::Delete { entity, .. } if entity == "Acme, Inc"
        ));
    }

    #[test]
    fn comments_and_blank_lines_ignored() {
        let input = r#"
# This is a comment
FROM ./base.vindex

# Another comment
PATCH ./patch.vlp
"#;
        let vf = parse_vindexfile_str(input).unwrap();
        assert_eq!(vf.directives.len(), 2);
    }
}
