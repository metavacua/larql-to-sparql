//! The VINDEX3-only public model reference grammar.
//!
//! Four public forms, parsed by shape alone — no probing the filesystem or
//! network to disambiguate, so parsing stays pure and exhaustively
//! testable:
//!
//! ```text
//! qwen3.8                  registry, default variant
//! qwen3.8:27b-nvfp4        registry, named variant
//! hf://owner/repo[@rev]    explicit HuggingFace, bypasses the registry
//! /explicit/local/path     explicit local directory, bypasses the registry
//! ```
//!
//! # Why `/` is the whole disambiguator
//!
//! A [`ModelName`] structurally cannot contain `/` (see its grammar
//! below). That makes the four forms disjoint by construction: the
//! `hf://` prefix is checked first, and *any* remaining string containing
//! `/` is a local path — never a registry name that merely looks
//! unfamiliar. This is deliberately narrower than `cache::resolve_model`'s
//! three-way heuristic (design doc §1), which guesses at a bare
//! `owner/name` HF shorthand; this grammar has no such guess to make — a
//! shorthand HF reference is written `hf://owner/name`, never inferred
//! from a stray slash.
//!
//! `/` alone under-covers one real shape: a native Windows absolute path
//! (`C:\Users\...`) contains no `/` at all, so on Windows the local-path
//! check also accepts anything [`std::path::Path::is_absolute`] agrees
//! with — a drive-relative string like `C:foo` (no root) still isn't
//! absolute there, so it can't collide with a short registry name that
//! happens to contain a `:`.

use std::path::PathBuf;

use super::error::RegistryError;
use crate::format::huggingface::is_hf_path;

fn valid_component_char(c: char) -> bool {
    c.is_ascii_lowercase() || c.is_ascii_digit() || c == '.' || c == '-'
}

/// Validate one `name`/`variant` component in the context of the whole
/// reference it came from, so the error names what the caller actually
/// typed rather than just the fragment that failed.
fn parse_component(
    component: &str,
    what: &str,
    whole_reference: &str,
) -> Result<String, RegistryError> {
    let malformed = |reason: String| RegistryError::MalformedReference {
        reference: whole_reference.to_string(),
        reason,
    };
    if component.is_empty() {
        return Err(malformed(format!("{what} is empty")));
    }
    if !component.chars().all(valid_component_char) {
        return Err(malformed(format!(
            "{what} '{component}' must be lowercase alphanumeric, '.', or '-' only"
        )));
    }
    let first = component.chars().next().expect("checked non-empty above");
    let last = component.chars().last().expect("checked non-empty above");
    if !first.is_ascii_alphanumeric() || !last.is_ascii_alphanumeric() {
        return Err(malformed(format!(
            "{what} '{component}' must start and end with a letter or digit"
        )));
    }
    Ok(component.to_string())
}

/// An official registry model name — the part before an optional
/// `:variant`.
///
/// Lowercase ASCII letters, digits, `.` and `-` only, starting and ending
/// with an alphanumeric character. Chosen to admit the initiative's own
/// worked example (`qwen3.8`) while excluding every character the other
/// three public forms need to stay unambiguous: `/` (local paths,
/// `hf://`), `:` (the variant separator), whitespace.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ModelName(String);

/// A registry variant name — same grammar as [`ModelName`], different role
/// (see [`RegistryModel::variants`](super::manifest::RegistryModel)).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VariantName(String);

impl ModelName {
    /// Validate a bare model name on its own (outside a full reference).
    pub fn parse(raw: &str) -> Result<Self, RegistryError> {
        parse_component(raw, "model name", raw).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl VariantName {
    /// Validate a bare variant name on its own (outside a full reference).
    pub fn parse(raw: &str) -> Result<Self, RegistryError> {
        parse_component(raw, "variant name", raw).map(Self)
    }

    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for ModelName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl std::fmt::Display for VariantName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A reference that bypasses the registry entirely — the "keep existing
/// explicit legacy/local behaviour working" escape hatch (design doc §7).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ExplicitReference {
    /// `hf://owner/repo` or `hf://owner/repo@revision`.
    HuggingFace {
        repo: String,
        revision: Option<String>,
    },
    /// Any path form containing `/` that is not `hf://`-prefixed.
    Local(PathBuf),
}

/// A parsed public VINDEX3 model reference — the sole entry point into
/// [`super::resolver`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ModelReference {
    /// `name` or `name:variant` — resolved against the registry.
    Registry {
        name: ModelName,
        variant: Option<VariantName>,
    },
    /// Bypasses the registry.
    Explicit(ExplicitReference),
}

impl ModelReference {
    /// Parse one of the four public forms.
    ///
    /// Fails closed: an empty string, stray leading/trailing whitespace,
    /// or extra `:`/`@` separators refuse by name rather than guessing
    /// which field was meant.
    pub fn parse(raw: &str) -> Result<Self, RegistryError> {
        if raw.is_empty() || raw.trim() != raw {
            return Err(RegistryError::MalformedReference {
                reference: raw.to_string(),
                reason: "reference is empty or has leading/trailing whitespace".to_string(),
            });
        }
        if is_hf_path(raw) {
            return parse_hf(raw);
        }
        if raw.contains('/') || PathBuf::from(raw).is_absolute() {
            return Ok(Self::Explicit(ExplicitReference::Local(PathBuf::from(raw))));
        }
        parse_registry(raw)
    }
}

fn parse_hf(raw: &str) -> Result<ModelReference, RegistryError> {
    let body = &raw["hf://".len()..];
    let malformed = |reason: &str| RegistryError::MalformedReference {
        reference: raw.to_string(),
        reason: reason.to_string(),
    };
    if body.is_empty() {
        return Err(malformed("hf:// reference names no repo"));
    }
    let (repo, revision) = match body.split_once('@') {
        Some((repo, rev)) => {
            if rev.is_empty() {
                return Err(malformed("hf:// revision pin ('@...') is empty"));
            }
            (repo, Some(rev.to_string()))
        }
        None => (body, None),
    };
    if repo.matches('/').count() != 1 || repo.starts_with('/') || repo.ends_with('/') {
        return Err(malformed("hf:// repo must be exactly 'owner/name'"));
    }
    Ok(ModelReference::Explicit(ExplicitReference::HuggingFace {
        repo: repo.to_string(),
        revision,
    }))
}

fn parse_registry(raw: &str) -> Result<ModelReference, RegistryError> {
    let mut parts = raw.splitn(3, ':');
    let name_part = parts.next().unwrap_or_default();
    let variant_part = parts.next();
    if parts.next().is_some() {
        return Err(RegistryError::MalformedReference {
            reference: raw.to_string(),
            reason: "more than one ':' — expected 'name' or 'name:variant'".to_string(),
        });
    }
    let name = ModelName(parse_component(name_part, "model name", raw)?);
    let variant = match variant_part {
        Some(v) => Some(VariantName(parse_component(v, "variant name", raw)?)),
        None => None,
    };
    Ok(ModelReference::Registry { name, variant })
}
