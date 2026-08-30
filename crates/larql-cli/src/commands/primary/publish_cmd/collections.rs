//! Collection-level titling and the HF-collection publish step.
//!
//! Everything here derives a human-facing title (model / family /
//! library) from a vindex's `index.json` `model` field, then groups the
//! uploaded repos from a publish run into the corresponding HF
//! collections. VINDEX2-only today — `load_vindex_config` is the VINDEX2
//! config loader; see `super::run`'s VINDEX3-safe-defaults handling for
//! why a VINDEX3 source skips this step by default instead of crashing
//! on it.

use std::path::Path;

use super::upload::StepOutcome;
use super::PublishArgs;

pub(super) fn resolve_collection_list(
    raw: &[String],
) -> Result<Vec<String>, Box<dyn std::error::Error>> {
    if raw.len() == 1 && raw[0].eq_ignore_ascii_case("none") {
        return Ok(Vec::new());
    }
    let mut out = Vec::with_capacity(raw.len());
    for name in raw {
        let lower = name.trim().to_ascii_lowercase();
        match lower.as_str() {
            "model" | "family" | "library" => out.push(lower),
            other => {
                return Err(format!(
                    "invalid collection level '{other}'. Valid: model, family, library, none"
                )
                .into());
            }
        }
    }
    Ok(out)
}

/// Parse `OWNER/NAME` → `OWNER`. Returns an error for bare names so we
/// don't accidentally treat a missing namespace as valid.
pub(super) fn namespace_of(repo: &str) -> Result<&str, Box<dyn std::error::Error>> {
    repo.split_once('/')
        .map(|(ns, _)| ns)
        .ok_or_else(|| format!("--repo must be `OWNER/NAME`, got '{repo}'").into())
}

/// Extract the short model name from whatever `index.json` happens to
/// carry in its `model` field. Handles:
///
///   * `google/gemma-4-31b-it`               → `gemma-4-31b-it`
///   * `/absolute/path/...gemma-4-31b-it/`   → `gemma-4-31b-it`
///   * `.../models--google--gemma-4-31B-it/` → `gemma-4-31B-it` (HF cache layout)
///   * `gemma-4-31b-it`                      → `gemma-4-31b-it`
fn short_model_name(model_field: &str) -> &str {
    // Drop trailing slashes so `rsplit` doesn't return the empty string.
    let trimmed = model_field.trim_end_matches('/');

    // HF cache layout: `.../models--{owner}--{name}/snapshots/{hash}/`
    // At this point the trailing `snapshots/{hash}` is already trimmed
    // by `rsplit` below; the `models--…` directory is what remains.
    let last = trimmed.rsplit('/').next().unwrap_or(trimmed);
    if let Some(rest) = last.strip_prefix("models--") {
        // `google--gemma-4-31B-it` → `gemma-4-31B-it`
        if let Some((_owner, name)) = rest.split_once("--") {
            return name;
        }
        return rest;
    }
    // Walk back up looking for a `models--…` segment (when the tail is a
    // hash directory like `.../snapshots/abc123/`).
    for seg in trimmed.rsplit('/') {
        if let Some(rest) = seg.strip_prefix("models--") {
            if let Some((_owner, name)) = rest.split_once("--") {
                return name;
            }
            return rest;
        }
    }
    last
}

/// Title-case a `-`-separated segment sequence, e.g. `4 31b it` →
/// `4 31b It`. Shared by [`default_model_title`] (all segments) and
/// [`default_family`] (the non-digit prefix only) so the two can't drift
/// on how a segment gets capitalised.
fn title_case_segments(segs: &[&str]) -> String {
    segs.iter()
        .map(|s| {
            let mut chars = s.chars();
            match chars.next() {
                Some(c) => c.to_ascii_uppercase().to_string() + chars.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}

/// Default model title derived from the vindex's `model` field in
/// `index.json`. Title-cases segments separated by `-` so
/// `gemma-4-31b-it` → `Gemma 4 31b It`. Override with `--model-title`
/// when clarity matters.
pub(super) fn default_model_title(model_field: &str) -> String {
    let short = short_model_name(model_field);
    title_case_segments(&short.split('-').collect::<Vec<_>>())
}

/// Default family = prefix of the model id up to (but not including) the
/// first segment that looks like a size/version token — one starting with
/// a digit. `gemma-4-31b-it` → `Gemma`; `gemma-3-4b-it` → `Gemma`;
/// `llama-3-8b-instruct` → `Llama`.
pub(super) fn default_family(model_field: &str) -> String {
    let short = short_model_name(model_field);
    let mut segs: Vec<&str> = Vec::new();
    for seg in short.split('-') {
        if seg
            .chars()
            .next()
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
        {
            break;
        }
        segs.push(seg);
    }
    if segs.is_empty() {
        return short.to_string();
    }
    title_case_segments(&segs)
}

fn note_for_preset(preset: &str) -> &'static str {
    match preset {
        "client" => "2-tier client — attention + embed + norms. Pair with `larql run --ffn URL`.",
        "attn" | "attention" => {
            "3-tier attention client — attn + norms only. Pair with `larql run --embed URL --ffn URL` (ADR-0008)."
        }
        "embed" | "embed-server" => {
            "Embed-server slice — embeddings + tokenizer. Pair with `larql serve --embed-only` (ADR-0008)."
        }
        "server" => "FFN-only slice — pair with `larql serve --ffn-only`.",
        "browse" => "Browse-only slice — DESCRIBE / WALK / SELECT, no forward pass.",
        "router" => "Router slice — MoE router weights only (ADR-0003).",
        "all" => "Full mirror.",
        _ => "Sliced variant.",
    }
}

fn note_for_full() -> &'static str {
    "Canonical full vindex — INFER + DESCRIBE."
}

pub(super) fn build_collections(
    src: &Path,
    args: &PublishArgs,
    uploaded: &[StepOutcome],
    levels: &[String],
) -> Result<Vec<(String, String)>, Box<dyn std::error::Error>> {
    let namespace = namespace_of(&args.repo)?;
    let cfg = larql_vindex::load_vindex_config(src)?;

    let model_title = args
        .model_title
        .clone()
        .unwrap_or_else(|| format!("{} — LARQL Vindex", default_model_title(&cfg.model)));
    let family = args
        .family
        .clone()
        .unwrap_or_else(|| default_family(&cfg.model));
    let family_title = format!("{family} Family — LARQL Vindexes");
    let library_title = args.library_title.clone();

    let items: Vec<larql_vindex::CollectionItem> = uploaded
        .iter()
        .map(|r| larql_vindex::CollectionItem {
            repo_id: r.repo.clone(),
            repo_type: args.repo_type.clone(),
            note: Some(if r.label == "full" {
                note_for_full().into()
            } else {
                note_for_preset(&r.label).into()
            }),
        })
        .collect();

    if args.dry_run {
        // Shouldn't normally hit this path (dry_run returns earlier), but
        // keep the branch so future refactors don't accidentally upload.
        return Ok(Vec::new());
    }

    let mut urls = Vec::new();
    for level in levels {
        let (level_title, description) = match level.as_str() {
            "model" => (
                model_title.clone(),
                format!(
                    "All deployment variants of {} as LARQL vindexes — full, client, server, browse.",
                    default_model_title(&cfg.model)
                ),
            ),
            "family" => (
                family_title.clone(),
                format!("LARQL vindexes for the {family} model family."),
            ),
            "library" => (
                library_title.clone(),
                "Every LARQL vindex in one place — browse, client, server, and full mirrors for each supported model."
                    .to_string(),
            ),
            _ => continue,
        };

        println!(
            "\n→ Updating collection `{}` under `{}`…",
            level_title, namespace
        );
        let url =
            larql_vindex::ensure_collection(namespace, &level_title, Some(&description), &items)?;
        println!("  {url}");
        urls.push((level.clone(), url));
    }
    Ok(urls)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_collection_levels_are_all_three() {
        // Matches the clap default_value on --collections (both read
        // `DEFAULT_COLLECTIONS`, so this can't drift from it). The
        // default publishes to every level so a single run produces the
        // full docs structure (library → family → model).
        let raw: Vec<String> = super::super::DEFAULT_COLLECTIONS
            .split(',')
            .map(String::from)
            .collect();
        let got = resolve_collection_list(&raw).unwrap();
        assert_eq!(got, vec!["model", "family", "library"]);
    }

    #[test]
    fn collection_level_none_disables_all() {
        let got = resolve_collection_list(&["none".into()]).unwrap();
        assert!(got.is_empty());
        // Case-insensitive.
        let got_caps = resolve_collection_list(&["NONE".into()]).unwrap();
        assert!(got_caps.is_empty());
    }

    #[test]
    fn collection_level_invalid_errors() {
        let err = resolve_collection_list(&["world".into()]).unwrap_err();
        assert!(
            err.to_string().contains("invalid collection level"),
            "got: {err}"
        );
    }

    #[test]
    fn collection_level_is_lowercased() {
        let got = resolve_collection_list(&["Model".into(), "FAMILY".into()]).unwrap();
        assert_eq!(got, vec!["model", "family"]);
    }

    #[test]
    fn namespace_of_rejects_bare_name() {
        assert!(namespace_of("chrishayuk/gemma-4-31b").is_ok());
        assert_eq!(
            namespace_of("chrishayuk/gemma-4-31b").unwrap(),
            "chrishayuk"
        );
        assert!(namespace_of("gemma-4-31b").is_err());
    }

    #[test]
    fn default_model_title_strips_hf_namespace() {
        assert_eq!(
            default_model_title("google/gemma-4-31b-it"),
            "Gemma 4 31b It"
        );
        assert_eq!(default_model_title("gemma-3-4b-it"), "Gemma 3 4b It");
        assert_eq!(
            default_model_title("llama-3-70b-instruct"),
            "Llama 3 70b Instruct"
        );
    }

    #[test]
    fn short_model_name_handles_hf_cache_layout() {
        // Absolute paths from the HF cache trim trailing slashes and
        // strip the `models--{owner}--` prefix so we don't end up with
        // empty titles.
        let cached =
            "/Users/me/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/abc123/";
        assert_eq!(short_model_name(cached), "gemma-4-31B-it");

        // Plain path without the `models--` prefix falls back to the
        // last segment, handling trailing slash correctly.
        assert_eq!(short_model_name("/path/to/gemma-3-4b-it/"), "gemma-3-4b-it");

        // HuggingFace `owner/name` format → `name`.
        assert_eq!(short_model_name("google/gemma-4-31b-it"), "gemma-4-31b-it");

        // Already-short name is returned unchanged.
        assert_eq!(short_model_name("gemma-3-4b-it"), "gemma-3-4b-it");
    }

    #[test]
    fn default_model_title_from_hf_cache_path() {
        // Regression guard: this exact layout is what the 31B Q4K vindex
        // produces in its index.json, and the first pass gave an empty
        // string because `rsplit('/').next()` returned "" for trailing `/`.
        let cached =
            "/Users/me/.cache/huggingface/hub/models--google--gemma-4-31B-it/snapshots/abc123/";
        assert_eq!(default_model_title(cached), "Gemma 4 31B It");
        assert_eq!(default_family(cached), "Gemma");
    }

    #[test]
    fn default_family_stops_at_first_digit_segment() {
        assert_eq!(default_family("google/gemma-4-31b-it"), "Gemma");
        assert_eq!(default_family("gemma-3-4b-it"), "Gemma");
        assert_eq!(default_family("llama-3-8b-instruct"), "Llama");
        assert_eq!(default_family("mistral-7b-v0.3"), "Mistral");
    }

    #[test]
    fn default_family_multi_word_prefix_preserved() {
        // e.g. `tiny-llama-1b` → `Tiny Llama` (both non-digit segments kept).
        assert_eq!(default_family("tiny-llama-1b"), "Tiny Llama");
    }

    #[test]
    fn default_family_no_digit_title_cases_all_segments() {
        // When there's no version token (no digit-leading segment), every
        // segment becomes part of the family name — title-cased so the
        // collection header reads cleanly. The key invariant is that we
        // don't produce an empty family string.
        assert_eq!(default_family("my-custom-model"), "My Custom Model");
        assert!(!default_family("singleword").is_empty());
    }

    #[test]
    fn note_for_preset_covers_every_default_slice() {
        // Every slice preset has a hand-written note so the collection
        // card explains the variant. Any future preset wired into
        // `slice_cmd::preset_parts` should also land here.
        assert!(note_for_preset("client").contains("2-tier"));
        assert!(note_for_preset("attn").contains("3-tier"));
        assert!(note_for_preset("attention").contains("3-tier"));
        assert!(note_for_preset("embed").contains("Embed-server"));
        assert!(note_for_preset("embed-server").contains("Embed-server"));
        assert!(note_for_preset("server").contains("FFN-only"));
        assert!(note_for_preset("browse").contains("Browse-only"));
        assert!(note_for_preset("router").contains("MoE"));
        // Unknown preset falls back to a generic note.
        assert_eq!(note_for_preset("zzz"), "Sliced variant.");
    }
}
