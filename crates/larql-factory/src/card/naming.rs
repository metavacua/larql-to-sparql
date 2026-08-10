//! §9's Hub repo naming: `{owner}/{repo_template}` with
//! `{model_slug}`/`{slice_suffix}` placeholders filled in. Lives
//! outside `body/` because the eventual PUBLISH stage (§7) needs the
//! same resolution, not just the card.

#[cfg(target_arch = "wasm32")]
use crate::prelude::*;

use crate::HubPublish;

const MODEL_SLUG_PLACEHOLDER: &str = "{model_slug}";
const SLICE_SUFFIX_PLACEHOLDER: &str = "{slice_suffix}";

/// Resolve `hub.repo_template` into a full `owner/repo` name.
/// `slice_suffix` is empty for the full/unsliced output, or
/// `-{preset}` for a slice (§9: `{owner}/{model_slug}-vindex` for full,
/// `{owner}/{model_slug}-vindex-{preset}` for slices).
pub fn hub_repo_name(hub: &HubPublish, model_slug: &str, slice_suffix: &str) -> String {
    let repo = hub
        .repo_template
        .replace(MODEL_SLUG_PLACEHOLDER, model_slug)
        .replace(SLICE_SUFFIX_PLACEHOLDER, slice_suffix);
    format!("{}/{repo}", hub.owner)
}

/// The `-{preset}` suffix for a slice, or empty for the full output.
pub fn slice_suffix(preset: &str) -> String {
    if preset == "full" {
        String::new()
    } else {
        format!("-{preset}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hub() -> HubPublish {
        HubPublish {
            owner: "chrishayuk".into(),
            repo_template: "{model_slug}-vindex{slice_suffix}".into(),
            repo_type: "dataset".into(),
            collection: None,
            tags: vec![],
            visibility: "private-until-verified".into(),
        }
    }

    #[test]
    fn full_output_has_no_suffix() {
        assert_eq!(slice_suffix("full"), "");
        assert_eq!(
            hub_repo_name(&hub(), "gemma-3-4b-it", &slice_suffix("full")),
            "chrishayuk/gemma-3-4b-it-vindex"
        );
    }

    #[test]
    fn slice_output_gets_a_preset_suffix() {
        assert_eq!(slice_suffix("client"), "-client");
        assert_eq!(
            hub_repo_name(&hub(), "gemma-3-4b-it", &slice_suffix("client")),
            "chrishayuk/gemma-3-4b-it-vindex-client"
        );
    }

    #[test]
    fn honors_a_custom_template() {
        let mut h = hub();
        h.repo_template = "vindexes/{model_slug}{slice_suffix}".into();
        assert_eq!(
            hub_repo_name(&h, "tiny", &slice_suffix("browse")),
            "chrishayuk/vindexes/tiny-browse"
        );
    }
}
