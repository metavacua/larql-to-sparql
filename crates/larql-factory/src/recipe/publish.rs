//! `spec.publish` — Hub and R2 publication targets.

use serde::{Deserialize, Serialize};

/// Publication targets. Not part of `build_id`.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Publish {
    /// HuggingFace Hub target.
    pub hub: HubPublish,
    /// R2 mirror target.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub mirror: Option<MirrorPublish>,
}

/// Hub publication target.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct HubPublish {
    /// Hub namespace owner.
    pub owner: String,
    /// Repo name template, e.g. `"{model_slug}-vindex{slice_suffix}"`.
    pub repo_template: String,
    /// `dataset` or `model` — PIN §13.1 open, both accepted structurally.
    pub repo_type: String,
    /// Collection to add the repo to at RELEASE.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub collection: Option<String>,
    /// Hub tags applied to the repo.
    #[serde(default)]
    pub tags: Vec<String>,
    /// Visibility policy. Expected: `private-until-verified`, `private`,
    /// or `public`.
    pub visibility: String,
}

/// R2 mirror target — the durable, zero-egress source of truth for bytes.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct MirrorPublish {
    /// R2 bucket name.
    pub r2_bucket: String,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_hub() -> HubPublish {
        HubPublish {
            owner: "chrishayuk".into(),
            repo_template: "{model_slug}-vindex{slice_suffix}".into(),
            repo_type: "dataset".into(),
            collection: Some("larql/vindex-v1".into()),
            tags: vec!["vindex".into()],
            visibility: "private-until-verified".into(),
        }
    }

    #[test]
    fn mirror_omitted_when_absent() {
        let p = Publish {
            hub: sample_hub(),
            mirror: None,
        };
        let json = serde_json::to_string(&p).unwrap();
        assert!(!json.contains("mirror"));
    }

    #[test]
    fn round_trips_with_mirror() {
        let p = Publish {
            hub: sample_hub(),
            mirror: Some(MirrorPublish {
                r2_bucket: "chuk-vindex".into(),
            }),
        };
        let json = serde_json::to_string(&p).unwrap();
        let back: Publish = serde_json::from_str(&json).unwrap();
        assert_eq!(back.hub.owner, "chrishayuk");
        assert_eq!(back.mirror.unwrap().r2_bucket, "chuk-vindex");
    }
}
