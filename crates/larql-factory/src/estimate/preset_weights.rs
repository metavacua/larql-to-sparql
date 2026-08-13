//! Which coarse [`ComponentBytes`] fields sum to which preset.
//!
//! Coarser than `larql_cli::commands::primary::slice_cmd::preset_parts`'s
//! byte-exact `Part` sets (Gate/DownMeta/Norms/Tokenizer/Manifest/Labels
//! are small metadata this module doesn't model separately — folded
//! into "close enough" or omitted, never invented as their own byte
//! count). Kept in sync by hand against that function's preset names,
//! same as [`crate::validate`]'s `KNOWN_PRESETS` — see that module's
//! doc comment for why a shared registry isn't worth the dependency
//! direction it would require.

use super::bytes::ComponentBytes;

/// One coarse weight component. Deliberately smaller than
/// `slice_cmd::Part` — this module estimates order-of-magnitude bytes,
/// not an exact manifest.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Component {
    Embed,
    Attn,
    Ffn,
    LmHead,
    Router,
}

/// Components included in `preset`, or an empty slice for an unknown
/// preset name (structural validation already rejects those — this
/// just degrades to a zero estimate rather than panicking).
fn preset_components(preset: &str) -> &'static [Component] {
    use Component::*;
    match preset.to_ascii_lowercase().as_str() {
        "full" | "all" => &[Embed, Attn, Ffn, LmHead, Router],
        "client" => &[Embed, Attn],
        "attn" | "attention" => &[Attn],
        "embed" | "embed-server" => &[Embed],
        "server" | "ffn" | "ffn-service" => &[Embed, Ffn],
        // Browse carries gate vectors + down_meta (compact per-feature
        // metadata), not full FFN weight blocks — approximated as
        // embed-only, a deliberate underestimate of the small
        // gate/down_meta contribution rather than double-counting FFN.
        "browse" => &[Embed],
        "router" => &[Router],
        "expert-server" | "expert_server" | "moe-server" => &[Embed, Ffn, Router],
        _ => &[],
    }
}

/// Sum the components `preset` includes.
pub fn estimate_preset_bytes(preset: &str, components: &ComponentBytes) -> u64 {
    preset_components(preset)
        .iter()
        .map(|c| match c {
            Component::Embed => components.embed,
            Component::Attn => components.attn,
            Component::Ffn => components.ffn,
            Component::LmHead => components.lm_head,
            Component::Router => components.router,
        })
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_components() -> ComponentBytes {
        ComponentBytes {
            embed: 100,
            attn: 10,
            ffn: 1000,
            lm_head: 100,
            router: 5,
        }
    }

    #[test]
    fn full_and_all_sum_every_component() {
        let c = sample_components();
        let expected = c.embed + c.attn + c.ffn + c.lm_head + c.router;
        assert_eq!(estimate_preset_bytes("full", &c), expected);
        assert_eq!(estimate_preset_bytes("all", &c), expected);
    }

    #[test]
    fn client_is_embed_plus_attn_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("client", &c), c.embed + c.attn);
    }

    #[test]
    fn attn_preset_is_attn_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("attn", &c), c.attn);
        assert_eq!(estimate_preset_bytes("attention", &c), c.attn);
    }

    #[test]
    fn embed_preset_is_embed_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("embed", &c), c.embed);
        assert_eq!(estimate_preset_bytes("embed-server", &c), c.embed);
    }

    #[test]
    fn server_preset_is_embed_plus_ffn() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("server", &c), c.embed + c.ffn);
    }

    #[test]
    fn browse_preset_is_embed_only_approximation() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("browse", &c), c.embed);
    }

    #[test]
    fn router_preset_is_router_only() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("router", &c), c.router);
    }

    #[test]
    fn expert_server_preset_excludes_attention() {
        let c = sample_components();
        assert_eq!(
            estimate_preset_bytes("expert-server", &c),
            c.embed + c.ffn + c.router
        );
    }

    #[test]
    fn unknown_preset_estimates_to_zero_rather_than_panicking() {
        let c = sample_components();
        assert_eq!(estimate_preset_bytes("bogus", &c), 0);
    }

    #[test]
    fn preset_name_matching_is_case_insensitive() {
        let c = sample_components();
        assert_eq!(
            estimate_preset_bytes("FULL", &c),
            estimate_preset_bytes("full", &c)
        );
    }
}
