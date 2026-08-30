use super::*;

const GPT_OSS: AttentionGeometryQuery = AttentionGeometryQuery {
    head_dim: 64,
    num_q_heads: 64,
    num_kv_heads: 8,
    span: 0,
};
const GLIMMER: AttentionGeometryQuery = AttentionGeometryQuery {
    head_dim: 128,
    num_q_heads: 32,
    num_kv_heads: 2,
    span: 0,
};

fn at(mut q: AttentionGeometryQuery, span: u32) -> AttentionGeometryQuery {
    q.span = span;
    q
}

/// The gpt-oss row IS the KV-B1 policy: for every span the planner under
/// an unset request agrees with `slices_for(Unset, 64, span)`, so moving
/// the decode path onto the planner changes nothing the A/B/C ladder
/// licensed.
#[test]
fn gpt_oss_row_is_the_kv_b1_policy() {
    for span in [
        1u32, 36, 128, 255, 256, 511, 512, 513, 767, 768, 1023, 1024, 2048, 4096,
    ] {
        let planner = choose_attention_geometry(SeqparRequest::Unset, &at(GPT_OSS, span)).slices();
        let policy = slices_for(SeqparRequest::Unset, 64, span);
        assert_eq!(planner, policy, "span {span}");
    }
}

/// The Glimmer row: serial below 1024 (the 512 block was direction-only,
/// so unlicensed), 8 slices — KV-B1's ceiling at head_dim 128 — from
/// 1024 up, where the 1K/2K/4K blocks all had 8 ahead.
#[test]
fn glimmer_row_is_serial_short_and_eight_slices_from_1k() {
    for span in [1u32, 128, 512, 1023] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Unset, &at(GLIMMER, span)),
            AttentionGeometry::Serial,
            "span {span}"
        );
    }
    for span in [1024u32, 2048, 4000, 4096] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Unset, &at(GLIMMER, span)),
            AttentionGeometry::SeqPar { slices: 8 },
            "span {span}"
        );
    }
}

/// An unmeasured geometry runs serial under an unset request — an
/// unmeasured policy is not a default.
#[test]
fn unmeasured_geometry_is_serial_when_unset() {
    let odd = AttentionGeometryQuery {
        head_dim: 64,
        num_q_heads: 16,
        num_kv_heads: 16,
        span: 2048,
    };
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Unset, &odd),
        AttentionGeometry::Serial,
        "same head_dim as a measured row is not the same geometry"
    );
}

#[test]
fn off_is_serial_everywhere() {
    for q in [at(GPT_OSS, 4096), at(GLIMMER, 4096)] {
        assert_eq!(
            choose_attention_geometry(SeqparRequest::Off, &q),
            AttentionGeometry::Serial
        );
    }
}

#[test]
fn explicit_slices_are_honoured_and_bounded() {
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(4), &at(GLIMMER, 512)),
        AttentionGeometry::SeqPar { slices: 4 }
    );
    // 16 x 128 = 2048 threads exceeds the kernel's tg_partial bound: clamped
    // to 8, not refused and not overrun.
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(16), &at(GLIMMER, 512)),
        AttentionGeometry::SeqPar { slices: 8 }
    );
    // One slice partitions nothing.
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(1), &at(GLIMMER, 512)),
        AttentionGeometry::Serial
    );
    // A head_dim past the bound cannot host two slices at all.
    let wide = AttentionGeometryQuery {
        head_dim: 1024,
        num_q_heads: 8,
        num_kv_heads: 8,
        span: 4096,
    };
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Slices(8), &wide),
        AttentionGeometry::Serial
    );
}

/// `auto` is the occupancy heuristic on any geometry: at head_dim 128 the
/// 512/768/1024-thread tiers mean 4/6/8 slices.
#[test]
fn auto_is_the_occupancy_heuristic_on_any_geometry() {
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 256)),
        AttentionGeometry::SeqPar { slices: 4 }
    );
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 768)),
        AttentionGeometry::SeqPar { slices: 6 }
    );
    assert_eq!(
        choose_attention_geometry(SeqparRequest::Auto, &at(GLIMMER, 2048)),
        AttentionGeometry::SeqPar { slices: 8 }
    );
}

#[test]
fn slices_accessor_matches_variant() {
    assert_eq!(AttentionGeometry::Serial.slices(), 0);
    assert_eq!(AttentionGeometry::SeqPar { slices: 6 }.slices(), 6);
}
