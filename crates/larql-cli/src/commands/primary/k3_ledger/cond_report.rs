//! `larql k3-ledger cond-census` — fetch and rendering for ETC-0A.
//!
//! I/O, excluded from coverage. Every statistic here comes from
//! [`cond_arms`](super::cond_arms) and every price from
//! [`transport_bpw`](super::transport_bpw); this module reads bytes and prints.
//!
//! The one piece of judgement it does carry is the **shape refusal**. The census
//! aligns experts by weight coordinate, so two tensors of different shape are
//! not a data property to report — they mean the wrong tensors were fetched, and
//! coordinate `i` of one is not coordinate `i` of the other. Truncating to the
//! shorter would produce a plausible number from a meaningless alignment, which
//! is the failure mode the whole ladder keeps rediscovering.

use serde_json::json;

use super::args::CondCensusArgs;
use super::cond_arms::{self, CondCensus, ExpertStream};
use super::fetch;
use super::frontier::ServingPremises;
use super::geometry::K3Geometry;
use super::scale_stream::{self, FoldedScales};
use super::transport_bpw::{self as tb, TransportContext};

type R = Result<(), Box<dyn std::error::Error>>;

/// Mutual information below which the zero/non-zero split is withheld: the
/// decomposition is exact, but its *share* is a ratio of two noise terms.
const ZERO_SPLIT_MIN_MI: f64 = 1e-3;

/// Bytes fetched and the streams built from them.
struct Fetched {
    streams: Vec<ExpertStream>,
    bytes_read: u64,
    shape: Vec<u64>,
}

fn expert_index(name: &str) -> Option<usize> {
    name.split(".experts.")
        .nth(1)?
        .split('.')
        .next()?
        .parse()
        .ok()
}

/// Pull the first `rows` rows of one tensor family for experts `0..count`.
fn fetch_rows(
    repo: &fetch::Repo,
    shard: &str,
    hdr_len: u64,
    candidates: &[(String, Vec<u64>, u64, u64)],
    branch: &str,
    count: usize,
    rows: usize,
    kind: &str,
) -> Result<Fetched, Box<dyn std::error::Error>> {
    let mut chosen: Vec<(usize, &(String, Vec<u64>, u64, u64))> = candidates
        .iter()
        .filter(|(name, ..)| name.contains(&format!(".{branch}.")))
        .filter_map(|t| expert_index(&t.0).map(|i| (i, t)))
        .filter(|(i, _)| *i < count)
        .collect();
    chosen.sort_by_key(|(i, _)| *i);

    if chosen.len() < 2 {
        return Err(format!(
            "found {} {kind} tensors for branch '{branch}' in {shard}; the census \
             needs at least 2 — check the branch name (w1/w2/w3) and the shard",
            chosen.len()
        )
        .into());
    }

    let shape = chosen[0].1 .1.clone();
    if let Some((bad, t)) = chosen.iter().find(|(_, t)| t.1 != shape) {
        return Err(format!(
            "expert {bad}'s {kind} tensor is {:?} but expert {}'s is {shape:?} — \
             the census aligns by weight coordinate, so mismatched shapes mean the \
             wrong tensors were fetched, not a property of the bank",
            t.1, chosen[0].0
        )
        .into());
    }

    let row_bytes = *shape.get(1).ok_or("tensor shape has no row length")? as usize;
    let available = *shape.first().unwrap_or(&0) as usize;
    let take_rows = rows.min(available);
    let take = (take_rows * row_bytes) as u64;
    if take == 0 {
        return Err(format!("{kind} tensors for '{branch}' are empty").into());
    }

    let mut streams = Vec::with_capacity(chosen.len());
    let mut bytes_read = 0;
    for (id, (name, _, lo, _)) in &chosen {
        let raw = repo.tensor_bytes_at(shard, hdr_len, *lo, lo + take)?;
        bytes_read += raw.len() as u64;
        eprintln!("  read {:>7.1} KB of {name}", raw.len() as f64 / 1e3);
        streams.push(ExpertStream {
            id: *id,
            nibbles: raw,
        });
    }
    Ok(Fetched {
        streams,
        bytes_read,
        shape,
    })
}

/// Packed bytes hold two weights each; the census works one symbol per element.
fn unpack_nibbles(streams: &mut [ExpertStream]) {
    for s in streams.iter_mut() {
        let mut n = Vec::with_capacity(s.nibbles.len() * 2);
        for &b in &s.nibbles {
            n.push(b & 0xF);
            n.push(b >> 4);
        }
        s.nibbles = n;
    }
}

/// Fold e8m0 bytes into the census alphabet, refusing if any escape.
fn fold_scale_streams(streams: &mut [ExpertStream]) -> Result<FoldedScales, String> {
    let base = streams
        .iter()
        .map(|s| scale_stream::window_base(&s.nibbles))
        .min()
        .unwrap_or(0);
    let mut merged = FoldedScales {
        symbols: Vec::new(),
        base,
        escapes: 0,
        sentinels: 0,
    };
    for s in streams.iter_mut() {
        let folded = scale_stream::fold(&s.nibbles, base);
        merged.escapes += folded.escapes;
        merged.sentinels += folded.sentinels;
        s.nibbles = folded.symbols;
    }
    if !merged.is_valid() {
        return Err(format!(
            "{} scale bytes fell outside a 16-wide window at base {base}; the \
             banked K3 range is 119..122, so this shard disagrees with the \
             transcode scan and the fold is not safe",
            merged.escapes
        ));
    }
    Ok(merged)
}

pub fn run(
    repo: &fetch::Repo,
    geom: &K3Geometry,
    premises: &ServingPremises,
    shard: &str,
    a: &CondCensusArgs,
    as_json: bool,
) -> R {
    let (hdr, hdr_len) = repo.shard_header_with_len(shard)?;

    eprintln!("fetching packed payload for {} experts", a.experts);
    let packed = fetch::packed_expert_tensors(&hdr);
    let mut payload = fetch_rows(
        repo, shard, hdr_len, &packed, &a.branch, a.experts, a.rows, "packed",
    )?;
    unpack_nibbles(&mut payload.streams);

    eprintln!("fetching e8m0 scales for the same rows");
    let scales_hdr = fetch::scale_tensors(&hdr);
    let mut scales = fetch_rows(
        repo,
        shard,
        hdr_len,
        &scales_hdr,
        &a.branch,
        a.experts,
        a.rows,
        "scale",
    )?;
    let fold = fold_scale_streams(&mut scales.streams)
        .map_err(|e| -> Box<dyn std::error::Error> { e.into() })?;

    // Row length in SYMBOLS: packed bytes hold two weights, scale bytes one.
    let payload_row_len = *payload.shape.get(1).unwrap_or(&0) as usize * 2;
    let scale_row_len = *scales.shape.get(1).unwrap_or(&0) as usize;
    let payload_census = cond_arms::run(&payload.streams, payload_row_len, a.fit_fraction, a.seed)
        .ok_or("payload census refused its inputs (ragged, too few, or too short)")?;
    let scale_census = cond_arms::run(&scales.streams, scale_row_len, a.fit_fraction, a.seed)
        .ok_or("scale census refused its inputs")?;

    let ctx = tb::context(geom, premises, a.link_gb_s);
    let scale_bits = scale_stream::bits_per_weight(scale_census.best_conditional_entropy());
    let measured = tb::row_with_parent(
        &ctx,
        premises,
        payload_census.best_conditional_entropy(),
        scale_bits,
        geom.n_experts,
    );
    let realisable_bits = payload_census.best_realisable_bpw();
    let realisable =
        tb::row_with_parent(&ctx, premises, realisable_bits, scale_bits, geom.n_experts);
    let table = tb::kill_table(&ctx, premises, &a.payload_grid, scale_bits);

    if as_json {
        println!(
            "{}",
            serde_json::to_string_pretty(&json!({
                "branch": a.branch,
                "bytes_read": payload.bytes_read + scales.bytes_read,
                "payload_shape": payload.shape,
                "scale_fold": fold,
                "payload": payload_census,
                "scales": scale_census,
                "context": ctx,
                "measured_entropy_floor": measured,
                "measured_realisable": realisable,
                "kill_table": table,
            }))?
        );
        return Ok(());
    }

    render(
        a,
        &payload,
        &scales,
        &fold,
        &payload_census,
        &scale_census,
        &ctx,
        &measured,
        &realisable,
        &table,
        realisable_bits,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn render(
    a: &CondCensusArgs,
    payload: &Fetched,
    scales: &Fetched,
    fold: &FoldedScales,
    census: &CondCensus,
    scale_census: &CondCensus,
    ctx: &TransportContext,
    measured: &tb::TransportRow,
    realisable: &tb::TransportRow,
    table: &[tb::TransportRow],
    realisable_bits: f64,
) {
    println!();
    println!(
        "=== ETC-0A: conditional MXFP4 census, branch {} ===",
        a.branch
    );
    println!(
        "  {} experts, shape {:?}, {} coordinates each ({} fit / {} scored)",
        census.experts, payload.shape, census.coords_total, census.coords_fit, census.coords_scored,
    );
    println!(
        "  {:.1} MB fetched by range request; no checkpoint downloaded",
        (payload.bytes_read + scales.bytes_read) as f64 / 1e6
    );
    println!();

    println!("--- payload arms (bits per weight; scored on held-out coordinates) ---");
    println!(
        "{:<34} {:>8} {:>8} {:>9} {:>8} {:>9} {:>9}",
        "arm", "H(t|c)", "MI", "MI corr", "zero%", "agree", "m/e bpw"
    );
    for arm in &census.arms {
        // A share of near-zero mutual information is a ratio of noise, and
        // prints as a confident-looking percentage. Withhold it instead.
        let zero_share = if arm.mutual_information > ZERO_SPLIT_MIN_MI {
            format!(
                "{:>7.1}%",
                100.0 * arm.mi_zero_involved / arm.mutual_information
            )
        } else {
            format!("{:>8}", "n/a")
        };
        println!(
            "{:<34} {:>8.4} {:>8.4} {:>9.4} {zero_share} {:>9.4} {:>9.4}",
            arm.label,
            arm.conditional_entropy,
            arm.mutual_information,
            arm.mi_bias_corrected,
            arm.agreement_zero_merged,
            arm.match_escape_bpw,
        );
    }
    println!(
        "  marginal control H(t) = {:.4} bits (banked symbol census: 3.7525)",
        census
            .arms
            .first()
            .map(|a| a.marginal_entropy)
            .unwrap_or(0.0)
    );
    println!(
        "  nulls {} — a null showing MI means the arms measure their estimator",
        if census.nulls_clean {
            "CLEAN"
        } else {
            "DIRTY, RESULT REFUSED"
        }
    );
    println!();

    println!("--- does a graph beat a dictionary? (decides whether ETC-0B runs) ---");
    println!(
        "  H(t | prototype)              {:.4} bits",
        census
            .arm(cond_arms::ARM_PROTOTYPE)
            .map(|a| a.conditional_entropy)
            .unwrap_or(f64::NAN)
    );
    println!(
        "  H(t | prototype, parent)      {:.4} bits",
        census.triple_conditional_entropy
    );
    println!(
        "  parent adds over prototype    {:.4} bits   {}",
        census.parent_gain_over_prototype,
        if census.parent_gain_over_prototype < 0.05 {
            "<- nil: a resident dictionary is enough, ETC-0B is redundant"
        } else {
            "<- a specific parent carries its own information"
        }
    );
    println!(
        "  selection beats random by     {:.4} bits   {}",
        census.selection_gain_over_random,
        if census.selection_gain_over_random < 0.05 {
            "<- no compression topology: any expert would have done"
        } else {
            "<- there is a topology to route against"
        }
    );
    println!(
        "  distinct parents chosen       {} of {}   {}",
        census.distinct_parents,
        census.experts,
        if census.distinct_parents <= 2 {
            "<- a hub, i.e. a prototype wearing a graph's hat"
        } else {
            ""
        }
    );
    println!();

    println!("--- scale channel (e8m0, folded at base {}) ---", fold.base);
    println!(
        "  {} scale bytes, {} sentinels, {} escapes",
        scales.bytes_read, fold.sentinels, fold.escapes
    );
    println!(
        "  raw e8m0 as stored             {:.5} bpw",
        scale_stream::RAW_SCALE_BPW
    );
    println!(
        "  AdaptiveSuperblockDelta        {:.5} bpw  <- already banked, NO \
         conditioning; the whole conditional-scale prize is bounded by this",
        tb::banked_scale_bpw()
    );
    println!(
        "  best H(scale | context)        {:.5} bpw  ({:.4} bits/group)",
        scale_stream::bits_per_weight(scale_census.best_conditional_entropy()),
        scale_census.best_conditional_entropy(),
    );
    println!();

    println!("--- what it is worth (R4 applied first) ---");
    println!(
        "  resident arm: dense {:.2} GB + routed {:.2} GB per token -> {:.2} tok/s today",
        ctx.dense_bytes / 1e9,
        ctx.routed_bytes / 1e9,
        ctx.baseline_resident_tok_s,
    );
    println!(
        "  R4 zero-out: routed bytes to ZERO gives {:.2} tok/s = {:.2}x. No expert \
         transport codec can exceed that.",
        ctx.resident_r4_ceiling_tok_s, ctx.expert_side_ceiling,
    );
    println!(
        "  external arm: {:.2} GB/token over a {:.1} GB/s link -> {:.4} tok/s; the \
         gain column below is hit-rate-invariant, the tok/s column is not.",
        ctx.routed_bytes / 1e9,
        ctx.link_gb_s,
        ctx.baseline_external_tok_s,
    );
    println!();
    println!(
        "{:>10} {:>10} {:>10} {:>10} {:>10} {:>10} {:>10}",
        "payload", "eff bpw", "vs 4.25", "vs floor", "res tok/s", "res gain", "ext gain"
    );
    for r in table {
        println!(
            "{:>10.3} {:>10.4} {:>9.2}x {:>9.2}x {:>10.2} {:>9.2}x {:>9.2}x",
            r.payload_bits,
            r.effective_bpw,
            r.compression_vs_native,
            r.compression_vs_exact_floor,
            r.resident_tok_s,
            r.resident_gain,
            r.external_gain,
        );
    }
    println!();
    println!("  MEASURED, entropy floor (serial arithmetic coder, not a kernel):");
    println!(
        "{:>10.3} {:>10.4} {:>9.2}x {:>9.2}x {:>10.2} {:>9.2}x {:>9.2}x",
        measured.payload_bits,
        measured.effective_bpw,
        measured.compression_vs_native,
        measured.compression_vs_exact_floor,
        measured.resident_tok_s,
        measured.resident_gain,
        measured.external_gain,
    );
    println!("  MEASURED, match/escape code (a kernel could run this):");
    println!(
        "{:>10.3} {:>10.4} {:>9.2}x {:>9.2}x {:>10.2} {:>9.2}x {:>9.2}x",
        realisable_bits,
        realisable.effective_bpw,
        realisable.compression_vs_native,
        realisable.compression_vs_exact_floor,
        realisable.resident_tok_s,
        realisable.resident_gain,
        realisable.external_gain,
    );
    println!(
        "  The gap between those two rows is the realisability tax (R11). An \
         entropy is not a kernel."
    );
    println!();
    println!("VERDICT: {}", census.verdict());
    println!(
        "  best conditional entropy {:.4} bits against bands 3.0 / 1.8 / 1.0",
        census.best_conditional_entropy()
    );
}
