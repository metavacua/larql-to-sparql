//! Playback-clock analysis of a frame-timestamp trace — the step-5
//! buffer/underrun measurement surface (`docs/tts-funnel.md` §5).
//!
//! Generation rate alone doesn't decide whether speech is realtime; the
//! DAC consumes one frame every frame interval *forever*, so what
//! matters is the relationship between each frame's arrival time and its
//! playback deadline. Everything here is pure arithmetic over a list of
//! arrival instants: the driver records when frames existed, this module
//! says what a player at the model's frame rate would have experienced.
//!
//! No audio device is involved — this is the simulation half of the
//! step-5 gate. The real ring-buffer runtime (§5's inference thread →
//! codec → PCM ring → callback) consumes the same numbers when it
//! arrives; the module exists so the metric definitions live in one
//! place and are unit-tested rather than embedded in a CLI.

/// Seconds of audio per generated frame — 12.5 Hz. A protocol constant
/// of the MOSS-TTS-Realtime checkpoint family (frame = 1920 samples at
/// 24 kHz), spelled here with that provenance; callers pass it to
/// [`analyze_stream`] explicitly so nothing below assumes one family.
pub const SECONDS_PER_FRAME: f64 = 0.08;

/// What a playback clock would have experienced over one generation.
///
/// All fields derive from the same model: playback consumes one frame
/// per interval starting the moment the first frame exists (plus an
/// optional pre-buffer delay); frame `i`'s deadline is
/// `first_arrival + prebuffer + i * seconds_per_frame`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StreamReport {
    /// The smallest pre-buffer delay (seconds of held-back playback
    /// start) under which no frame misses its deadline. Zero means
    /// playback could start the instant the first frame exists.
    pub min_stable_prebuffer_seconds: f64,
    /// Deadline misses if playback starts cold — first frame, zero
    /// pre-buffer.
    pub cold_start_underruns: usize,
    /// Audio-seconds gained per wall-clock second of sustained
    /// generation (realtime factor minus one): positive means the
    /// buffer grows, negative means it drains.
    pub buffer_growth_per_second: f64,
    /// Median buffer occupancy (seconds of decoded-but-unplayed audio)
    /// at frame arrivals, with the minimum stable pre-buffer applied.
    pub occupancy_p50_seconds: f64,
    /// 95th-percentile buffer occupancy under the same clock.
    pub occupancy_p95_seconds: f64,
}

/// Analyse the arrival instants of a generated frame sequence.
///
/// `instants` are seconds from an arbitrary epoch, one per frame in
/// emission order; `seconds_per_frame` is the playback interval. Needs
/// at least two frames to speak about rates; returns `None` below that.
pub fn analyze_stream(instants: &[f64], seconds_per_frame: f64) -> Option<StreamReport> {
    if instants.len() < 2 {
        return None;
    }
    let first = instants[0];
    let elapsed = instants[instants.len() - 1] - first;
    if elapsed <= 0.0 {
        return None;
    }

    // Frame i's lateness against a cold start: arrival minus deadline.
    // The largest lateness is exactly the pre-buffer that would have
    // absorbed it.
    let lateness = |index: usize, arrival: f64| -> f64 {
        (arrival - first) - index as f64 * seconds_per_frame
    };
    let mut worst_lateness = 0.0f64;
    let mut cold_start_underruns = 0usize;
    for (index, &arrival) in instants.iter().enumerate() {
        let late_by = lateness(index, arrival);
        if late_by > 0.0 && index > 0 {
            cold_start_underruns += 1;
        }
        worst_lateness = worst_lateness.max(late_by);
    }
    let min_stable_prebuffer_seconds = worst_lateness;

    // Occupancy at each arrival under the stable clock: audio produced
    // so far minus audio played so far.
    let playback_start = first + min_stable_prebuffer_seconds;
    let mut occupancy: Vec<f64> = instants
        .iter()
        .enumerate()
        .map(|(index, &arrival)| {
            let produced = (index + 1) as f64 * seconds_per_frame;
            let played = ((arrival - playback_start) / seconds_per_frame)
                .clamp(0.0, index as f64 + 1.0)
                * seconds_per_frame;
            produced - played
        })
        .collect();
    occupancy.sort_by(|a, b| a.partial_cmp(b).expect("occupancy is finite"));
    let percentile = |p: f64| -> f64 {
        let rank = ((occupancy.len() as f64 - 1.0) * p / 100.0).round() as usize;
        occupancy[rank]
    };

    let produced_after_first = (instants.len() - 1) as f64 * seconds_per_frame;
    Some(StreamReport {
        min_stable_prebuffer_seconds,
        cold_start_underruns,
        buffer_growth_per_second: produced_after_first / elapsed - 1.0,
        occupancy_p50_seconds: percentile(50.0),
        occupancy_p95_seconds: percentile(95.0),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    const INTERVAL: f64 = 0.08;

    /// Arrivals at a fixed production interval, starting at an epoch.
    fn steady(epoch: f64, count: usize, production_interval: f64) -> Vec<f64> {
        (0..count)
            .map(|index| epoch + index as f64 * production_interval)
            .collect()
    }

    #[test]
    fn needs_two_frames_and_forward_time() {
        assert_eq!(analyze_stream(&[], INTERVAL), None);
        assert_eq!(analyze_stream(&[1.0], INTERVAL), None);
        assert_eq!(analyze_stream(&[1.0, 1.0], INTERVAL), None);
    }

    #[test]
    fn faster_than_realtime_needs_no_prebuffer_and_grows() {
        // Producing every 40 ms against an 80 ms clock: 2x realtime.
        let report = analyze_stream(&steady(5.0, 51, 0.04), INTERVAL).unwrap();
        assert_eq!(report.min_stable_prebuffer_seconds, 0.0);
        assert_eq!(report.cold_start_underruns, 0);
        assert!((report.buffer_growth_per_second - 1.0).abs() < 1e-9);
        // Occupancy keeps climbing; the median sits mid-run, the p95
        // near the end.
        assert!(report.occupancy_p95_seconds > report.occupancy_p50_seconds);
    }

    #[test]
    fn slower_than_realtime_underruns_from_cold_and_drains() {
        // Producing every 100 ms against an 80 ms clock: every frame
        // after the first misses its cold-start deadline.
        let count = 26;
        let report = analyze_stream(&steady(0.0, count, 0.10), INTERVAL).unwrap();
        assert_eq!(report.cold_start_underruns, count - 1);
        // The last frame is (count-1) * 20 ms late — exactly the
        // pre-buffer that would have hidden it.
        let expected = (count - 1) as f64 * 0.02;
        assert!((report.min_stable_prebuffer_seconds - expected).abs() < 1e-9);
        assert!((report.buffer_growth_per_second - (-0.2)).abs() < 1e-9);
    }

    #[test]
    fn single_spike_sets_the_prebuffer() {
        // Realtime production with one frame arriving 50 ms late.
        let mut instants = steady(2.0, 20, INTERVAL);
        for arrival in instants.iter_mut().skip(10) {
            *arrival += 0.05;
        }
        let report = analyze_stream(&instants, INTERVAL).unwrap();
        assert!((report.min_stable_prebuffer_seconds - 0.05).abs() < 1e-9);
        assert!(report.cold_start_underruns >= 1);
    }

    #[test]
    fn exactly_realtime_holds_steady() {
        let report = analyze_stream(&steady(0.0, 40, INTERVAL), INTERVAL).unwrap();
        assert_eq!(report.min_stable_prebuffer_seconds, 0.0);
        assert_eq!(report.cold_start_underruns, 0);
        assert!(report.buffer_growth_per_second.abs() < 1e-9);
        // With a zero pre-buffer and a lockstep clock, one frame is
        // always in hand at its arrival instant.
        assert!((report.occupancy_p50_seconds - INTERVAL).abs() < 1e-9);
    }
}
