//! USD cost estimate.
//!
//! Deliberately does **not** derive a wall-clock duration from bytes —
//! there's no real throughput measurement anywhere in this codebase to
//! ground that in, and inventing one would be exactly the fabricated
//! number this module exists to avoid. Instead: multiply a real hourly
//! rate by the recipe's own declared `budget.max_wall_minutes` ceiling,
//! so the output is "what the declared budget would cost if fully
//! used", not a runtime prediction.

use serde::Serialize;

use super::executor::ExecutorClass;

/// `docs/dec-funnel-v0.2.md` §7's rate basis for a high-RAM CPU-only
/// rig worker — extraction is CPU/IO-bound (§7.1 of
/// vindex-factory.md), so this is the relevant tier, not a GPU one.
/// `docs/dec-funnel.md:255` confirms the basis is still live.
const LEASED_CPU_HOST_USD_PER_HOUR_LOW: f64 = 0.25;
const LEASED_CPU_HOST_USD_PER_HOUR_HIGH: f64 = 0.50;

/// "Marketplace prices float — treat as ±30%" (dec-funnel-v0.2.md §7).
const RATE_UNCERTAINTY_BAND: f64 = 0.30;

/// Fly ephemeral machine cost for a VF-0-scale build — the spec's own
/// §15.1 estimate is "~£0".
const INLINE_EXECUTOR_USD: f64 = 0.0;

const MINUTES_PER_HOUR: f64 = 60.0;

/// A low/high USD band, not a point estimate — the underlying rate
/// itself is a floating marketplace price.
#[derive(Clone, Copy, Debug, Serialize)]
pub struct CostRange {
    /// Low end of the band.
    pub low_usd: f64,
    /// High end of the band.
    pub high_usd: f64,
}

/// Estimate the cost of fully using `max_wall_minutes` on `executor`.
pub fn estimated_cost_at_declared_wall_time(
    executor: ExecutorClass,
    max_wall_minutes: u32,
) -> CostRange {
    match executor {
        ExecutorClass::Inline => CostRange {
            low_usd: INLINE_EXECUTOR_USD,
            high_usd: INLINE_EXECUTOR_USD,
        },
        ExecutorClass::Leased => {
            let hours = max_wall_minutes as f64 / MINUTES_PER_HOUR;
            CostRange {
                low_usd: LEASED_CPU_HOST_USD_PER_HOUR_LOW * hours * (1.0 - RATE_UNCERTAINTY_BAND),
                high_usd: LEASED_CPU_HOST_USD_PER_HOUR_HIGH * hours * (1.0 + RATE_UNCERTAINTY_BAND),
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inline_executor_is_free() {
        let range = estimated_cost_at_declared_wall_time(ExecutorClass::Inline, 240);
        assert_eq!(range.low_usd, 0.0);
        assert_eq!(range.high_usd, 0.0);
    }

    #[test]
    fn leased_executor_scales_with_declared_wall_minutes() {
        let one_hour = estimated_cost_at_declared_wall_time(ExecutorClass::Leased, 60);
        let four_hours = estimated_cost_at_declared_wall_time(ExecutorClass::Leased, 240);
        assert!(four_hours.low_usd > one_hour.low_usd);
        assert!(four_hours.high_usd > one_hour.high_usd);
        // Scales linearly with declared minutes.
        assert!((four_hours.low_usd - one_hour.low_usd * 4.0).abs() < 1e-9);
    }

    #[test]
    fn leased_low_is_always_below_high() {
        let range = estimated_cost_at_declared_wall_time(ExecutorClass::Leased, 240);
        assert!(range.low_usd < range.high_usd);
    }

    #[test]
    fn zero_minutes_costs_zero() {
        let range = estimated_cost_at_declared_wall_time(ExecutorClass::Leased, 0);
        assert_eq!(range.low_usd, 0.0);
        assert_eq!(range.high_usd, 0.0);
    }
}
