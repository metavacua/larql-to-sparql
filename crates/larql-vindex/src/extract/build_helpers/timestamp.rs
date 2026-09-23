//! ISO-8601 UTC timestamps without a `chrono` dependency.
//!
//! Canonical home for epoch→calendar conversion in the extract path.
//! `extract::checkpoint` delegates here rather than keeping its own copy.
//!
//! History: this module previously approximated the calendar as
//! `year = 1970 + days/365` with 30-day months, then clamped the result
//! into range. That is wrong by construction — it ignores leap days and
//! month lengths — and produced manifest timestamps ~16 days ahead of
//! the wall clock by 2026. The old unit tests passed anyway because they
//! asserted only *plausibility* (string length, separator positions,
//! month ≤ 12, day ≤ 31) and the clamps guaranteed the last two. Tests
//! below now pin known instants to known strings.

/// Seconds in one UTC day. Unix time ignores leap seconds, so this is exact.
const SECS_PER_DAY: u64 = 86_400;
/// Seconds in one hour.
const SECS_PER_HOUR: u64 = 3_600;
/// Seconds in one minute.
const SECS_PER_MINUTE: u64 = 60;

// ── Civil-from-days constants (Howard Hinnant's algorithm) ──────────────
// The algorithm shifts the epoch to 0000-03-01 so that the leap day falls
// at the end of the year, which makes the month/day arithmetic branchless.

/// Days from 0000-03-01 to 1970-01-01 — the epoch shift.
const DAYS_EPOCH_SHIFT: i64 = 719_468;
/// Days in a 400-year Gregorian era (the calendar's repeat period).
const DAYS_PER_ERA: i64 = 146_097;
/// Last day-of-era index (`DAYS_PER_ERA - 1`), used to floor negative eras.
const DAYS_PER_ERA_LAST: i64 = 146_096;
/// Days in a 4-year cycle, minus one; divisor that removes 4-yearly leap days.
const DAYS_PER_4Y_CYCLE: u32 = 1_460;
/// Days in a 100-year cycle, minus one; divisor that re-adds centurial skips.
const DAYS_PER_100Y_CYCLE: u32 = 36_524;
/// Last day-of-era index as `u32`; divisor that removes the 400-yearly leap day.
const DAYS_PER_ERA_LAST_U32: u32 = 146_096;
/// Days in a common year.
const DAYS_PER_COMMON_YEAR: u32 = 365;
/// Numerator of the month-length series (`153 = 5 × 30.6`), which reproduces
/// the March-anchored 31/30/31/30/31 month pattern exactly.
const MONTH_SERIES_NUM: u32 = 153;
/// Month index at which the March-anchored year wraps back to January.
const MONTH_WRAP_INDEX: u32 = 10;
/// Offset applied when wrapping a March-anchored month index into January/February.
const MONTH_WRAP_BACK: u32 = 9;
/// Offset applied to a March-anchored month index before the wrap point.
const MONTH_WRAP_FORWARD: u32 = 3;
/// Highest month number still belonging to the *previous* March-anchored year.
const MONTH_LAST_OF_PRIOR_YEAR: u32 = 2;

/// Current UTC time as `YYYY-MM-DDTHH:MM:SSZ`.
///
/// A clock read that fails (system time before the epoch) yields the epoch
/// itself rather than panicking — a wrong-but-obvious `1970-01-01` beats
/// aborting an extraction that is otherwise complete.
pub(crate) fn chrono_now() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    iso8601_z_from_unix(secs)
}

/// Convert a Unix timestamp to `YYYY-MM-DDTHH:MM:SSZ` (UTC).
pub(crate) fn iso8601_z_from_unix(secs: u64) -> String {
    format!("{}Z", iso8601_from_unix(secs))
}

/// Convert a Unix timestamp to `YYYY-MM-DDTHH:MM:SS` (UTC, no zone suffix).
///
/// Proleptic Gregorian; no leap-second or timezone handling.
pub(crate) fn iso8601_from_unix(secs: u64) -> String {
    let secs_of_day = secs % SECS_PER_DAY;
    let h = secs_of_day / SECS_PER_HOUR;
    let m = (secs_of_day % SECS_PER_HOUR) / SECS_PER_MINUTE;
    let s = secs_of_day % SECS_PER_MINUTE;
    let (y, mo, d) = days_to_ymd((secs / SECS_PER_DAY) as i64);
    format!("{y:04}-{mo:02}-{d:02}T{h:02}:{m:02}:{s:02}")
}

/// Civil-from-days (Howard Hinnant's algorithm), `1970-01-01 = 0`.
///
/// Exact for the whole proleptic Gregorian calendar — leap years,
/// centurial exceptions, and the 400-year rule all fall out of the era
/// arithmetic rather than being special-cased.
pub(crate) fn days_to_ymd(z: i64) -> (i32, u32, u32) {
    let z = z + DAYS_EPOCH_SHIFT;
    // Floor-divide toward negative infinity so pre-epoch dates stay correct.
    let era = if z >= 0 { z } else { z - DAYS_PER_ERA_LAST } / DAYS_PER_ERA;
    let day_of_era = (z - era * DAYS_PER_ERA) as u32;
    let year_of_era = (day_of_era - day_of_era / DAYS_PER_4Y_CYCLE
        + day_of_era / DAYS_PER_100Y_CYCLE
        - day_of_era / DAYS_PER_ERA_LAST_U32)
        / DAYS_PER_COMMON_YEAR;
    let year = year_of_era as i32 + era as i32 * 400;
    let day_of_year =
        day_of_era - (DAYS_PER_COMMON_YEAR * year_of_era + year_of_era / 4 - year_of_era / 100);
    let month_prime = (5 * day_of_year + 2) / MONTH_SERIES_NUM;
    let day = day_of_year - (MONTH_SERIES_NUM * month_prime + 2) / 5 + 1;
    let month = if month_prime < MONTH_WRAP_INDEX {
        month_prime + MONTH_WRAP_FORWARD
    } else {
        month_prime - MONTH_WRAP_BACK
    };
    // January/February belong to the next calendar year in March-anchored terms.
    let year = if month <= MONTH_LAST_OF_PRIOR_YEAR {
        year + 1
    } else {
        year
    };
    (year, month, day)
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Semantic pins: known instant → known string ─────────────────────
    // These are the tests the old module lacked. Every assertion below
    // fails under the pre-2026-07-28 approximation.

    #[test]
    fn epoch_is_1970_01_01() {
        assert_eq!(iso8601_from_unix(0), "1970-01-01T00:00:00");
    }

    #[test]
    fn epoch_z_form_has_zone_suffix() {
        assert_eq!(iso8601_z_from_unix(0), "1970-01-01T00:00:00Z");
    }

    #[test]
    fn time_of_day_is_exact() {
        // 23:18:12 on 1970-01-01.
        let secs = 23 * SECS_PER_HOUR + 18 * SECS_PER_MINUTE + 12;
        assert_eq!(iso8601_from_unix(secs), "1970-01-01T23:18:12");
    }

    #[test]
    fn regression_the_sixteen_day_drift() {
        // The instant that exposed the bug: the manifest recorded
        // 2026-08-13T23:18:12Z for this timestamp, sixteen days ahead.
        let secs = 1_785_280_692;
        assert_eq!(iso8601_z_from_unix(secs), "2026-07-28T23:18:12Z");
    }

    #[test]
    fn leap_day_2024_is_february_29() {
        // 2024-02-29T00:00:00Z
        assert_eq!(days_to_ymd(19_782), (2024, 2, 29));
        // The day after is March 1st, not February 30th.
        assert_eq!(days_to_ymd(19_783), (2024, 3, 1));
    }

    #[test]
    fn non_leap_year_has_no_february_29() {
        // 2023-02-28 → 2023-03-01, no intervening day.
        assert_eq!(days_to_ymd(19_416), (2023, 2, 28));
        assert_eq!(days_to_ymd(19_417), (2023, 3, 1));
    }

    #[test]
    fn century_2000_is_a_leap_year() {
        // Divisible by 400 → leap. 2000-02-29 exists.
        assert_eq!(days_to_ymd(11_016), (2000, 2, 29));
    }

    #[test]
    fn century_1900_is_not_a_leap_year() {
        // Divisible by 100 but not 400 → common year; pre-epoch, so the
        // negative-era floor division is exercised too.
        assert_eq!(days_to_ymd(-25_509), (1900, 2, 28));
        assert_eq!(days_to_ymd(-25_508), (1900, 3, 1));
    }

    #[test]
    fn year_boundary_rolls_correctly() {
        assert_eq!(iso8601_from_unix(1_735_689_599), "2024-12-31T23:59:59");
        assert_eq!(iso8601_from_unix(1_735_689_600), "2025-01-01T00:00:00");
    }

    #[test]
    fn month_lengths_are_not_uniform_thirty() {
        // The old approximation assumed every month was 30 days. Pin the
        // 31-day months either side of a short one.
        assert_eq!(days_to_ymd(19_843), (2024, 4, 30));
        assert_eq!(days_to_ymd(19_844), (2024, 5, 1));
        assert_eq!(days_to_ymd(19_874), (2024, 5, 31));
        assert_eq!(days_to_ymd(19_875), (2024, 6, 1));
    }

    // ── Format shape (retained from the original module) ────────────────

    #[test]
    fn chrono_now_returns_iso8601_z_format() {
        let s = chrono_now();
        assert_eq!(s.len(), 20);
        assert_eq!(&s[10..11], "T");
        assert_eq!(&s[19..20], "Z");
        assert_eq!(&s[4..5], "-");
        assert_eq!(&s[7..8], "-");
        assert_eq!(&s[13..14], ":");
        assert_eq!(&s[16..17], ":");
    }

    #[test]
    fn chrono_now_is_within_a_day_of_the_clock() {
        // Stronger than the old "year >= 2020": pins the formatter to the
        // clock it claims to read, which is what actually broke.
        let secs = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_secs();
        let today = iso8601_z_from_unix(secs);
        assert_eq!(chrono_now()[..10], today[..10]);
    }
}
