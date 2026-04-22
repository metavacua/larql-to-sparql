//! # Date expert
//!
//! Gregorian-calendar date arithmetic via Julian Day Number. Dates are encoded
//! as `{year: int, month: 1..=12, day: 1..=31}` objects.
//!
//! ## Ops
//!
//! - `days_between {from: date, to: date} → int`
//! - `weeks_between {from: date, to: date} → int`
//! - `day_of_week {date: date} → int` (ISO-8601: 1 = Monday, 7 = Sunday)
//! - `add_days {date: date, days: int} → date`
//! - `is_leap_year {year: int} → bool`
//! - `days_in_month {year: int, month: 1..=12} → int`

use expert_interface::{arg_i64, arg_u64, expert_exports, json, Value};

expert_exports!(
    id = "date",
    tier = 2,
    description = "Date arithmetic: days between, day of week, date +/- N days, leap year",
    version = "0.2.0",
    ops = [
        "days_between", "weeks_between",
        "day_of_week",
        "add_days",
        "is_leap_year",
        "days_in_month",
    ],
    dispatch = dispatch
);

fn dispatch(op: &str, args: &Value) -> Option<Value> {
    match op {
        "days_between" => {
            let (ay, am, ad) = ymd(args, "from")?;
            let (by, bm, bd) = ymd(args, "to")?;
            Some(json!((to_jdn(by, bm, bd) - to_jdn(ay, am, ad)).abs()))
        }
        "weeks_between" => {
            let (ay, am, ad) = ymd(args, "from")?;
            let (by, bm, bd) = ymd(args, "to")?;
            Some(json!((to_jdn(by, bm, bd) - to_jdn(ay, am, ad)).abs() / 7))
        }
        "day_of_week" => {
            let (y, m, d) = ymd(args, "date")?;
            Some(json!(day_of_week_index(to_jdn(y, m, d))))
        }
        "add_days" => {
            let (y, m, d) = ymd(args, "date")?;
            let delta = arg_i64(args, "days")?;
            let (ny, nm, nd) = from_jdn(to_jdn(y, m, d) + delta);
            Some(json!({"year": ny, "month": nm, "day": nd}))
        }
        "is_leap_year" => Some(json!(is_leap(arg_i64(args, "year")? as i32))),
        "days_in_month" => {
            let y = arg_i64(args, "year")? as i32;
            let m = arg_u64(args, "month")? as u32;
            Some(json!(days_in_month(y, m)))
        }
        _ => None,
    }
}

/// Extract `{year, month, day}` object into an (i32, u32, u32) triple.
fn ymd(args: &Value, key: &str) -> Option<(i32, u32, u32)> {
    let obj = args.get(key)?;
    let y = obj.get("year")?.as_i64()? as i32;
    let m = obj.get("month")?.as_u64()? as u32;
    let d = obj.get("day")?.as_u64()? as u32;
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) { return None; }
    Some((y, m, d))
}

fn to_jdn(y: i32, m: u32, d: u32) -> i64 {
    let a = (14 - m as i32) / 12;
    let y4 = y + 4800 - a;
    let m4 = m as i32 + 12 * a - 3;
    d as i64 + (153 * m4 as i64 + 2) / 5 + 365 * y4 as i64
        + y4 as i64 / 4 - y4 as i64 / 100 + y4 as i64 / 400 - 32045
}

fn from_jdn(jdn: i64) -> (i32, u32, u32) {
    let a = jdn + 32044;
    let b = (4 * a + 3) / 146097;
    let c = a - (146097 * b) / 4;
    let d = (4 * c + 3) / 1461;
    let e = c - (1461 * d) / 4;
    let m = (5 * e + 2) / 153;
    let day = (e - (153 * m + 2) / 5 + 1) as u32;
    let month = (m + 3 - 12 * (m / 10)) as u32;
    let year = (100 * b + d - 4800 + m / 10) as i32;
    (year, month, day)
}

/// ISO-8601 day-of-week: 1 = Monday, 7 = Sunday.
fn day_of_week_index(jdn: i64) -> i64 {
    jdn.rem_euclid(7) + 1
}

fn is_leap(y: i32) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn days_in_month(y: i32, m: u32) -> u32 {
    match m {
        1 | 3 | 5 | 7 | 8 | 10 | 12 => 31,
        4 | 6 | 9 | 11 => 30,
        2 => if is_leap(y) { 29 } else { 28 },
        _ => 0,
    }
}
