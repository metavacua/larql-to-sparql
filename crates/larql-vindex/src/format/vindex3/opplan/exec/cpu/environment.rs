//! Whether this machine may be measured on at all.
//!
//! Three contamination events in one session, each caught by accident
//! rather than by checking:
//!
//! ```text
//!   Adobe Creative Cloud at 96% CPU   Q8 read 4x slower than BF16,
//!                                     which is impossible
//!   switched to BATTERY mid-session   a `pmset` check made for another
//!                                     reason
//!   XProtect at 15.9%, load 14.38     BF16 decode read 793 ms against a
//!                                     stable 449-459
//! ```
//!
//! Every one was caught because a number was ABSURD. Contamination that
//! produced merely plausible numbers would have gone into the cost model
//! unchallenged, and at this point plausible contamination is a larger
//! risk than a small sample.
//!
//! So this refuses rather than warns. **An invalid environment produces
//! no result, not a result with a caveat attached** — a caveat survives
//! about as long as the paragraph it is written in, and the number
//! outlives it.
//!
//! The thresholds are not physically meaningful and are not meant to be.
//! They are the line between "quiet enough that the measurement is about
//! LARQL" and "something else is using this machine".

use std::fmt;

/// Load average above which the machine is not quiet.
///
/// Clean runs in this programme sat at 1.86; the contaminated one read
/// 14.38. Anywhere in between is a judgement, and this is the line.
const MAX_LOAD: f64 = 2.5;

/// Total non-LARQL CPU, as a percentage of one core.
const MAX_BACKGROUND_CPU: f64 = 40.0;

/// The largest single foreign process, as a percentage of one core.
///
/// Separate from the total because one process at 96% and forty at 1%
/// are different problems, and the first is the one that moved a
/// measurement by 4x.
const MAX_SINGLE_PROCESS_CPU: f64 = 12.0;

/// When the machine is being checked.
///
/// The two are not the same question. `loadavg` counts THIS process, and
/// a model open is 16 seconds of I/O and quantisation across every core —
/// so by the time a measurement is about to start, LARQL has raised the
/// one-minute average itself. Checking load after that is partly asking
/// whether LARQL recently worked hard, which it always has.
///
/// So load gates ADMISSION, before any work; afterwards the question is
/// only whether something ELSE arrived, which external CPU answers
/// without counting us.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Phase {
    /// Before the model is opened: every check applies.
    BeforeWork,
    /// After LARQL's own load phase: external signals only.
    AfterWork,
}

/// Why a machine was refused.
#[derive(Debug, Clone, PartialEq)]
pub enum Disqualifier {
    OnBattery,
    Load {
        average: f64,
        limit: f64,
    },
    Background {
        percent: f64,
        limit: f64,
    },
    Process {
        name: String,
        percent: f64,
        limit: f64,
    },
}

impl fmt::Display for Disqualifier {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::OnBattery => write!(
                f,
                "on battery power — core frequency policy differs from AC and is not the \
                 machine any earlier number was taken on"
            ),
            Self::Load { average, limit } => write!(
                f,
                "load average {average:.2} exceeds {limit:.1} — something else is running"
            ),
            Self::Background { percent, limit } => {
                write!(f, "background CPU {percent:.0}% exceeds {limit:.0}%")
            }
            Self::Process {
                name,
                percent,
                limit,
            } => write!(
                f,
                "`{name}` is using {percent:.0}% of a core, over the {limit:.0}% limit"
            ),
        }
    }
}

/// What the machine looks like right now.
#[derive(Debug, Clone, Default)]
pub struct Environment {
    pub on_ac: Option<bool>,
    pub load: Option<f64>,
    pub background_cpu: Option<f64>,
    pub busiest: Option<(String, f64)>,
}

impl Environment {
    /// Read the machine.
    ///
    /// Every field is optional because a platform that does not report
    /// one must not be refused for it — an unknown is not a violation,
    /// and `describe` says which checks actually ran.
    pub fn read() -> Self {
        Self {
            on_ac: on_ac_power(),
            load: load_average(),
            background_cpu: process_cpu().map(|(total, _)| total),
            busiest: process_cpu().and_then(|(_, busiest)| busiest),
        }
    }

    /// Every reason this machine may not be measured on.
    ///
    /// Empty means eligible — including on a platform that reports
    /// nothing, where the honest answer is that nothing disqualified it
    /// rather than that it is quiet.
    pub fn disqualifiers(&self, phase: Phase) -> Vec<Disqualifier> {
        let mut out = Vec::new();
        if self.on_ac == Some(false) {
            out.push(Disqualifier::OnBattery);
        }
        if let Some(average) = self.load {
            if phase == Phase::BeforeWork && average > MAX_LOAD {
                out.push(Disqualifier::Load {
                    average,
                    limit: MAX_LOAD,
                });
            }
        }
        if let Some(percent) = self.background_cpu {
            if percent > MAX_BACKGROUND_CPU {
                out.push(Disqualifier::Background {
                    percent,
                    limit: MAX_BACKGROUND_CPU,
                });
            }
        }
        if let Some((name, percent)) = &self.busiest {
            if *percent > MAX_SINGLE_PROCESS_CPU {
                out.push(Disqualifier::Process {
                    name: name.clone(),
                    percent: *percent,
                    limit: MAX_SINGLE_PROCESS_CPU,
                });
            }
        }
        out
    }

    /// One line per check, including the ones that could not run.
    pub fn describe(&self) -> String {
        let say = |label: &str, value: Option<String>| match value {
            Some(v) => format!("{label} {v}"),
            None => format!("{label} unknown"),
        };
        [
            say(
                "power",
                self.on_ac.map(|ac| {
                    if ac {
                        "AC".into()
                    } else {
                        "BATTERY".to_string()
                    }
                }),
            ),
            say("load", self.load.map(|l| format!("{l:.2}"))),
            say(
                "background",
                self.background_cpu.map(|c| format!("{c:.0}%")),
            ),
            say(
                "busiest",
                self.busiest.as_ref().map(|(n, c)| format!("{n} {c:.0}%")),
            ),
        ]
        .join(", ")
    }
}

#[cfg(target_os = "macos")]
fn on_ac_power() -> Option<bool> {
    let out = std::process::Command::new("pmset")
        .args(["-g", "batt"])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    Some(text.contains("'AC Power'"))
}

#[cfg(not(target_os = "macos"))]
fn on_ac_power() -> Option<bool> {
    None
}

#[cfg(target_os = "macos")]
fn load_average() -> Option<f64> {
    let out = std::process::Command::new("sysctl")
        .args(["-n", "vm.loadavg"])
        .output()
        .ok()?;
    parse_loadavg(&String::from_utf8(out.stdout).ok()?)
}

#[cfg(not(target_os = "macos"))]
fn load_average() -> Option<f64> {
    parse_loadavg(&std::fs::read_to_string("/proc/loadavg").ok()?)
}

/// The one-minute figure out of a `{ 1.86 3.49 3.33 }` or `1.86 3.49 …`.
pub(super) fn parse_loadavg(text: &str) -> Option<f64> {
    text.split_whitespace().find_map(|token| {
        token
            .trim_matches(|c: char| !c.is_ascii_digit() && c != '.')
            .parse()
            .ok()
    })
}

/// Total foreign CPU and the busiest foreign process.
///
/// Excludes THIS PROCESS ONLY, by pid.
///
/// The first version excluded anything named `larql`, reasoning that a
/// benchmark is not disqualified by being the benchmark. That was wrong
/// in the direction that matters: a STALE `larql` from a previous run, or
/// a `larql-server` holding another model resident in a different
/// worktree, competes for exactly the memory bandwidth being measured —
/// and a gate that excused it by name would call such a machine quiet.
///
/// Two idle `larql-server` processes were in fact running while this was
/// written. They happened to be harmless (0.0% CPU, 0.01 GB resident,
/// models never loaded), which is precisely why the name-based rule would
/// have survived unnoticed until the day one of them was busy.
fn process_cpu() -> Option<(f64, Option<(String, f64)>)> {
    let out = std::process::Command::new("ps")
        .args(["-A", "-o", "pid=,%cpu=,comm="])
        .output()
        .ok()?;
    let text = String::from_utf8(out.stdout).ok()?;
    Some(summarise_ps(&text, std::process::id()))
}

/// Split out so the parse is testable without a process table.
pub(super) fn summarise_ps(text: &str, own_pid: u32) -> (f64, Option<(String, f64)>) {
    let mut total = 0.0;
    let mut busiest: Option<(String, f64)> = None;
    for line in text.lines() {
        // `split_whitespace` and not `splitn`: `ps` pads its columns, and
        // splitting on each whitespace CHARACTER turns the padding into
        // empty fields. The command is rejoined because a process name
        // can contain spaces — "Creative Cloud" is one of them, and it is
        // the process this gate exists to catch.
        let mut fields = line.split_whitespace();
        let (Some(pid), Some(percent)) = (fields.next(), fields.next()) else {
            continue;
        };
        let (Ok(pid), Ok(percent)) = (pid.parse::<u32>(), percent.parse::<f64>()) else {
            continue;
        };
        if pid == own_pid {
            continue;
        }
        let command = fields.collect::<Vec<_>>().join(" ");
        let name = command.trim();
        total += percent;
        if busiest.as_ref().is_none_or(|(_, p)| percent > *p) {
            let short = name.rsplit('/').next().unwrap_or(name).to_string();
            busiest = Some((short, percent));
        }
    }
    (total, busiest)
}

#[cfg(test)]
#[path = "tests/environment.rs"]
mod tests;
