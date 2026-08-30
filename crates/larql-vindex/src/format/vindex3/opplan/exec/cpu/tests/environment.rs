//! The gate must refuse the machines that actually contaminated this
//! programme, and must not refuse a quiet one.

use super::super::environment::{parse_loadavg, summarise_ps, Disqualifier, Environment, Phase};

/// The three real contamination events, replayed as inputs.
///
/// Written from what was actually observed rather than from invented
/// numbers, so the gate is tested against the failures it exists for
/// instead of against a threshold restated as a fixture.
#[test]
fn it_refuses_every_machine_that_contaminated_a_measurement() {
    let adobe = Environment {
        on_ac: Some(true),
        load: Some(4.24),
        background_cpu: Some(123.0),
        busiest: Some(("Creative Cloud".into(), 96.5)),
    };
    let battery = Environment {
        on_ac: Some(false),
        load: Some(1.9),
        background_cpu: Some(20.0),
        busiest: Some(("WindowServer".into(), 4.0)),
    };
    let xprotect = Environment {
        on_ac: Some(true),
        load: Some(14.38),
        background_cpu: Some(48.0),
        busiest: Some(("XprotectService".into(), 15.9)),
    };
    for (name, env) in [
        ("adobe", adobe),
        ("battery", battery),
        ("xprotect", xprotect),
    ] {
        assert!(
            !env.disqualifiers(Phase::BeforeWork).is_empty(),
            "`{name}` would have been measured on: {}",
            env.describe()
        );
    }
    // And each for the right reason, so a gate that refused everything
    // for one blunt reason would not pass.
    assert!(matches!(
        Environment {
            on_ac: Some(false),
            ..Default::default()
        }
        .disqualifiers(Phase::BeforeWork)
        .as_slice(),
        [Disqualifier::OnBattery]
    ));
}

/// A quiet machine is eligible.
///
/// The other half of the gate: one that refused everything would satisfy
/// the test above and be useless.
#[test]
fn it_admits_a_quiet_machine() {
    let quiet = Environment {
        on_ac: Some(true),
        load: Some(1.86),
        background_cpu: Some(35.0),
        busiest: Some(("WindowServer".into(), 4.4)),
    };
    assert!(
        quiet.disqualifiers(Phase::BeforeWork).is_empty(),
        "the clean run's own environment was refused: {:?}",
        quiet.disqualifiers(Phase::BeforeWork)
    );
}

/// A platform that reports nothing is not refused for it.
///
/// An unknown is not a violation. `describe` has to say which checks
/// actually ran, or a Linux run would look like a verified-quiet one.
#[test]
fn unknown_is_not_a_violation_but_is_reported() {
    let blind = Environment::default();
    assert!(blind.disqualifiers(Phase::BeforeWork).is_empty());
    let text = blind.describe();
    assert_eq!(text.matches("unknown").count(), 4, "{text}");
}

/// One busy process is refused even when the total looks calm.
///
/// Adobe sat at 96% of a core while most of the machine was idle; a gate
/// on the aggregate alone would have admitted it.
#[test]
fn a_single_busy_process_is_enough() {
    let env = Environment {
        on_ac: Some(true),
        load: Some(1.2),
        background_cpu: Some(20.0),
        busiest: Some(("Creative Cloud".into(), 96.5)),
    };
    assert!(matches!(
        env.disqualifiers(Phase::BeforeWork).as_slice(),
        [Disqualifier::Process { .. }]
    ));
}

#[test]
fn the_load_average_parse_takes_the_one_minute_figure() {
    assert_eq!(parse_loadavg("{ 1.86 3.49 3.33 }"), Some(1.86));
    assert_eq!(parse_loadavg("14.38 8.00 5.94 1/900 12345"), Some(14.38));
    assert_eq!(parse_loadavg("not a load average"), None);
}

/// Only THIS process is excused, and a stale sibling is not.
///
/// The first version excluded anything named `larql`. A `larql-server`
/// holding another model resident, or a leftover run that has not exited,
/// competes for exactly the bandwidth being measured — excusing it by
/// name would call such a machine quiet. Two idle `larql-server`
/// processes were running when this was written, harmless only by luck.
#[test]
fn only_this_process_is_excused() {
    let table = concat!(
        "  100  380.0 /path/to/target/release/larql\n",
        "  200   96.5 /Applications/Adobe/Creative Cloud\n",
        "  300   55.0 /other/worktree/target/release/larql-server\n",
        "  400    4.4 /System/.../WindowServer\n",
    );
    let (total, busiest) = summarise_ps(table, 100);
    assert!(
        (total - 155.9).abs() < 0.01,
        "a sibling larql must count against the machine: {total}"
    );
    assert_eq!(busiest, Some(("Creative Cloud".into(), 96.5)));

    // And with a different pid excused, our own 380% counts.
    let (total, busiest) = summarise_ps(table, 999);
    assert!((total - 535.9).abs() < 0.01, "{total}");
    assert_eq!(busiest, Some(("larql".into(), 380.0)));
}
