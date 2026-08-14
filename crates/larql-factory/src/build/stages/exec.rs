//! The one place a stage turns a non-zero exit code into a stage error.
//!
//! [`super::super::runner`] owns *transport* — spawning a process and
//! capturing its output — and deliberately reports a non-zero exit as a
//! normal [`CommandOutput`] rather than an `Err`, since a caller may
//! want to inspect stdout either way. Every stage here does want the
//! same thing though: fail, and say which command failed and what it
//! wrote to stderr. That policy lived in six near-identical copies
//! before this module, which is how five of them end up with a useful
//! error message and the sixth silently swallows stderr.

use crate::build::runner::{CommandOutput, CommandRunner, Invocation};

/// Separator between the failure description and the captured stderr.
const STDERR_SEPARATOR: &str = ": ";

/// Run `invocation`, mapping a non-zero exit code to
/// `"{what} failed: {stderr}"`.
///
/// `what` names the command and, for stages that run once per output,
/// which item it was running for — e.g. `"larql slice --preset client"`
/// or `"larql verify for /scratch/full.vindex"`. A build log is read
/// after the fact, so naming the item is what makes a failed multi-
/// output build diagnosable.
///
/// A spawn failure (missing binary, OS error) propagates unchanged from
/// [`CommandRunner::run`] — that isn't the command failing, it's the
/// command never running.
pub fn run_checked(
    runner: &dyn CommandRunner,
    invocation: &Invocation,
    what: &str,
) -> Result<CommandOutput, String> {
    let out = runner.run(invocation)?;
    if !out.success() {
        return Err(format!("{what} failed{STDERR_SEPARATOR}{}", out.stderr));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::build::runner::MockRunner;

    fn invocation() -> Invocation {
        Invocation::new(&["verify"], vec!["/scratch/full.vindex".into()])
    }

    fn runner_returning(status: i32, stderr: &str) -> MockRunner {
        MockRunner::new().expect(
            invocation(),
            CommandOutput {
                status,
                stderr: stderr.into(),
                ..Default::default()
            },
        )
    }

    #[test]
    fn zero_exit_returns_the_output() {
        let runner = runner_returning(0, "");
        assert!(run_checked(&runner, &invocation(), "larql verify").is_ok());
    }

    #[test]
    fn stdout_survives_a_successful_run() {
        let runner = MockRunner::new().expect(
            invocation(),
            CommandOutput {
                status: 0,
                stdout: "all shards ok".into(),
                ..Default::default()
            },
        );
        let out = run_checked(&runner, &invocation(), "larql verify").unwrap();
        assert_eq!(out.stdout, "all shards ok");
    }

    #[test]
    fn nonzero_exit_names_the_command_and_quotes_stderr() {
        let runner = runner_returning(1, "checksum mismatch");
        let err = run_checked(&runner, &invocation(), "larql verify").unwrap_err();
        assert_eq!(err, "larql verify failed: checksum mismatch");
    }

    #[test]
    fn the_caller_supplied_qualifier_reaches_the_message() {
        let runner = runner_returning(1, "unknown part");
        let err = run_checked(&runner, &invocation(), "larql slice --preset client").unwrap_err();
        assert!(err.contains("client"), "{err}");
        assert!(err.contains("unknown part"), "{err}");
    }

    #[test]
    fn empty_stderr_still_produces_a_named_error() {
        // A command that fails silently must not yield a bare ": ".
        let runner = runner_returning(2, "");
        let err = run_checked(&runner, &invocation(), "larql verify").unwrap_err();
        assert!(err.starts_with("larql verify failed"), "{err}");
    }

    #[test]
    fn a_spawn_error_propagates_unchanged() {
        // Not a command failure — the command never ran, so the message
        // must not be rewritten as one.
        let runner = MockRunner::new().expect_spawn_error(invocation(), "no such binary");
        let err = run_checked(&runner, &invocation(), "larql verify").unwrap_err();
        assert_eq!(err, "no such binary");
    }
}
