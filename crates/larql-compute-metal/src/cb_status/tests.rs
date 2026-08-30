use super::*;
use std::path::Path;

/// Every `.rs` file under `src/`, recursively.
fn rust_sources(dir: &Path, out: &mut Vec<std::path::PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("src dir readable") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_sources(&path, out);
        } else if path.extension().is_some_and(|e| e == "rs") {
            out.push(path);
        }
    }
}

/// The containment rule for #229: no production code waits on a command
/// buffer without reading its status afterwards. `wait_until_completed`
/// returns for a failed or ignored buffer exactly as for a finished one,
/// so a naked wait is a step that can hand GPU garbage to the sampler.
/// This module is the only place the raw call may appear.
#[test]
fn no_naked_wait_until_completed_outside_cb_status() {
    let src = Path::new(env!("CARGO_MANIFEST_DIR")).join("src");
    let mut files = Vec::new();
    rust_sources(&src, &mut files);
    assert!(files.len() > 50, "walked too few files: {}", files.len());

    let mut offenders = Vec::new();
    for path in files {
        // The definition site, and this test's own needle below.
        if path.ends_with("cb_status.rs") || path.ends_with("cb_status/tests.rs") {
            continue;
        }
        let text = std::fs::read_to_string(&path).expect("source readable");
        for (i, line) in text.lines().enumerate() {
            let code = line.split("//").next().unwrap_or("");
            if code.contains(concat!(".wait_until_", "completed()")) {
                offenders.push(format!("{}:{}", path.display(), i + 1));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "naked wait_until_completed — use cb_status::wait_checked:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn non_completed_count_starts_at_zero_and_is_monotonic() {
    let a = non_completed_count();
    let b = non_completed_count();
    assert!(b >= a);
}

#[cfg(target_os = "macos")]
#[test]
fn a_completed_empty_command_buffer_passes() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    let cmd = queue.new_command_buffer();
    cmd.commit();
    let before = non_completed_count();
    assert!(wait_checked(cmd, "cb_status test").is_ok());
    assert_eq!(non_completed_count(), before);
}

#[cfg(target_os = "macos")]
#[test]
fn a_buffer_that_never_ran_is_reported_and_counted() {
    let Some(device) = metal::Device::system_default() else {
        return;
    };
    let queue = device.new_command_queue();
    // Never committed: status stays NotEnqueued, and there is no NSError,
    // which exercises the "<no NSError>" arm.
    let cmd = queue.new_command_buffer();
    let before = non_completed_count();
    let err = check_completed(cmd, "cb_status test: not enqueued").expect_err("not completed");
    assert!(err.contains("NotEnqueued"), "{err}");
    assert!(err.contains("<no NSError>"), "{err}");
    assert!(err.contains("cb_status test: not enqueued"), "{err}");
    assert_eq!(non_completed_count(), before + 1);
}

#[test]
fn ns_string_of_nil_is_none() {
    // SAFETY: nil is a valid argument; the function checks it first.
    assert!(unsafe { ns_string(std::ptr::null_mut()) }.is_none());
}

#[cfg(target_os = "macos")]
#[test]
fn ns_string_reads_an_nsstring() {
    use objc::{class, msg_send, sel, sel_impl};
    let text = c"command buffer text";
    // SAFETY: standard Foundation call; the returned NSString is
    // autoreleased and only read.
    let got = unsafe {
        let ns: *mut Object = msg_send![class!(NSString), stringWithUTF8String: text.as_ptr()];
        ns_string(ns)
    };
    assert_eq!(got.as_deref(), Some("command buffer text"));
}
