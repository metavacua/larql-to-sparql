//! Shared test fixtures for the LFS sibling test suites.

use std::io::Write as _;

use crate::format::huggingface::publish::PublishCallbacks;

/// Set `LARQL_HF_BASE_URL` for the lifetime of the guard, restoring the
/// previous value on drop. Wraps the env var that
/// `super::super::protocol::hf_base()` reads, letting tests point all
/// HF traffic at a `mockito::Server`.
pub(super) struct EnvBaseGuard {
    prev: Option<String>,
}
impl EnvBaseGuard {
    pub(super) fn new(value: &str) -> Self {
        let prev = std::env::var(crate::format::huggingface::publish::protocol::TEST_BASE_ENV).ok();
        std::env::set_var(
            crate::format::huggingface::publish::protocol::TEST_BASE_ENV,
            value,
        );
        Self { prev }
    }
}
impl Drop for EnvBaseGuard {
    fn drop(&mut self) {
        match self.prev.take() {
            Some(v) => std::env::set_var(
                crate::format::huggingface::publish::protocol::TEST_BASE_ENV,
                v,
            ),
            None => {
                std::env::remove_var(crate::format::huggingface::publish::protocol::TEST_BASE_ENV)
            }
        }
    }
}

#[derive(Default)]
pub(super) struct CapturingCallbacks {
    pub(super) progress_calls: Vec<(String, u64, u64)>,
}
impl PublishCallbacks for CapturingCallbacks {
    fn on_file_progress(&mut self, filename: &str, bytes_sent: u64, total_bytes: u64) {
        self.progress_calls
            .push((filename.to_string(), bytes_sent, total_bytes));
    }
}

pub(super) fn write_temp_bytes(bytes: &[u8]) -> (tempfile::TempDir, std::path::PathBuf) {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("payload.bin");
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(bytes).unwrap();
    f.flush().unwrap();
    (dir, path)
}
