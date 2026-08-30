//! The indicatif-backed `PublishCallbacks` implementation.
//!
//! One `MultiProgress` per upload-step (i.e. per sibling repo). Each file
//! gets its own bar via `on_file_start`; `on_file_progress` ticks it as
//! bytes flow through the counting-reader upload body (see
//! `larql_vindex::upload_file_to_hf`). Skipped files get a finished bar
//! so the line stays visible in the scrollback.

use indicatif::{MultiProgress, ProgressBar, ProgressStyle};

pub(super) struct CliPublishCallbacks {
    mp: MultiProgress,
    current: Option<ProgressBar>,
}

impl CliPublishCallbacks {
    pub(super) fn new() -> Self {
        Self {
            mp: MultiProgress::new(),
            current: None,
        }
    }
}

fn make_upload_style() -> ProgressStyle {
    ProgressStyle::with_template(
        "    {msg:28} [{elapsed_precise}] [{wide_bar:.green/blue}] \
         {bytes:>10}/{total_bytes:<10} {bytes_per_sec:>10} ({eta})",
    )
    .unwrap()
    .progress_chars("#>-")
}

fn truncate_msg(s: &str, max: usize) -> String {
    if s.len() > max {
        format!("…{}", &s[s.len() - (max - 1)..])
    } else {
        s.to_string()
    }
}

impl larql_vindex::PublishCallbacks for CliPublishCallbacks {
    fn on_start(&mut self, repo: &str) {
        eprintln!("  Creating repo: {}", repo);
    }

    fn on_file_start(&mut self, filename: &str, size: u64) {
        let mb = size as f64 / 1024.0 / 1024.0;
        eprintln!("  ↑ {filename} ({mb:.0} MB)");
        let bar = self.mp.add(ProgressBar::new(size));
        bar.set_style(make_upload_style());
        bar.set_message(truncate_msg(filename, 28));
        self.current = Some(bar);
    }

    fn on_file_progress(&mut self, _filename: &str, bytes_sent: u64, _total_bytes: u64) {
        if let Some(ref bar) = self.current {
            bar.set_position(bytes_sent);
        }
    }

    fn on_file_done(&mut self, filename: &str) {
        if let Some(bar) = self.current.take() {
            bar.finish();
        }
        eprintln!("    ✓ {filename}");
    }

    fn on_file_skipped(&mut self, filename: &str, _size: u64, sha256: &str) {
        // Print a plain line above the active bars rather than adding a
        // finished-bar stub. `MultiProgress::println` cooperates with
        // indicatif's cursor handling so the output stays one-line-per-
        // file even on wide terminals; the earlier bar-based approach
        // let indicatif pack multiple "skipped" entries on the same row
        // when it thought it had horizontal space.
        let short_sha = sha256.get(..12).unwrap_or(sha256);
        let _ = self.mp.println(format!(
            "    {:<28} [skipped — unchanged, sha256 {}…]",
            truncate_msg(filename, 28),
            short_sha
        ));
    }

    fn on_retry(
        &mut self,
        filename: &str,
        attempt: u32,
        max_attempts: u32,
        reason: &str,
        wait: std::time::Duration,
    ) {
        // Say it out loud: a silent multi-second sleep inside a stalled
        // progress bar is indistinguishable from a hung upload.
        let _ = self.mp.println(format!(
            "    ⟳ {} — {reason}, retrying in {:.0}s (attempt {attempt}/{max_attempts})",
            truncate_msg(filename, 40),
            wait.as_secs_f32(),
        ));
    }

    fn on_file_deleted(&mut self, filename: &str) {
        let _ = self
            .mp
            .println(format!("    ✗ {filename} [pruned — not in source vindex]"));
    }

    fn on_complete(&mut self, url: &str) {
        eprintln!("  URL: {}", url);
    }
}

pub(super) fn human_size(bytes: u64) -> String {
    const K: u64 = 1024;
    const M: u64 = K * 1024;
    const G: u64 = M * 1024;
    if bytes >= G {
        format!("{:.2} GB", bytes as f64 / G as f64)
    } else if bytes >= M {
        format!("{:.1} MB", bytes as f64 / M as f64)
    } else if bytes >= K {
        format!("{:.1} KB", bytes as f64 / K as f64)
    } else {
        format!("{bytes} B")
    }
}
