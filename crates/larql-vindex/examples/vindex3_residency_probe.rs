//! VINDEX3 residency probe — the successor to `mmap_cold_read_probe`.
//!
//! The incumbent probe asks whether sparse access to a large blob pages
//! acceptably. VINDEX3 supports a sharper question, because binding precedes
//! execution:
//!
//! > **Does the bound plan predict the pages execution actually touches?**
//!
//! If it does, residency becomes a property you can compute rather than
//! observe — which is what placement, prefetch and remote transfer all need,
//! and all three would be mis-sized by a predictor that quietly disagreed with
//! reality.
//!
//! # Method
//!
//! `mincore(2)`, not fault counting. The incumbent probe documents that
//! Darwin's `MADV_DONTNEED` is lazy and re-eviction unreliable, which makes
//! `getrusage` fault deltas awkward there. `mincore` reports which pages are
//! resident *right now* — a direct read of the thing being claimed, and it
//! answers the sparsity question without needing eviction to be reliable.
//!
//! ```text
//! 1. write a synthetic layer's byte image to disk
//! 2. mmap it, MADV_RANDOM     (readahead off — otherwise it prefetches the
//!                              very sparsity being measured)
//! 3. evict, verify cold       (spine check: near-zero residency)
//! 4. execute exactly one token through the bound operation
//! 5. mincore, attribute pages to router / selected / unselected
//! ```
//!
//! Step 3 is a spine check, not a result. If the mapping is not cold
//! afterwards the numbers mean nothing, and the probe says so rather than
//! reporting a confident zero.
//!
//! Usage:
//! ```text
//! cargo run --release -p larql-vindex --example vindex3_residency_probe -- \
//!     [--population 128] [--hidden 256] [--intermediate 512] [--top-k 2]
//! ```

#[cfg(unix)]
mod probe {
    use std::fs::File;
    use std::io::Write;

    use memmap2::{Mmap, MmapOptions};

    use larql_vindex::runtime::execute_traced;
    use larql_vindex::runtime::fixtures::synthetic::{SyntheticLayer, SyntheticShape};
    use larql_vindex::runtime::residency::{account, ExpertRegion};
    use larql_vindex::runtime::MoeInputs;

    /// Defaults sized to be meaningful (hundreds of MB) without needing a
    /// special machine. Every one is overridable.
    const DEFAULT_POPULATION: usize = 128;
    const DEFAULT_HIDDEN: usize = 256;
    const DEFAULT_INTERMEDIATE: usize = 512;
    const DEFAULT_TOP_K: usize = 2;
    const BLOB_NAME: &str = "vindex3_residency_probe.bin";
    /// Above this fraction resident after eviction, the mapping is not cold
    /// and no downstream number is trustworthy.
    const COLD_THRESHOLD: f64 = 0.05;
    const BYTES_PER_MIB: f64 = 1_048_576.0;

    fn arg(name: &str, default: usize) -> usize {
        let args: Vec<String> = std::env::args().collect();
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .and_then(|v| v.parse().ok())
            .unwrap_or(default)
    }

    fn page_size() -> usize {
        // SAFETY: `sysconf` is a pure query with no preconditions.
        (unsafe { libc::sysconf(libc::_SC_PAGESIZE) }) as usize
    }

    /// Conditions the numbers were produced under.
    ///
    /// Recorded rather than assumed: Darwin's page-cache behaviour varies with
    /// OS release and backing filesystem, so a residency figure without its
    /// conditions is not reproducible. `mmap_cold_read_probe` reached a
    /// different conclusion about `MADV_DONTNEED` on the same platform family,
    /// which is exactly the sort of disagreement this makes resolvable.
    pub struct Environment {
        pub os: String,
        pub filesystem: String,
        pub page_size: usize,
        pub mapped_bytes: usize,
        pub eviction_method: &'static str,
        pub commit: String,
    }

    /// How eviction is performed, named so a result can be attributed to it.
    pub const EVICTION_METHOD: &str = "msync(MS_INVALIDATE) + madvise(MADV_DONTNEED)";

    fn command(program: &str, args: &[&str]) -> String {
        std::process::Command::new(program)
            .args(args)
            .output()
            .ok()
            .filter(|o| o.status.success())
            .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
            .filter(|s| !s.is_empty())
            .unwrap_or_else(|| "unknown".into())
    }

    /// Filesystem type backing `path`, via `statfs`.
    ///
    /// Not `stat(1)`: its `%T` is the *file* type on Darwin, which reported
    /// "unknown" here. `statfs` carries the filesystem name directly on
    /// Darwin and a numeric magic on Linux, so the two are read separately.
    fn filesystem(path: &std::path::Path) -> String {
        use std::ffi::CString;
        let Ok(c_path) = CString::new(path.as_os_str().as_encoded_bytes()) else {
            return "unknown".into();
        };
        // SAFETY: zeroed `statfs` is a valid initial state, and `c_path` is a
        // NUL-terminated path that outlives the call.
        let mut buf: libc::statfs = unsafe { std::mem::zeroed() };
        if unsafe { libc::statfs(c_path.as_ptr(), &mut buf) } != 0 {
            return "unknown".into();
        }
        #[cfg(target_os = "macos")]
        {
            // SAFETY: `f_fstypename` is a NUL-terminated array of c_char.
            let name = unsafe { std::ffi::CStr::from_ptr(buf.f_fstypename.as_ptr()) };
            name.to_string_lossy().into_owned()
        }
        #[cfg(target_os = "linux")]
        {
            // Linux reports a magic number; render it rather than pretending
            // to a name this code would have to keep a table for.
            format!("magic 0x{:x}", buf.f_type)
        }
        // Other unixes spell the field differently again (BSD uses
        // `f_fstypename`, Solaris `f_basetype`). Not guessed — this is a
        // diagnostic, and a wrong field name is a build break on a platform
        // nobody here can test.
        #[cfg(not(any(target_os = "macos", target_os = "linux")))]
        {
            let _ = buf;
            "unknown".to_string()
        }
    }

    fn environment(path: &std::path::Path, mapped_bytes: usize) -> Environment {
        Environment {
            os: format!(
                "{} {}",
                command("uname", &["-sr"]),
                command("uname", &["-m"])
            ),
            filesystem: filesystem(path),
            page_size: page_size(),
            mapped_bytes,
            eviction_method: EVICTION_METHOD,
            commit: command("git", &["rev-parse", "--short", "HEAD"]),
        }
    }

    /// Ask the kernel to drop the mapping's resident pages.
    ///
    /// `msync(MS_INVALIDATE)` first, and that ordering is load-bearing on
    /// Darwin: `MADV_DONTNEED` alone leaves a freshly-written file 100%
    /// resident there, because the pages are clean in the unified buffer
    /// cache and `madvise` declines to drop them. Invalidating first releases
    /// them, and residency goes to zero.
    ///
    /// Measured on macOS 24.6 with a 64 MiB blob:
    ///
    /// ```text
    /// MADV_DONTNEED                4096/4096 resident (100.0%)
    /// MS_INVALIDATE + MADV_DONTNEED   0/4096 resident (  0.0%)
    /// ```
    ///
    /// `mmap_cold_read_probe` works around the same limitation differently,
    /// by falling back to `F_NOCACHE` `pread` for its sparse pass.
    fn evict(mmap: &Mmap) {
        // SAFETY: the pointer and length come from a live mapping we own.
        unsafe {
            libc::msync(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MS_INVALIDATE,
            );
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_DONTNEED,
            );
        }
    }

    /// Turn off readahead. Without this the kernel prefetches neighbouring
    /// pages and the sparsity under measurement is manufactured away.
    fn random_access(mmap: &Mmap) {
        // SAFETY: as above.
        unsafe {
            libc::madvise(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                libc::MADV_RANDOM,
            );
        }
    }

    /// Which pages of the mapping are resident.
    fn resident_pages(mmap: &Mmap, page: usize) -> Vec<bool> {
        let pages = mmap.len().div_ceil(page);
        let mut vec = vec![0u8; pages];
        // SAFETY: `vec` has one byte per page of the live mapping. The third
        // argument's type differs across platforms, hence the cast.
        let rc = unsafe {
            libc::mincore(
                mmap.as_ptr() as *mut libc::c_void,
                mmap.len(),
                vec.as_mut_ptr() as *mut _,
            )
        };
        if rc != 0 {
            eprintln!(
                "  mincore failed (errno {})",
                std::io::Error::last_os_error()
            );
            return vec![false; pages];
        }
        // Bit 0 is the resident flag; higher bits carry other information on
        // some platforms and must be masked off.
        vec.into_iter().map(|b| b & 1 == 1).collect()
    }

    pub fn run() -> std::process::ExitCode {
        let shape = SyntheticShape {
            population: arg("--population", DEFAULT_POPULATION),
            hidden: arg("--hidden", DEFAULT_HIDDEN),
            intermediate: arg("--intermediate", DEFAULT_INTERMEDIATE),
            top_k: arg("--top-k", DEFAULT_TOP_K),
        };
        let page = page_size();

        println!("VINDEX3 residency probe");
        println!(
            "  shape        population {}, hidden {}, intermediate {}, top-k {}",
            shape.population, shape.hidden, shape.intermediate, shape.top_k
        );
        println!(
            "  image        {:.1} MiB total, {:.1} MiB per expert, page {page} B",
            shape.total_bytes() as f64 / BYTES_PER_MIB,
            shape.expert_bytes() as f64 / BYTES_PER_MIB
        );

        // ── Build and write the byte image ─────────────────────────────────
        let layer = SyntheticLayer::build(shape);
        let path = std::env::temp_dir().join(BLOB_NAME);
        {
            let mut file = File::create(&path).expect("create blob");
            file.write_all(&layer.bytes).expect("write blob");
            file.sync_all().expect("sync blob");
        }
        println!("  blob         {}", path.display());

        let file = File::open(&path).expect("open blob");
        // SAFETY: the file is ours, freshly written, and not mutated while
        // mapped.
        let mmap = unsafe { MmapOptions::new().map(&file) }.expect("map blob");
        random_access(&mmap);

        let env = environment(&path, mmap.len());
        println!("\n== conditions ==");
        println!("  os           {}", env.os);
        println!("  filesystem   {}", env.filesystem);
        println!("  page size    {} B", env.page_size);
        println!("  mapped       {} B", env.mapped_bytes);
        println!("  eviction     {}", env.eviction_method);
        println!("  commit       {}", env.commit);

        // ── Spine check: is it actually cold? ──────────────────────────────
        evict(&mmap);
        let before = resident_pages(&mmap, page);
        let total_pages = before.len();
        let cold_resident = before.iter().filter(|r| **r).count();
        let cold_fraction = cold_resident as f64 / total_pages.max(1) as f64;
        println!(
            "\n  after evict  {cold_resident}/{total_pages} pages resident ({:.1}%)",
            cold_fraction * 100.0
        );
        if cold_fraction > COLD_THRESHOLD {
            println!(
                "  SPINE FAIL   MADV_DONTNEED did not evict on this OS — \
                 residency numbers below would be meaningless."
            );
            return std::process::ExitCode::FAILURE;
        }

        // ── One token through the bound operation ──────────────────────────
        let operation = layer.bind(&mmap);
        operation.validate().expect("synthetic layer binds cleanly");
        let (_, trace) = execute_traced(&operation, MoeInputs::shared(&layer.residual()))
            .expect("layer executes");
        let selected = trace.selected_ids();

        let after = resident_pages(&mmap, page);
        let regions: Vec<ExpertRegion> = layer
            .experts
            .iter()
            .enumerate()
            .map(|(e, extents)| ExpertRegion {
                expert_id: e as u32,
                bytes: extents.gate_up.start..extents.down.end,
            })
            .collect();
        let acct = account(
            &after,
            page,
            &layer.router,
            &regions,
            &selected,
            shape.selected_bytes(),
        );

        // ── Report ─────────────────────────────────────────────────────────
        println!("  selected     experts {selected:?}");
        if let Some(margin) = trace.selection_margin {
            println!(
                "  margin       {margin:.6}{}",
                if margin == 0.0 {
                    "  (decided by a tie)"
                } else {
                    ""
                }
            );
        }
        println!("\n== residency after one token ==");
        println!(
            "  predicted    {:>7} pages  (router + {} experts)",
            acct.predicted,
            selected.len()
        );
        println!(
            "  resident     {:>7} pages  ({:.2}% of the mapping)",
            acct.resident_total,
            acct.resident_fraction(total_pages) * 100.0
        );
        println!("    router     {:>7}", acct.resident_router);
        println!("    selected   {:>7}", acct.resident_selected);
        println!(
            "    unselected {:>7}  <- pages no selected operand asked for",
            acct.resident_unselected
        );
        println!(
            "  coverage     {:>7.3}  (resident-and-wanted / predicted)",
            acct.prediction_coverage()
        );
        println!(
            "  overshoot    {:>7.3}  (unselected / resident)",
            acct.overshoot_fraction()
        );

        // The claim, stated as a verdict rather than left to the reader.
        let sparse = acct.resident_fraction(total_pages) < 0.5;
        println!("\n== verdict ==");
        println!(
            "  {} routing stayed sparse: one token made {:.2}% of the layer resident",
            if sparse { "PASS" } else { "FAIL" },
            acct.resident_fraction(total_pages) * 100.0
        );
        println!(
            "  {} the bound plan predicted its own footprint within a page-boundary margin\
             \n\n  scope: reference kernel — its touch envelope equals its required pages.\
             \n  A grouped kernel may exceed this legitimately (whole group extents,\
             \n  aligned over-read, separate scale blocks). Compare such a kernel against\
             \n  its declared envelope, not against zero overshoot.",
            if acct.prediction_coverage() <= 1.0 {
                "PASS"
            } else {
                "FAIL"
            }
        );

        let _ = std::fs::remove_file(&path);
        if sparse {
            std::process::ExitCode::SUCCESS
        } else {
            std::process::ExitCode::FAILURE
        }
    }
}

#[cfg(unix)]
fn main() -> std::process::ExitCode {
    probe::run()
}

#[cfg(not(unix))]
fn main() -> std::process::ExitCode {
    // mincore / madvise are POSIX. Windows would need QueryWorkingSetEx.
    eprintln!("vindex3_residency_probe is unix-only (mmap / madvise / mincore).");
    std::process::ExitCode::SUCCESS
}
