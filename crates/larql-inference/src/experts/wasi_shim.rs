//! Minimal WASI snapshot-preview1 shim for wasmi.
//!
//! Registers host functions under `wasi_snapshot_preview1` so that experts
//! compiled with the Rust `wasm32-wasip1` target can be loaded and called.
//! Provides real fd_write (stdout/stderr pass-through) and correct environ/args
//! initialization stubs; everything else returns ERRNO_NOSYS or ERRNO_BADF.

use wasmi::{Caller, Linker};

use super::loader::ExpertStore;

const WASI_MODULE: &str = "wasi_snapshot_preview1";
const WASM_MEMORY: &str = "memory";

// WASI errno constants.
const ERRNO_SUCCESS: i32 = 0;
const ERRNO_BADF: i32 = 8;
const ERRNO_FAULT: i32 = 21;
const ERRNO_NOSYS: i32 = 52;

/// Register all WASI preview1 host functions into `linker`.
pub fn add_to_linker(linker: &mut Linker<ExpertStore>) -> anyhow::Result<()> {
    // ── fd_write ──────────────────────────────────────────────────────────────
    // Real implementation: writes to stdout (fd=1) or stderr (fd=2).
    linker.func_wrap(
        WASI_MODULE,
        "fd_write",
        |mut caller: Caller<ExpertStore>,
         fd: i32,
         iovs_ptr: i32,
         iovs_len: i32,
         nwritten_ptr: i32|
         -> i32 {
            let mem = match caller.get_export(WASM_MEMORY).and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return ERRNO_BADF,
            };

            // Collect iovec (buf_ptr, buf_len) pairs before taking a mut borrow.
            let to_write: Vec<Vec<u8>> = {
                let data = mem.data(&caller);
                let mut v = Vec::with_capacity(iovs_len as usize);
                for i in 0..iovs_len as usize {
                    let off = iovs_ptr as usize + i * 8;
                    if off + 8 > data.len() {
                        return ERRNO_FAULT;
                    }
                    let buf_ptr =
                        u32::from_le_bytes(data[off..off + 4].try_into().unwrap()) as usize;
                    let buf_len =
                        u32::from_le_bytes(data[off + 4..off + 8].try_into().unwrap()) as usize;
                    if buf_ptr + buf_len > data.len() {
                        return ERRNO_FAULT;
                    }
                    v.push(data[buf_ptr..buf_ptr + buf_len].to_vec());
                }
                v
            };

            let total: u32 = to_write.iter().map(|b| b.len() as u32).sum();

            {
                use std::io::Write;
                match fd {
                    1 => {
                        for bytes in &to_write {
                            let _ = std::io::stdout().write_all(bytes);
                        }
                    }
                    2 => {
                        for bytes in &to_write {
                            let _ = std::io::stderr().write_all(bytes);
                        }
                    }
                    _ => return ERRNO_BADF,
                }
            }

            let data_mut = mem.data_mut(&mut caller);
            let nw = nwritten_ptr as usize;
            if nw + 4 <= data_mut.len() {
                data_mut[nw..nw + 4].copy_from_slice(&total.to_le_bytes());
            }
            ERRNO_SUCCESS
        },
    )?;

    // ── proc_exit ─────────────────────────────────────────────────────────────
    // Trap with an unreachable code so the host sees an error.
    linker.func_wrap(
        WASI_MODULE,
        "proc_exit",
        |_caller: Caller<ExpertStore>, _code: i32| -> Result<(), wasmi::Error> {
            Err(wasmi::TrapCode::UnreachableCodeReached.into())
        },
    )?;

    // ── environ_sizes_get / environ_get ───────────────────────────────────────
    // No environment variables in the sandbox.
    linker.func_wrap(
        WASI_MODULE,
        "environ_sizes_get",
        |mut caller: Caller<ExpertStore>, count_ptr: i32, buf_size_ptr: i32| -> i32 {
            write_u32_pair(&mut caller, count_ptr, buf_size_ptr, 0, 0)
        },
    )?;

    linker.func_wrap(
        WASI_MODULE,
        "environ_get",
        |_caller: Caller<ExpertStore>, _environ: i32, _environ_buf: i32| -> i32 { ERRNO_SUCCESS },
    )?;

    // ── args_sizes_get / args_get ─────────────────────────────────────────────
    // No command-line arguments.
    linker.func_wrap(
        WASI_MODULE,
        "args_sizes_get",
        |mut caller: Caller<ExpertStore>, argc_ptr: i32, argv_buf_size_ptr: i32| -> i32 {
            write_u32_pair(&mut caller, argc_ptr, argv_buf_size_ptr, 0, 0)
        },
    )?;

    linker.func_wrap(
        WASI_MODULE,
        "args_get",
        |_caller: Caller<ExpertStore>, _argv: i32, _argv_buf: i32| -> i32 { ERRNO_SUCCESS },
    )?;

    // ── clock_time_get ────────────────────────────────────────────────────────
    linker.func_wrap(
        WASI_MODULE,
        "clock_time_get",
        |mut caller: Caller<ExpertStore>, _clock_id: i32, _precision: i64, time_ptr: i32| -> i32 {
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_nanos() as u64;
            let mem = match caller.get_export(WASM_MEMORY).and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return ERRNO_BADF,
            };
            let data = mem.data_mut(&mut caller);
            let tp = time_ptr as usize;
            if tp + 8 <= data.len() {
                data[tp..tp + 8].copy_from_slice(&now.to_le_bytes());
            }
            ERRNO_SUCCESS
        },
    )?;

    // ── clock_res_get ─────────────────────────────────────────────────────────
    linker.func_wrap(
        WASI_MODULE,
        "clock_res_get",
        |mut caller: Caller<ExpertStore>, _clock_id: i32, resolution_ptr: i32| -> i32 {
            let mem = match caller.get_export(WASM_MEMORY).and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return ERRNO_BADF,
            };
            let data = mem.data_mut(&mut caller);
            let rp = resolution_ptr as usize;
            if rp + 8 <= data.len() {
                data[rp..rp + 8].copy_from_slice(&1_000_000u64.to_le_bytes()); // 1 ms
            }
            ERRNO_SUCCESS
        },
    )?;

    // ── random_get ────────────────────────────────────────────────────────────
    linker.func_wrap(
        WASI_MODULE,
        "random_get",
        |mut caller: Caller<ExpertStore>, buf_ptr: i32, buf_len: i32| -> i32 {
            let mem = match caller.get_export(WASM_MEMORY).and_then(|e| e.into_memory()) {
                Some(m) => m,
                None => return ERRNO_BADF,
            };
            let data = mem.data_mut(&mut caller);
            let start = buf_ptr as usize;
            let len = buf_len as usize;
            if start + len > data.len() {
                return ERRNO_FAULT;
            }
            // Non-cryptographic fill — sufficient for HashMap seed randomisation.
            let seed = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .subsec_nanos();
            for (i, b) in data[start..start + len].iter_mut().enumerate() {
                *b = (seed ^ (seed >> 8) ^ i as u32).wrapping_mul(2654435761) as u8;
            }
            ERRNO_SUCCESS
        },
    )?;

    // ── sched_yield ───────────────────────────────────────────────────────────
    linker.func_wrap(
        WASI_MODULE,
        "sched_yield",
        |_caller: Caller<ExpertStore>| -> i32 { ERRNO_SUCCESS },
    )?;

    // ── fd_prestat_get ────────────────────────────────────────────────────────
    // Return BADF to signal no preopened directories.
    linker.func_wrap(
        WASI_MODULE,
        "fd_prestat_get",
        |_caller: Caller<ExpertStore>, _fd: i32, _buf: i32| -> i32 { ERRNO_BADF },
    )?;

    linker.func_wrap(
        WASI_MODULE,
        "fd_prestat_dir_name",
        |_caller: Caller<ExpertStore>, _fd: i32, _path: i32, _path_len: i32| -> i32 { ERRNO_BADF },
    )?;

    // ── fd_fdstat_get ─────────────────────────────────────────────────────────
    linker.func_wrap(
        WASI_MODULE,
        "fd_fdstat_get",
        |_caller: Caller<ExpertStore>, _fd: i32, _buf: i32| -> i32 { ERRNO_BADF },
    )?;

    linker.func_wrap(
        WASI_MODULE,
        "fd_fdstat_set_flags",
        |_caller: Caller<ExpertStore>, _fd: i32, _flags: i32| -> i32 { ERRNO_BADF },
    )?;

    // ── remaining fd_* stubs ──────────────────────────────────────────────────
    macro_rules! stub_i32 {
        ($name:literal, $($param:ty),*) => {
            #[allow(unused_variables)]
            linker.func_wrap(WASI_MODULE, $name,
                |_: Caller<ExpertStore> $(, _: $param)*| -> i32 { ERRNO_NOSYS })?;
        };
    }

    stub_i32!("fd_close", i32);
    stub_i32!("fd_read", i32, i32, i32, i32);
    stub_i32!("fd_seek", i32, i64, i32, i32);
    stub_i32!("fd_tell", i32, i32);
    stub_i32!("fd_sync", i32);
    stub_i32!("fd_datasync", i32);
    stub_i32!("fd_advise", i32, i64, i64, i32);
    stub_i32!("fd_allocate", i32, i64, i64);
    stub_i32!("fd_filestat_get", i32, i32);
    stub_i32!("fd_filestat_set_size", i32, i64);
    stub_i32!("fd_filestat_set_times", i32, i64, i64, i32);
    stub_i32!("fd_pread", i32, i32, i32, i64, i32);
    stub_i32!("fd_pwrite", i32, i32, i32, i64, i32);
    stub_i32!("fd_readdir", i32, i32, i32, i64, i32);
    stub_i32!("fd_renumber", i32, i32);

    // ── path_* stubs ──────────────────────────────────────────────────────────
    stub_i32!("path_create_directory", i32, i32, i32);
    stub_i32!("path_filestat_get", i32, i32, i32, i32, i32);
    stub_i32!("path_filestat_set_times", i32, i32, i32, i32, i64, i64, i32);
    stub_i32!("path_link", i32, i32, i32, i32, i32, i32, i32);
    stub_i32!("path_open", i32, i32, i32, i32, i32, i64, i64, i32, i32);
    stub_i32!("path_readlink", i32, i32, i32, i32, i32, i32);
    stub_i32!("path_remove_directory", i32, i32, i32);
    stub_i32!("path_rename", i32, i32, i32, i32, i32, i32);
    stub_i32!("path_symlink", i32, i32, i32, i32, i32);
    stub_i32!("path_unlink_file", i32, i32, i32);

    // ── poll / sock stubs ─────────────────────────────────────────────────────
    stub_i32!("poll_oneoff", i32, i32, i32, i32);
    stub_i32!("proc_raise", i32);
    stub_i32!("sock_accept", i32, i32, i32);
    stub_i32!("sock_recv", i32, i32, i32, i32, i32, i32);
    stub_i32!("sock_send", i32, i32, i32, i32, i32);
    stub_i32!("sock_shutdown", i32, i32);

    Ok(())
}

/// Write two u32 values to the guest memory at `ptr_a` and `ptr_b`.
fn write_u32_pair(
    caller: &mut Caller<ExpertStore>,
    ptr_a: i32,
    ptr_b: i32,
    val_a: u32,
    val_b: u32,
) -> i32 {
    let mem = match caller.get_export(WASM_MEMORY).and_then(|e| e.into_memory()) {
        Some(m) => m,
        None => return ERRNO_BADF,
    };
    let data = mem.data_mut(caller);
    let a = ptr_a as usize;
    let b = ptr_b as usize;
    if a + 4 <= data.len() {
        data[a..a + 4].copy_from_slice(&val_a.to_le_bytes());
    }
    if b + 4 <= data.len() {
        data[b..b + 4].copy_from_slice(&val_b.to_le_bytes());
    }
    ERRNO_SUCCESS
}
