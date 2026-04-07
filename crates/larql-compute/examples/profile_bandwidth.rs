//! Raw memory bandwidth test — what's the floor on this machine?
//!
//! Tests:
//!   1. Raw sequential memcpy (malloc'd memory)
//!   2. Raw sequential mmap read (file-backed, no madvise)
//!   3. Mmap with MADV_SEQUENTIAL + MADV_WILLNEED
//!   4. BLAS gemv on the same data (what the walk actually does)
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example bench_bandwidth -- \
//!     output/gemma3-4b-v2.vindex/down_features.bin

extern crate larql_compute; // provides BLAS
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1)
        .unwrap_or_else(|| "output/gemma3-4b-v2.vindex/down_features.bin".into());

    let file = std::fs::File::open(&path)?;
    let file_size = file.metadata()?.len() as usize;
    println!("=== Memory Bandwidth Test ===");
    println!("File: {path} ({:.1} GB)\n", file_size as f64 / 1e9);

    let n = 3;

    // 1. Raw sequential read from mmap (no hints)
    {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        // Warmup: touch all pages
        let mut sink = 0u64;
        for chunk in mmap.chunks(4096) {
            sink += chunk[0] as u64;
        }
        std::hint::black_box(sink);

        let t0 = Instant::now();
        for _ in 0..n {
            let mut s = 0u64;
            for chunk in mmap.chunks(4096) {
                s += chunk[0] as u64;
            }
            std::hint::black_box(s);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = file_size as f64 / ms / 1e6;
        println!("Mmap (no hints, warm):       {ms:>6.1}ms  {gbps:>6.1} GB/s");
    }

    // 2. Mmap with MADV_SEQUENTIAL + MADV_WILLNEED
    {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        #[cfg(unix)]
        unsafe {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            libc::madvise(ptr, mmap.len(), libc::MADV_SEQUENTIAL);
            libc::madvise(ptr, mmap.len(), libc::MADV_WILLNEED);
        }
        // Warmup
        let mut sink = 0u64;
        for chunk in mmap.chunks(4096) { sink += chunk[0] as u64; }
        std::hint::black_box(sink);

        let t0 = Instant::now();
        for _ in 0..n {
            let mut s = 0u64;
            for chunk in mmap.chunks(4096) {
                s += chunk[0] as u64;
            }
            std::hint::black_box(s);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = file_size as f64 / ms / 1e6;
        println!("Mmap (SEQUENTIAL+WILLNEED): {ms:>6.1}ms  {gbps:>6.1} GB/s");
    }

    // 3. Full sequential read (sum all bytes, force cache-hot)
    {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        #[cfg(unix)]
        unsafe {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            libc::madvise(ptr, mmap.len(), libc::MADV_SEQUENTIAL);
            libc::madvise(ptr, mmap.len(), libc::MADV_WILLNEED);
        }
        // Full warmup: read every byte
        let mut sink: u64 = 0;
        for &b in mmap.iter() { sink = sink.wrapping_add(b as u64); }
        std::hint::black_box(sink);

        let t0 = Instant::now();
        for _ in 0..n {
            let mut s: u64 = 0;
            let data = &mmap[..];
            // Read in 64-byte cache-line chunks
            let ptr = data.as_ptr();
            let len = data.len();
            for i in (0..len).step_by(64) {
                unsafe { s = s.wrapping_add(*ptr.add(i) as u64); }
            }
            std::hint::black_box(s);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = file_size as f64 / ms / 1e6;
        println!("Mmap (full scan, warm):      {ms:>6.1}ms  {gbps:>6.1} GB/s");
    }

    // 4. BLAS gemv on one layer (105 MB) — what the walk actually does
    {
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        #[cfg(unix)]
        unsafe {
            let ptr = mmap.as_ptr() as *mut libc::c_void;
            libc::madvise(ptr, mmap.len(), libc::MADV_SEQUENTIAL);
            libc::madvise(ptr, mmap.len(), libc::MADV_WILLNEED);
        }

        // One layer: [10240, 2560] f32 = 105 MB
        let intermediate = 10240;
        let hidden = 2560;
        let layer_bytes = intermediate * hidden * 4;
        if file_size >= layer_bytes {
            let data = unsafe {
                let ptr = mmap.as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, intermediate * hidden)
            };
            let matrix = ndarray::ArrayView2::from_shape((intermediate, hidden), data).unwrap();

            // Input vector
            let x = ndarray::Array1::from_vec(vec![1.0f32; hidden]);

            // Warmup
            let _ = matrix.dot(&x);

            let t0 = Instant::now();
            for _ in 0..n {
                let _ = matrix.dot(&x);
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
            let gbps = layer_bytes as f64 / ms / 1e6;
            println!("BLAS gemv (105MB, warm):     {ms:>6.1}ms  {gbps:>6.1} GB/s");
        }
    }

    // 5. malloc + sequential write + read (pure RAM bandwidth)
    {
        let size = file_size.min(512 * 1024 * 1024); // cap at 512MB
        let mut buf = vec![0u8; size];
        // Write to force allocation
        for i in (0..size).step_by(4096) { buf[i] = 1; }

        let t0 = Instant::now();
        for _ in 0..n {
            let mut s: u64 = 0;
            let ptr = buf.as_ptr();
            for i in (0..size).step_by(64) {
                unsafe { s = s.wrapping_add(*ptr.add(i) as u64); }
            }
            std::hint::black_box(s);
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = size as f64 / ms / 1e6;
        println!("Malloc scan ({:.0}MB, warm):   {ms:>6.1}ms  {gbps:>6.1} GB/s", size as f64 / 1e6);
    }

    println!("\n=== Done ===");
    Ok(())
}
