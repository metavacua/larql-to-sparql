// Quick Q8 QKV benchmark — test fused projection speed
use std::time::Instant;

fn main() {
    #[cfg(feature = "metal")]
    {
        use metal::*;
        
        let device = Device::system_default().unwrap();
        let src = larql_compute::metal::shaders::all_shaders();
        let lib = device.new_library_with_source(&src, &CompileOptions::new()).unwrap();
        let pipeline = device.new_compute_pipeline_state_with_function(
            &lib.get_function("q8_qkv_proj", None).unwrap()
        ).unwrap();
        let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
        let queue = device.new_command_queue();
        
        // Gemma 3 4B dimensions
        let hidden = 2560usize;
        let q_dim = 2048usize;
        let kv_dim = 1024usize;
        let blocks = hidden / 32;
        let n = 50;
        
        // Generate Q8 data
        let wq: Vec<u8> = (0..q_dim * hidden).map(|i| (i % 200) as u8).collect();
        let wk: Vec<u8> = (0..kv_dim * hidden).map(|i| (i % 180) as u8).collect();
        let wv: Vec<u8> = (0..kv_dim * hidden).map(|i| (i % 160) as u8).collect();
        let wqs: Vec<f32> = vec![0.01; q_dim * blocks];
        let wks: Vec<f32> = vec![0.01; kv_dim * blocks];
        let wvs: Vec<f32> = vec![0.01; kv_dim * blocks];
        let x8: Vec<i8> = (0..hidden).map(|i| ((i % 100) as i8 - 50)).collect();
        let xs: Vec<f32> = vec![0.02; blocks];
        
        let buf_wq = bufs.get_bytes(&wq);
        let buf_wk = bufs.get_bytes(&wk);
        let buf_wv = bufs.get_bytes(&wv);
        let buf_x = bufs.transient_from_i8(&x8);
        let buf_wqs = bufs.transient_from_f32(&wqs);
        let buf_wks = bufs.transient_from_f32(&wks);
        let buf_wvs = bufs.transient_from_f32(&wvs);
        let buf_xs = bufs.transient_from_f32(&xs);
        let buf_q_out = bufs.output((q_dim * 4) as u64);
        let buf_k_out = bufs.output((kv_dim * 4) as u64);
        let buf_v_out = bufs.output((kv_dim * 4) as u64);
        
        let total_rows = (q_dim + kv_dim + kv_dim) as u32;
        let q_rows = q_dim as u32;
        let k_rows = kv_dim as u32;
        let v_rows = kv_dim as u32;
        let k_val = hidden as u32;
        
        // Warmup
        for _ in 0..3 {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&buf_wq), 0);
            enc.set_buffer(1, Some(&buf_wk), 0);
            enc.set_buffer(2, Some(&buf_wv), 0);
            enc.set_buffer(3, Some(&buf_x), 0);
            enc.set_buffer(4, Some(&buf_wqs), 0);
            enc.set_buffer(5, Some(&buf_wks), 0);
            enc.set_buffer(6, Some(&buf_wvs), 0);
            enc.set_buffer(7, Some(&buf_xs), 0);
            enc.set_buffer(8, Some(&buf_q_out), 0);
            enc.set_buffer(9, Some(&buf_k_out), 0);
            enc.set_buffer(10, Some(&buf_v_out), 0);
            enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((total_rows as u64) + 7) / 8, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        
        // Benchmark
        let t0 = Instant::now();
        for _ in 0..n {
            let cmd = queue.new_command_buffer();
            let enc = cmd.new_compute_command_encoder();
            enc.set_compute_pipeline_state(&pipeline);
            enc.set_buffer(0, Some(&buf_wq), 0);
            enc.set_buffer(1, Some(&buf_wk), 0);
            enc.set_buffer(2, Some(&buf_wv), 0);
            enc.set_buffer(3, Some(&buf_x), 0);
            enc.set_buffer(4, Some(&buf_wqs), 0);
            enc.set_buffer(5, Some(&buf_wks), 0);
            enc.set_buffer(6, Some(&buf_wvs), 0);
            enc.set_buffer(7, Some(&buf_xs), 0);
            enc.set_buffer(8, Some(&buf_q_out), 0);
            enc.set_buffer(9, Some(&buf_k_out), 0);
            enc.set_buffer(10, Some(&buf_v_out), 0);
            enc.set_bytes(11, 4, &q_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(12, 4, &k_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(13, 4, &v_rows as *const u32 as *const std::ffi::c_void);
            enc.set_bytes(14, 4, &k_val as *const u32 as *const std::ffi::c_void);
            enc.dispatch_thread_groups(
                MTLSize::new(((total_rows as u64) + 7) / 8, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            enc.end_encoding();
            cmd.commit();
            cmd.wait_until_completed();
        }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        
        let data_mb = (q_dim + kv_dim * 2) as f64 * hidden as f64 / 1e6;
        let gbps = data_mb / ms / 1000.0;
        
        // Also benchmark 3 separate Q8 matvecs for comparison
        let q8_pipeline = device.new_compute_pipeline_state_with_function(
            &lib.get_function("q8_matvec", None).unwrap()
        ).unwrap();
        
        let t0 = Instant::now();
        for _ in 0..n {
            for (w_buf, ws_buf, out_buf, rows) in &[
                (&buf_wq, &buf_wqs, &buf_q_out, q_dim),
                (&buf_wk, &buf_wks, &buf_k_out, kv_dim),
                (&buf_wv, &buf_wvs, &buf_v_out, kv_dim),
            ] {
                let cmd = queue.new_command_buffer();
                let enc = cmd.new_compute_command_encoder();
                enc.set_compute_pipeline_state(&q8_pipeline);
                enc.set_buffer(0, Some(w_buf), 0);
                enc.set_buffer(1, Some(&buf_x), 0);
                enc.set_buffer(2, Some(ws_buf), 0);
                enc.set_buffer(3, Some(&buf_xs), 0);
                enc.set_buffer(4, Some(out_buf), 0);
                let r = *rows as u32;
                enc.set_bytes(5, 4, &r as *const u32 as *const std::ffi::c_void);
                enc.set_bytes(6, 4, &k_val as *const u32 as *const std::ffi::c_void);
                enc.dispatch_thread_groups(
                    MTLSize::new(((*rows as u64) + 7) / 8, 1, 1),
                    MTLSize::new(256, 1, 1),
                );
                enc.end_encoding();
                cmd.commit();
                cmd.wait_until_completed();
            }
        }
        let sep_ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        
        println!("=== Q8 QKV Projection Benchmark ===");
        println!("  Gemma 3 4B: Q[{q_dim},{hidden}] + K[{kv_dim},{hidden}] + V[{kv_dim},{hidden}]");
        println!("  Data: {data_mb:.1} MB Q8\n");
        println!("  Fused Q+K+V (1 dispatch):    {ms:.3}ms  ({gbps:.1} GB/s)");
        println!("  Separate Q+K+V (3 dispatch):  {sep_ms:.3}ms");
        println!("  Speedup:                      {:.1}x", sep_ms / ms);
        println!("  Per 21 layers:                {:.1}ms fused, {:.1}ms separate", ms * 21.0, sep_ms * 21.0);
    }
    #[cfg(not(feature = "metal"))]
    println!("Metal not enabled");
}
