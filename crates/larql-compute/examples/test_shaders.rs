//! Test that all Metal shaders compile.

fn main() {
    #[cfg(feature = "metal")]
    {
        use metal::*;
        let device = Device::system_default().expect("No Metal device");
        let src = larql_compute::metal::shaders::all_shaders();
        println!("Shader source: {} chars", src.len());

        let opts = CompileOptions::new();
        match device.new_library_with_source(&src, &opts) {
            Ok(lib) => {
                println!("Compiled OK!");
                for name in &["sgemm", "sgemm_transb", "q4_matvec", "q4_vecmat",
                              "q4_f32_matvec", "geglu_silu", "quantize_q8", "causal_attention",
                              "rope_apply", "fused_attention",
                              "kv_attention", "kv_cache_append",
                              "q4_matvec_v2", "q4_matvec_v3", "q4_matvec_v4", "q4_matvec_v5",
                              "rms_norm_q8", "residual_norm", "residual_norm_q8",
                              "rms_norm", "residual_add", "q8_matvec",
                              "q8_proj_rope", "q8_qkv_proj",
                              "rms_norm_q8", "residual_norm", "residual_norm_q8",
                              "q4k_matvec", "q6k_matvec"] {
                    match lib.get_function(name, None) {
                        Ok(_) => println!("  ✓ {name}"),
                        Err(e) => println!("  ✗ {name}: {e}"),
                    }
                }
            }
            Err(e) => {
                println!("COMPILE ERROR: {e}");
                // Print first 500 chars for debugging
                println!("\nFirst 500 chars of source:");
                println!("{}", &src[..500.min(src.len())]);
            }
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("Metal not enabled");
}
