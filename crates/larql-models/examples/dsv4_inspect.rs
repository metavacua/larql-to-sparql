use larql_models::loading::gguf::{GgufFile, GgufValue};
use std::path::Path;

fn truncate_str(s: &str, max: usize) -> String {
    let mut end = max.min(s.len());
    while end > 0 && !s.is_char_boundary(end) {
        end -= 1;
    }
    if end >= s.len() {
        s.to_string()
    } else {
        format!("{}…", &s[..end])
    }
}

fn main() {
    let path =
        Path::new("/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf");
    let gguf = GgufFile::open(path).expect("open GGUF");

    println!("== METADATA ({} keys) ==", gguf.metadata.len());
    let mut keys: Vec<&String> = gguf.metadata.keys().collect();
    keys.sort();
    for k in &keys {
        let v = &gguf.metadata[*k];
        let summary = format!("{:?}", v);
        println!("  {}: {}", k, truncate_str(&summary, 200));
    }

    println!("\n== LAYER 0 TENSORS ==");
    let mut layer0: Vec<_> = gguf
        .tensor_infos
        .iter()
        .filter(|i| i.name().starts_with("blk.0."))
        .collect();
    layer0.sort_by_key(|i| i.name().to_string());
    for info in &layer0 {
        println!(
            "  {:?} {:?} type={}",
            info.name(),
            info.dims(),
            info.tensor_type()
        );
    }
    println!("\n  ({} layer-0 tensors)", layer0.len());

    println!("\n== GLOBAL TENSORS (not blk.*) ==");
    let mut globals: Vec<_> = gguf
        .tensor_infos
        .iter()
        .filter(|i| !i.name().starts_with("blk."))
        .collect();
    globals.sort_by_key(|i| i.name().to_string());
    for info in &globals {
        println!(
            "  {:?} {:?} type={}",
            info.name(),
            info.dims(),
            info.tensor_type()
        );
    }

    println!("\n== COMPRESSOR / INDEXER PRESENCE PER LAYER ==");
    let n_layer = gguf
        .metadata
        .get("deepseek4.block_count")
        .and_then(|v| v.as_u32())
        .unwrap_or(0) as usize;
    for l in 0..n_layer {
        let prefix = format!("blk.{}.", l);
        let has_attn_compressor = gguf
            .tensor_infos
            .iter()
            .any(|i| i.name().starts_with(&prefix) && i.name().contains("attn_compressor"));
        let has_indexer = gguf
            .tensor_infos
            .iter()
            .any(|i| i.name().starts_with(&prefix) && i.name().contains("indexer"));
        let has_tid2eid = gguf
            .tensor_infos
            .iter()
            .any(|i| i.name().starts_with(&prefix) && i.name().contains("tid2eid"));
        println!(
            "  layer {:>2}: compressor={} indexer={} tid2eid={}",
            l, has_attn_compressor, has_indexer, has_tid2eid
        );
    }

    // Dump the swiglu_clamp_exp array fully.
    if let Some(GgufValue::Array(arr)) = gguf.metadata.get("deepseek4.swiglu_clamp_exp") {
        println!("\n== swiglu_clamp_exp ({} entries) ==", arr.len());
        for (i, v) in arr.iter().enumerate() {
            if let GgufValue::F32(f) = v {
                println!("  [{}] = {}", i, f);
            }
        }
    }
}
