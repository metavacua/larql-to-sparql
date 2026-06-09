use larql_models::loading::gguf::GgufFile;
use std::collections::HashSet;
use std::path::Path;

fn main() {
    let path =
        Path::new("/tank/ai/deepseek-ai/DeepSeek-V4-Flash-GGUF/DeepSeek-V4-Flash-Q4_K_M.gguf");
    let gguf = GgufFile::open(path).expect("open GGUF");

    // Layer 2 (indexer + tid2eid) vs layer 4 (indexer only) vs layer 3
    // (neither) vs layer 0 (tid2eid only).
    for layer in [0, 1, 2, 3, 4, 5] {
        let prefix = format!("blk.{}.", layer);
        let mut suffixes: Vec<String> = gguf
            .tensor_infos
            .iter()
            .filter_map(|i| {
                i.name()
                    .strip_prefix(&prefix)
                    .map(|s| format!("{} [{:?} t{}]", s, i.dims(), i.tensor_type()))
            })
            .collect();
        suffixes.sort();
        println!("== LAYER {} ({} tensors) ==", layer, suffixes.len());
        for s in &suffixes {
            println!("  {}", s);
        }
        println!();
    }

    // Layer 0 vs layer 4 set diff — what's new in 4 that's not in 0?
    let l0: HashSet<String> = gguf
        .tensor_infos
        .iter()
        .filter_map(|i| i.name().strip_prefix("blk.0.").map(|s| s.to_string()))
        .collect();
    let l4: HashSet<String> = gguf
        .tensor_infos
        .iter()
        .filter_map(|i| i.name().strip_prefix("blk.4.").map(|s| s.to_string()))
        .collect();
    println!("== ONLY IN LAYER 4 (indexer-only layer) ==");
    let mut diff_4: Vec<&String> = l4.difference(&l0).collect();
    diff_4.sort();
    for s in &diff_4 {
        println!("  {}", s);
    }
    println!("\n== ONLY IN LAYER 0 (hash-routing layer) ==");
    let mut diff_0: Vec<&String> = l0.difference(&l4).collect();
    diff_0.sort();
    for s in &diff_0 {
        println!("  {}", s);
    }
}
