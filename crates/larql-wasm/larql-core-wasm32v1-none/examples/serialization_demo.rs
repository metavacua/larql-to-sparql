//! Serialization demo — JSON vs MessagePack, packed binary, CSV, format detection, bytes API.
//!
//! Run: cargo run --release -p larql-core --example serialization_demo

use larql_core::*;

fn main() {
    println!("=== LARQL Serialization Demo ===\n");

    // Build a sample graph
    let mut graph = Graph::new();
    for i in 0..1000 {
        graph.add_edge(
            Edge::new(
                format!("Entity_{}", i / 10),
                format!("rel_{}", i % 5),
                format!("Target_{i}"),
            )
            .with_confidence(0.5 + (i as f64 % 50.0) / 100.0)
            .with_source(SourceType::Parametric),
        );
    }
    println!(
        "Graph: {} edges, {} nodes\n",
        graph.edge_count(),
        graph.node_count()
    );

    // ── JSON ──
    let json_bytes = to_bytes(&graph, Format::Json).unwrap();
    println!(
        "JSON:     {} bytes ({:.1} KB)",
        json_bytes.len(),
        json_bytes.len() as f64 / 1024.0
    );

    // ── MessagePack ──
    #[cfg(feature = "msgpack")]
    let msgpack_bytes = {
        let bytes = to_bytes(&graph, Format::MessagePack).unwrap();
        println!(
            "MsgPack:  {} bytes ({:.1} KB)",
            bytes.len(),
            bytes.len() as f64 / 1024.0
        );
        println!(
            "  vs JSON: {:.0}% smaller",
            (1.0 - bytes.len() as f64 / json_bytes.len() as f64) * 100.0
        );
        bytes
    };
    #[cfg(not(feature = "msgpack"))]
    println!("MsgPack:  disabled (build with --features msgpack)");

    // ── Packed binary ──
    let packed_bytes = to_bytes(&graph, Format::Packed).unwrap();
    println!(
        "Packed:   {} bytes ({:.1} KB)",
        packed_bytes.len(),
        packed_bytes.len() as f64 / 1024.0
    );
    println!(
        "  vs JSON: {:.0}% smaller\n",
        (1.0 - packed_bytes.len() as f64 / json_bytes.len() as f64) * 100.0
    );

    // ── Roundtrip bytes ──
    let from_json = from_bytes(&json_bytes, Format::Json).unwrap();
    #[cfg(feature = "msgpack")]
    let from_msgpack = from_bytes(&msgpack_bytes, Format::MessagePack).unwrap();
    let from_packed = from_bytes(&packed_bytes, Format::Packed).unwrap();
    println!("Roundtrip JSON:    {} edges", from_json.edge_count());
    #[cfg(feature = "msgpack")]
    println!("Roundtrip MsgPack: {} edges", from_msgpack.edge_count());
    println!("Roundtrip Packed:  {} edges", from_packed.edge_count());

    // ── CSV with quoted fields ──
    let mut csv_graph = Graph::new();
    csv_graph.add_edge(Edge::new(
        "Washington, D.C.",
        "nickname",
        "The \"District\"",
    ));
    csv_graph.add_edge(Edge::new("Line\nBreak", "rel", "Value, with comma"));

    let tmp_csv = std::env::temp_dir().join("demo.larql.csv");
    save_csv(&csv_graph, &tmp_csv).unwrap();
    let csv_roundtrip = load_csv(&tmp_csv).unwrap();
    println!(
        "Roundtrip CSV:     {} edges, quoted fields preserved={}",
        csv_roundtrip.edge_count(),
        csv_roundtrip.exists("Washington, D.C.", "nickname", "The \"District\"")
    );
    std::fs::remove_file(&tmp_csv).ok();

    // ── File format detection ──
    println!("\nFormat detection:");
    for path in &[
        "graph.larql.json",
        "graph.json",
        "graph.larql.pak",
        "graph.pak",
    ] {
        let fmt = Format::from_path(path);
        println!("  {path:25} → {fmt:?}");
    }
    #[cfg(feature = "msgpack")]
    for path in &["graph.larql.bin", "graph.bin", "graph.msgpack"] {
        let fmt = Format::from_path(path);
        println!("  {path:25} → {fmt:?}");
    }

    // ── File save/load ──
    let tmp_json = std::env::temp_dir().join("demo.larql.json");

    save(&graph, &tmp_json).unwrap();
    #[cfg(feature = "msgpack")]
    let tmp_bin = {
        let path = std::env::temp_dir().join("demo.larql.bin");
        save(&graph, &path).unwrap();
        path
    };

    let json_size = std::fs::metadata(&tmp_json).unwrap().len();
    #[cfg(feature = "msgpack")]
    let bin_size = std::fs::metadata(&tmp_bin).unwrap().len();
    println!("\nFile sizes:");
    println!("  JSON:    {json_size} bytes");
    #[cfg(feature = "msgpack")]
    println!("  MsgPack: {bin_size} bytes");

    // Auto-detect on load
    let g1 = load(&tmp_json).unwrap();
    #[cfg(feature = "msgpack")]
    let g2 = load(&tmp_bin).unwrap();
    println!("\nAuto-detect load:");
    println!("  .larql.json → {} edges", g1.edge_count());
    #[cfg(feature = "msgpack")]
    println!("  .larql.bin  → {} edges", g2.edge_count());

    std::fs::remove_file(&tmp_json).ok();
    #[cfg(feature = "msgpack")]
    std::fs::remove_file(&tmp_bin).ok();

    // ── Cross-format ──
    #[cfg(feature = "msgpack")]
    {
        let cross = from_bytes(
            &to_bytes(
                &from_bytes(&json_bytes, Format::Json).unwrap(),
                Format::MessagePack,
            )
            .unwrap(),
            Format::MessagePack,
        )
        .unwrap();
        println!(
            "\nCross-format (JSON → MsgPack → Graph): {} edges",
            cross.edge_count()
        );
    }

    println!("\n=== Done ===");
}
