use larql_core::*;

#[test]
fn test_checkpoint_write_and_replay() {
    let path = std::env::temp_dir().join("test_checkpoint.log");
    // Clean up from any prior run
    std::fs::remove_file(&path).ok();

    {
        let mut cp = CheckpointLog::open(&path).unwrap();
        cp.append(&Edge::new("France", "capital-of", "Paris").with_confidence(0.89))
            .unwrap();
        cp.append(&Edge::new("Germany", "capital-of", "Berlin").with_confidence(0.81))
            .unwrap();
        assert_eq!(cp.edge_count(), 2);
    }

    // Replay
    let cp = CheckpointLog::open(&path).unwrap();
    let graph = cp.replay().unwrap();
    assert_eq!(graph.edge_count(), 2);
    assert!(graph.exists("France", "capital-of", "Paris"));
    assert!(graph.exists("Germany", "capital-of", "Berlin"));

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_checkpoint_append_across_sessions() {
    let path = std::env::temp_dir().join("test_checkpoint_append.log");
    std::fs::remove_file(&path).ok();

    // Session 1
    {
        let mut cp = CheckpointLog::open(&path).unwrap();
        cp.append(&Edge::new("France", "capital-of", "Paris"))
            .unwrap();
    }

    // Session 2
    {
        let mut cp = CheckpointLog::open(&path).unwrap();
        assert_eq!(cp.edge_count(), 1); // sees existing line
        cp.append(&Edge::new("Germany", "capital-of", "Berlin"))
            .unwrap();
        assert_eq!(cp.edge_count(), 2);
    }

    // Verify
    let cp = CheckpointLog::open(&path).unwrap();
    let graph = cp.replay().unwrap();
    assert_eq!(graph.edge_count(), 2);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_checkpoint_empty_file() {
    let path = std::env::temp_dir().join("test_checkpoint_empty.log");
    std::fs::remove_file(&path).ok();

    let cp = CheckpointLog::open(&path).unwrap();
    let graph = cp.replay().unwrap();
    assert_eq!(graph.edge_count(), 0);

    std::fs::remove_file(&path).ok();
}

#[test]
fn test_checkpoint_preserves_metadata() {
    let path = std::env::temp_dir().join("test_checkpoint_meta.log");
    std::fs::remove_file(&path).ok();

    {
        let mut cp = CheckpointLog::open(&path).unwrap();
        cp.append(
            &Edge::new("France", "capital-of", "Paris")
                .with_confidence(0.89)
                .with_source(SourceType::Parametric)
                .with_metadata("model", serde_json::json!("gemma-3")),
        )
        .unwrap();
    }

    let cp = CheckpointLog::open(&path).unwrap();
    let graph = cp.replay().unwrap();
    let edge = &graph.edges()[0];
    assert!((edge.confidence - 0.89).abs() < 0.001);
    assert_eq!(edge.source, SourceType::Parametric);
    assert_eq!(edge.metadata.as_ref().unwrap()["model"], "gemma-3");

    std::fs::remove_file(&path).ok();
}
