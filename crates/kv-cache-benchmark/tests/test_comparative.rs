use kv_cache_benchmark::*;
use kv_cache_benchmark::benchmark;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::markov_residual::MarkovResidual;
use kv_cache_benchmark::boundary_residual::BoundaryResidual;
use kv_cache_benchmark::graph_walk::GraphWalk;

#[test]
fn test_all_strategies_memory_ordering() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let boundary = BoundaryResidual::gemma_4b();
    let graph = GraphWalk::gemma_4b();

    for &seq_len in &[4096, 32768, 370_000] {
        let mem_std = standard.memory_bytes(&config, seq_len);
        let mem_tq = tq4.memory_bytes(&config, seq_len);
        let mem_mrk = markov.memory_bytes(&config, seq_len);
        let mem_brs = boundary.memory_bytes(&config, seq_len);
        let mem_gw = graph.memory_bytes(&config, seq_len);

        // Standard KV is always the largest.
        assert!(mem_std > mem_tq,  "At {seq_len}: Standard ({mem_std}) > TurboQuant ({mem_tq})");

        // MarkovRS W=512 is bounded by the hot window (~192 MB) regardless of seq_len.
        // At short contexts (<~11K) the window dominates and MarkovRS > TurboQuant.
        // At long contexts TurboQuant grows larger. Both beat standard KV.
        assert!(mem_std > mem_mrk, "At {seq_len}: Standard ({mem_std}) > Markov RS ({mem_mrk})");

        // BoundaryRS W=32 is always the smallest storage (besides graph walk).
        // Hot window is fixed ~11 MB; cold grows at 4 bytes/token.
        assert!(mem_brs < mem_mrk, "At {seq_len}: Boundary RS ({mem_brs}) < Markov RS ({mem_mrk})");
        assert!(mem_brs < mem_tq,  "At {seq_len}: Boundary RS ({mem_brs}) < TurboQuant ({mem_tq})");

        // Graph Walk is the absolute minimum (vindex lookup, no K/V stored).
        assert!(mem_gw < mem_brs,  "At {seq_len}: Graph Walk ({mem_gw}) < Boundary RS ({mem_brs})");
    }

    // At very long contexts, MarkovRS stays flat while TurboQuant grows O(n).
    // Crossover: MarkovRS fixed window (~192 MB) < TurboQuant at ~11K+ tokens.
    let mem_mrk_370k = markov.memory_bytes(&config, 370_000) as f64;
    let mem_tq_370k  = tq4.memory_bytes(&config, 370_000) as f64;
    assert!(mem_tq_370k > mem_mrk_370k,
        "At 370K: TurboQuant ({mem_tq_370k:.0}) should exceed Markov RS ({mem_mrk_370k:.0})");
}

#[test]
fn test_memory_sweep_produces_data() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov, &graph];
    let lengths = &[512, 4096, 32768];

    let points = benchmark::memory_sweep(&config, &strategies, lengths);

    // 4 strategies × 3 lengths = 12 points
    assert_eq!(points.len(), 12);

    for point in &points {
        assert!(point.memory_bytes > 0, "Zero memory for {}", point.strategy_name);
    }
}

#[test]
fn test_comparative_table_format() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov, &graph];
    let table = benchmark::format_comparative_table(&config, &strategies);

    assert!(table.contains("Gemma 3-4B"));
    assert!(table.contains("ELIMINATED"));
    assert!(table.contains("Standard KV"));
    assert!(table.contains("TurboQuant"));
    assert!(table.contains("Markov RS"));
    assert!(table.contains("Graph Walk"));
}

#[test]
fn test_370k_memory_ratios() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    let seq_len = 370_000;
    let mem_std = standard.memory_bytes(&config, seq_len) as f64;
    let mem_tq = tq4.memory_bytes(&config, seq_len) as f64;
    let mem_mrk = markov.memory_bytes(&config, seq_len) as f64;
    let mem_gw = graph.memory_bytes(&config, seq_len) as f64;

    let ratio_tq = mem_std / mem_tq;
    let ratio_mrk = mem_std / mem_mrk;
    let ratio_gw = mem_std / mem_gw;

    // TurboQuant: 4-6× compression
    assert!(ratio_tq > 2.0, "TQ ratio: {ratio_tq:.1}×");
    assert!(ratio_tq < 8.0, "TQ ratio: {ratio_tq:.1}×");

    // Markov RS: 100×+ compression
    assert!(ratio_mrk > 100.0, "Markov ratio: {ratio_mrk:.1}×");

    // Graph Walk: even more (same cold tier, no window overhead)
    assert!(ratio_gw > ratio_mrk, "Graph Walk should compress more than Markov RS");

    println!("At 370K tokens on {}:", config.name);
    println!("  Standard KV:   {:.1} GB", mem_std / 1e9);
    println!("  TurboQuant 4b: {:.1} GB ({ratio_tq:.1}×)", mem_tq / 1e9);
    println!("  Markov RS:     {:.1} MB ({ratio_mrk:.0}×)", mem_mrk / 1e6);
    println!("  Graph Walk:    {:.1} MB ({ratio_gw:.0}×)", mem_gw / 1e6);
}

#[test]
fn test_multi_model_memory() {
    let models = ModelConfig::all();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);

    for config in &models {
        let std_4k = standard.memory_bytes(config, 4096);
        let tq_4k = tq4.memory_bytes(config, 4096);
        assert!(
            std_4k > tq_4k,
            "{}: Standard ({std_4k}) should > TurboQuant ({tq_4k})",
            config.name
        );
    }
}

