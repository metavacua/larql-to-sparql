use kv_cache_benchmark::*;
use kv_cache_benchmark::benchmark;
use kv_cache_benchmark::model_config::ModelConfig;
use kv_cache_benchmark::standard_kv::StandardKv;
use kv_cache_benchmark::turboquant::TurboQuant;
use kv_cache_benchmark::markov_residual::MarkovResidual;
use kv_cache_benchmark::hybrid_cracked::HybridCrackedAttention;
use kv_cache_benchmark::graph_walk::GraphWalk;

#[test]
fn test_all_strategies_memory_ordering() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let graph = GraphWalk::gemma_4b();

    for &seq_len in &[4096, 32768, 370_000] {
        let mem_std = standard.memory_bytes(&config, seq_len);
        let mem_tq = tq4.memory_bytes(&config, seq_len);
        let mem_mrk = markov.memory_bytes(&config, seq_len);
        let mem_gw = graph.memory_bytes(&config, seq_len);

        // Ordering: Standard > TurboQuant > Markov RS > Graph Walk (per-conversation)
        assert!(
            mem_std > mem_tq,
            "At {seq_len}: Standard ({mem_std}) should > TurboQuant ({mem_tq})"
        );
        assert!(
            mem_tq > mem_mrk,
            "At {seq_len}: TurboQuant ({mem_tq}) should > Markov RS ({mem_mrk})"
        );
        // Graph Walk per-conversation is same as Markov RS cold tier
        assert!(
            mem_mrk >= mem_gw,
            "At {seq_len}: Markov RS ({mem_mrk}) should >= Graph Walk ({mem_gw})"
        );
    }
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

#[test]
fn test_five_strategy_memory_ordering() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let hybrid = HybridCrackedAttention::gemma_4b();
    let graph = GraphWalk::gemma_4b();

    let seq_len = 4096;
    let mem_std = standard.memory_bytes(&config, seq_len);
    let mem_tq = tq4.memory_bytes(&config, seq_len);
    let mem_hyb = hybrid.memory_bytes(&config, seq_len);
    let mem_gw = graph.memory_bytes(&config, seq_len);

    // Standard > TurboQuant > Hybrid > Graph Walk
    assert!(mem_std > mem_tq, "Standard > TurboQuant");
    assert!(mem_tq > mem_hyb, "TurboQuant > Hybrid");
    assert!(mem_hyb > mem_gw, "Hybrid > Graph Walk");
}

#[test]
fn test_five_strategy_table_format() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let hybrid = HybridCrackedAttention::gemma_4b();
    let graph = GraphWalk::gemma_4b();

    let strategies: Vec<&dyn KvStrategy> = vec![&standard, &tq4, &markov, &hybrid, &graph];
    let table = benchmark::format_comparative_table(&config, &strategies);

    assert!(table.contains("Hybrid RS+CA"));
    assert!(table.contains("ZERO (vindex)"));
    assert!(table.contains("~1-2L dynamic"));
}

#[test]
fn test_370k_five_strategy_ratios() {
    let config = ModelConfig::gemma_4b();
    let standard = StandardKv;
    let tq4 = TurboQuant::new(4);
    let markov = MarkovResidual::new(512);
    let hybrid = HybridCrackedAttention::gemma_4b();
    let graph = GraphWalk::gemma_4b();

    let seq_len = 370_000;
    let mem_std = standard.memory_bytes(&config, seq_len) as f64;
    let mem_tq = tq4.memory_bytes(&config, seq_len) as f64;
    let mem_mrk = markov.memory_bytes(&config, seq_len) as f64;
    let mem_hyb = hybrid.memory_bytes(&config, seq_len) as f64;
    let mem_gw = graph.memory_bytes(&config, seq_len) as f64;

    println!("At 370K tokens on {}:", config.name);
    println!("  Standard KV:   {:.1} GB", mem_std / 1e9);
    println!("  TurboQuant 4b: {:.1} GB ({:.1}x)", mem_tq / 1e9, mem_std / mem_tq);
    println!("  Markov RS:     {:.1} MB ({:.0}x)", mem_mrk / 1e6, mem_std / mem_mrk);
    println!("  Hybrid RS+CA:  {:.1} MB ({:.0}x)", mem_hyb / 1e6, mem_std / mem_hyb);
    println!("  Graph Walk:    {:.1} MB ({:.0}x)", mem_gw / 1e6, mem_std / mem_gw);

    // Hybrid should be between TQ and Markov RS in compression
    assert!(mem_std / mem_hyb > 5.0, "Hybrid compression too low");
}
