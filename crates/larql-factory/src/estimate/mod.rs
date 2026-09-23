//! `larql recipe estimate` — §6.1 step 4's PR-check estimate: upstream
//! download size, per-output size, executor recommendation, and a cost
//! band, so a recipe PR's approval is informed rather than blind.
//!
//! Split into a thin I/O half ([`estimate`]) and a pure computation
//! half ([`compute`]) on purpose — the interesting logic is fully
//! unit-testable without a network call.

mod bytes;
mod cost;
mod dims;
mod executor;
mod http;
mod preset_weights;

pub use dims::ModelDims;
pub use executor::ExecutorClass;
pub use http::HttpError;

use serde::Serialize;

use crate::Recipe;

const DOWN_Q4K_OPTION_KEY: &str = "down_q4k";

/// One declared output's estimated size.
#[derive(Debug, Serialize)]
pub struct OutputEstimate {
    /// The output's `preset` name, as declared in the recipe.
    pub preset: String,
    /// Estimated size in bytes.
    pub estimated_bytes: u64,
}

/// The full `larql recipe estimate` result.
#[derive(Debug, Serialize)]
pub struct SizeEstimate {
    /// Total bytes across the upstream repo's weight files.
    pub upstream_bytes: u64,
    /// Per-output estimated size.
    pub outputs: Vec<OutputEstimate>,
    /// Sum of every output's estimated size.
    pub total_estimated_output_bytes: u64,
    /// Executor class this estimate recommends.
    pub recommended_executor: ExecutorClass,
    /// The threshold (§13.4 PIN) the recommendation was made against.
    pub executor_threshold_gib: f64,
    /// What the recipe itself declares (`auto`/`inline`/`leased`).
    pub declared_executor: String,
    /// Cost band if the recipe's declared `max_wall_minutes` were
    /// fully used on the recommended executor.
    pub cost_estimate_usd: cost::CostRange,
    /// The recipe's declared `budget.max_usd`.
    pub declared_max_usd: f64,
    /// Set when `declared_max_usd` looks too low for the estimated
    /// cost band — a likely recipe-authoring mistake, not a hard error.
    pub budget_warning: Option<String>,
}

/// Fetch upstream data over the network and compute the estimate.
pub fn estimate(recipe: &Recipe) -> Result<SizeEstimate, HttpError> {
    estimate_from_base_url(recipe, http::HF_API_BASE)
}

/// Same as [`estimate`], but against an arbitrary API base — the seam
/// tests point at a local mock server through, so the real fetch
/// sequence (two calls, dims derived from the second) is exercised
/// without a network dependency.
fn estimate_from_base_url(recipe: &Recipe, base_url: &str) -> Result<SizeEstimate, HttpError> {
    let client = reqwest::blocking::Client::new();
    let token = http::hf_token();
    let upstream_bytes = http::fetch_upstream_bytes(
        &client,
        base_url,
        &recipe.spec.source.hf_repo,
        &recipe.spec.source.revision,
        token.as_deref(),
    )?;
    let config_json = http::fetch_config_json(
        &client,
        base_url,
        &recipe.spec.source.hf_repo,
        &recipe.spec.source.revision,
        token.as_deref(),
    )?;
    let dims = ModelDims::from_config_json(&config_json);
    Ok(compute(recipe, &dims, upstream_bytes))
}

/// Pure computation — given already-fetched dims and upstream bytes,
/// produce the estimate. Fully unit-testable without network access.
pub fn compute(recipe: &Recipe, dims: &ModelDims, upstream_bytes: u64) -> SizeEstimate {
    let down_q4k = recipe
        .spec
        .extractor
        .options
        .get(DOWN_Q4K_OPTION_KEY)
        .and_then(|v| v.as_bool())
        .unwrap_or(false);
    let components = bytes::estimate_component_bytes(dims, recipe.spec.extractor.dtype, down_q4k);

    let outputs: Vec<OutputEstimate> = recipe
        .spec
        .outputs
        .iter()
        .map(|o| OutputEstimate {
            preset: o.preset.clone(),
            estimated_bytes: preset_weights::estimate_preset_bytes(&o.preset, &components),
        })
        .collect();

    let total_estimated_output_bytes: u64 = outputs.iter().map(|o| o.estimated_bytes).sum();

    // Upper bound, not a peak-usage model: every declared output plus
    // the upstream download, as if all were resident at once. Errs
    // toward recommending `leased` over `inline` when uncertain, which
    // is the safer direction to be wrong in.
    let total_working_set_bytes = upstream_bytes + total_estimated_output_bytes;
    let recommended_executor = executor::recommend_executor(total_working_set_bytes);

    let cost_estimate_usd = cost::estimated_cost_at_declared_wall_time(
        recommended_executor,
        recipe.spec.budget.max_wall_minutes,
    );

    let budget_warning = (recipe.spec.budget.max_usd < cost_estimate_usd.low_usd).then(|| {
        format!(
            "declared budget.max_usd (${:.2}) is below the low end of the estimated cost (${:.2}-${:.2}) at the recommended '{}' tier",
            recipe.spec.budget.max_usd,
            cost_estimate_usd.low_usd,
            cost_estimate_usd.high_usd,
            recommended_executor.as_str()
        )
    });

    SizeEstimate {
        upstream_bytes,
        outputs,
        total_estimated_output_bytes,
        recommended_executor,
        executor_threshold_gib: executor::INLINE_EXECUTOR_THRESHOLD_GIB,
        declared_executor: recipe.spec.budget.executor.clone(),
        cost_estimate_usd,
        declared_max_usd: recipe.spec.budget.max_usd,
        budget_warning,
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    /// Like `http::tests::spawn_mock_server`, but replies to each
    /// successive connection with the next response in `responses` —
    /// `estimate_from_base_url` makes two sequential calls (tree
    /// listing, then config.json), in that order.
    fn spawn_mock_server_with_responses(responses: Vec<&'static str>) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().expect("local_addr").port();
        std::thread::spawn(move || {
            for response in responses {
                if let Ok((mut stream, _)) = listener.accept() {
                    let mut buf = [0u8; 1024];
                    let _ = stream.read(&mut buf);
                    let _ = stream.write_all(response.as_bytes());
                }
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    fn http_ok_json(body: String) -> String {
        format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        )
    }

    #[test]
    fn estimate_from_base_url_fetches_then_computes() {
        let tree_body = r#"[{"path":"model.safetensors","size":5000,"type":"file"}]"#.to_string();
        let config_body = r#"{"model_type":"llama","hidden_size":2560,"intermediate_size":10240,"num_hidden_layers":34,"vocab_size":262208}"#.to_string();
        let base_url = spawn_mock_server_with_responses(vec![
            Box::leak(http_ok_json(tree_body).into_boxed_str()),
            Box::leak(http_ok_json(config_body).into_boxed_str()),
        ]);

        let result = estimate_from_base_url(&sample_recipe(), &base_url).unwrap();
        assert_eq!(result.upstream_bytes, 5000);
        assert!(!result.outputs.is_empty());
    }

    #[test]
    fn estimate_from_base_url_propagates_the_first_fetch_error() {
        let base_url = spawn_mock_server_with_responses(vec![
            "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
        ]);
        let err = estimate_from_base_url(&sample_recipe(), &base_url).unwrap_err();
        assert!(matches!(err, HttpError::Status { status: 404, .. }));
    }

    use crate::test_support::sample_recipe;

    fn sample_dims() -> ModelDims {
        ModelDims {
            num_layers: 34,
            hidden_size: 2560,
            intermediate_size: 10240,
            vocab_size: 262_208,
            num_experts: None,
        }
    }

    #[test]
    fn compute_produces_one_output_estimate_per_declared_output() {
        let recipe = sample_recipe();
        let result = compute(&recipe, &sample_dims(), 8_000_000_000);
        assert_eq!(result.outputs.len(), recipe.spec.outputs.len());
    }

    #[test]
    fn total_estimated_output_bytes_sums_every_output() {
        let recipe = sample_recipe();
        let result = compute(&recipe, &sample_dims(), 8_000_000_000);
        let sum: u64 = result.outputs.iter().map(|o| o.estimated_bytes).sum();
        assert_eq!(result.total_estimated_output_bytes, sum);
    }

    #[test]
    fn small_model_recommends_inline() {
        let recipe = sample_recipe();
        let result = compute(&recipe, &sample_dims(), 8_000_000_000);
        assert_eq!(result.recommended_executor, ExecutorClass::Inline);
    }

    #[test]
    fn huge_upstream_recommends_leased() {
        let recipe = sample_recipe();
        let huge_bytes = 200u64 * 1024 * 1024 * 1024; // 200 GiB
        let result = compute(&recipe, &sample_dims(), huge_bytes);
        assert_eq!(result.recommended_executor, ExecutorClass::Leased);
    }

    #[test]
    fn declared_executor_reflects_the_recipe_verbatim() {
        let recipe = sample_recipe();
        let result = compute(&recipe, &sample_dims(), 8_000_000_000);
        assert_eq!(result.declared_executor, recipe.spec.budget.executor);
    }

    #[test]
    fn no_budget_warning_when_declared_max_usd_covers_the_estimate() {
        let recipe = sample_recipe();
        let result = compute(&recipe, &sample_dims(), 8_000_000_000);
        // Sample recipe is inline-class (free), so any positive max_usd covers it.
        assert!(result.budget_warning.is_none());
    }

    #[test]
    fn budget_warning_fires_when_declared_max_usd_is_too_low_for_leased() {
        let mut recipe = sample_recipe();
        recipe.spec.budget.max_usd = 0.01;
        recipe.spec.budget.max_wall_minutes = 240;
        let huge_bytes = 200u64 * 1024 * 1024 * 1024;
        let result = compute(&recipe, &sample_dims(), huge_bytes);
        assert_eq!(result.recommended_executor, ExecutorClass::Leased);
        assert!(result.budget_warning.is_some());
        assert!(result.budget_warning.unwrap().contains("max_usd"));
    }

    #[test]
    fn down_q4k_option_shrinks_ffn_heavy_output_estimates() {
        let mut recipe = sample_recipe();
        recipe
            .spec
            .extractor
            .options
            .insert(DOWN_Q4K_OPTION_KEY.into(), serde_json::Value::Bool(false));
        let without_q4k = compute(&recipe, &sample_dims(), 8_000_000_000);

        recipe
            .spec
            .extractor
            .options
            .insert(DOWN_Q4K_OPTION_KEY.into(), serde_json::Value::Bool(true));
        let with_q4k = compute(&recipe, &sample_dims(), 8_000_000_000);

        let full_without = without_q4k
            .outputs
            .iter()
            .find(|o| o.preset == "full")
            .unwrap();
        let full_with = with_q4k
            .outputs
            .iter()
            .find(|o| o.preset == "full")
            .unwrap();
        assert!(full_with.estimated_bytes < full_without.estimated_bytes);
    }
}
