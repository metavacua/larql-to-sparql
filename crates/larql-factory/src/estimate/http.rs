//! The estimate's only network I/O: list an upstream repo's weight
//! files (for total bytes) and fetch its `config.json` (for dims).
//!
//! `base_url` is a parameter rather than baked into every function so
//! tests can point at a local mock server instead of the real
//! HuggingFace API — see the `tests` module for a dependency-free
//! `TcpListener`-based one.

use serde::Deserialize;

/// Real HuggingFace API base — the only call site that should ever
/// pass this; everything else takes `base_url` as a parameter.
pub const HF_API_BASE: &str = "https://huggingface.co";

const HF_TOKEN_ENV: &str = "HF_TOKEN";
const SAFETENSORS_EXT: &str = ".safetensors";
const PYTORCH_BIN_EXT: &str = ".bin";
const TREE_FILE_TYPE: &str = "file";

/// Errors from the two fetch calls. Carries the URL so a failure is
/// diagnosable without re-deriving it from the recipe.
#[derive(Debug, thiserror::Error)]
pub enum HttpError {
    /// The request itself failed (DNS, TLS, connection reset, ...).
    #[error("network request to {url} failed: {source}")]
    Request {
        /// URL that was requested.
        url: String,
        /// Underlying reqwest error.
        source: reqwest::Error,
    },
    /// The server responded, but not with 2xx.
    #[error("{url} returned HTTP {status}")]
    Status {
        /// URL that was requested.
        url: String,
        /// HTTP status code.
        status: u16,
    },
    /// The response body wasn't parseable as the expected JSON shape.
    #[error("parsing response from {url} failed: {source}")]
    Parse {
        /// URL that was requested.
        url: String,
        /// Underlying reqwest/serde error.
        source: reqwest::Error,
    },
}

#[derive(Debug, Deserialize)]
struct TreeEntry {
    path: String,
    #[serde(default)]
    size: u64,
    #[serde(rename = "type", default)]
    entry_type: String,
}

/// Read an HF read token from `HF_TOKEN`. No `~/.huggingface/token`
/// fallback yet — gated repos need `HF_TOKEN` set explicitly for now.
pub fn hf_token() -> Option<String> {
    std::env::var(HF_TOKEN_ENV)
        .ok()
        .filter(|t| !t.trim().is_empty())
}

fn tree_url(base_url: &str, hf_repo: &str, revision: &str) -> String {
    format!("{base_url}/api/models/{hf_repo}/tree/{revision}")
}

fn config_url(base_url: &str, hf_repo: &str, revision: &str) -> String {
    format!("{base_url}/{hf_repo}/raw/{revision}/config.json")
}

fn sum_weight_file_bytes(entries: &[TreeEntry]) -> u64 {
    entries
        .iter()
        .filter(|e| {
            e.entry_type == TREE_FILE_TYPE
                && (e.path.ends_with(SAFETENSORS_EXT) || e.path.ends_with(PYTORCH_BIN_EXT))
        })
        .map(|e| e.size)
        .sum()
}

fn apply_auth(
    req: reqwest::blocking::RequestBuilder,
    token: Option<&str>,
) -> reqwest::blocking::RequestBuilder {
    match token {
        Some(t) => req.bearer_auth(t),
        None => req,
    }
}

/// Total bytes across every `.safetensors`/`.bin` file in
/// `hf_repo@revision`'s file tree.
pub fn fetch_upstream_bytes(
    client: &reqwest::blocking::Client,
    base_url: &str,
    hf_repo: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<u64, HttpError> {
    let url = tree_url(base_url, hf_repo, revision);
    let resp = apply_auth(client.get(&url), token)
        .send()
        .map_err(|source| HttpError::Request {
            url: url.clone(),
            source,
        })?;
    if !resp.status().is_success() {
        return Err(HttpError::Status {
            url,
            status: resp.status().as_u16(),
        });
    }
    let entries: Vec<TreeEntry> = resp.json().map_err(|source| HttpError::Parse {
        url: url.clone(),
        source,
    })?;
    Ok(sum_weight_file_bytes(&entries))
}

/// `hf_repo@revision`'s `config.json`, parsed.
pub fn fetch_config_json(
    client: &reqwest::blocking::Client,
    base_url: &str,
    hf_repo: &str,
    revision: &str,
    token: Option<&str>,
) -> Result<serde_json::Value, HttpError> {
    let url = config_url(base_url, hf_repo, revision);
    let resp = apply_auth(client.get(&url), token)
        .send()
        .map_err(|source| HttpError::Request {
            url: url.clone(),
            source,
        })?;
    if !resp.status().is_success() {
        return Err(HttpError::Status {
            url,
            status: resp.status().as_u16(),
        });
    }
    resp.json().map_err(|source| HttpError::Parse {
        url: url.clone(),
        source,
    })
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::net::TcpListener;

    use super::*;

    /// Spawn a one-shot HTTP server on an OS-assigned local port that
    /// replies with `response` (a full HTTP/1.1 response, status line
    /// through body) to the first connection it accepts, then exits.
    /// Dependency-free — just enough HTTP to exercise the real
    /// request/response path without hitting the network.
    fn spawn_mock_server(response: &'static str) -> String {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind mock server");
        let port = listener.local_addr().expect("local_addr").port();
        std::thread::spawn(move || {
            if let Ok((mut stream, _)) = listener.accept() {
                let mut buf = [0u8; 1024];
                let _ = stream.read(&mut buf);
                let _ = stream.write_all(response.as_bytes());
            }
        });
        format!("http://127.0.0.1:{port}")
    }

    #[test]
    #[serial_test::serial(hf_token_env)]
    fn hf_token_reads_the_env_var() {
        std::env::set_var(HF_TOKEN_ENV, "test-token-123");
        assert_eq!(hf_token(), Some("test-token-123".to_string()));
        std::env::remove_var(HF_TOKEN_ENV);
    }

    #[test]
    #[serial_test::serial(hf_token_env)]
    fn hf_token_treats_blank_as_absent() {
        std::env::set_var(HF_TOKEN_ENV, "   ");
        assert_eq!(hf_token(), None);
        std::env::remove_var(HF_TOKEN_ENV);
    }

    #[test]
    fn tree_url_shape() {
        assert_eq!(
            tree_url("https://huggingface.co", "google/gemma-3-4b-it", "main"),
            "https://huggingface.co/api/models/google/gemma-3-4b-it/tree/main"
        );
    }

    #[test]
    fn config_url_shape() {
        assert_eq!(
            config_url("https://huggingface.co", "google/gemma-3-4b-it", "main"),
            "https://huggingface.co/google/gemma-3-4b-it/raw/main/config.json"
        );
    }

    #[test]
    fn sum_weight_file_bytes_filters_to_safetensors_and_bin() {
        let entries = vec![
            TreeEntry {
                path: "model-00001-of-00002.safetensors".into(),
                size: 1000,
                entry_type: "file".into(),
            },
            TreeEntry {
                path: "model-00002-of-00002.safetensors".into(),
                size: 2000,
                entry_type: "file".into(),
            },
            TreeEntry {
                path: "pytorch_model.bin".into(),
                size: 500,
                entry_type: "file".into(),
            },
            TreeEntry {
                path: "README.md".into(),
                size: 999_999,
                entry_type: "file".into(),
            },
            TreeEntry {
                path: "subdir".into(),
                size: 0,
                entry_type: "directory".into(),
            },
        ];
        assert_eq!(sum_weight_file_bytes(&entries), 3500);
    }

    #[test]
    fn fetch_upstream_bytes_sums_the_tree_response() {
        let body = r#"[{"path":"a.safetensors","size":1000,"type":"file"},{"path":"README.md","size":50,"type":"file"}]"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base_url = spawn_mock_server(Box::leak(response.into_boxed_str()));
        let client = reqwest::blocking::Client::new();
        let bytes = fetch_upstream_bytes(&client, &base_url, "owner/repo", "main", None).unwrap();
        assert_eq!(bytes, 1000);
    }

    #[test]
    fn fetch_upstream_bytes_errors_on_non_success_status() {
        let response = "HTTP/1.1 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let base_url = spawn_mock_server(response);
        let client = reqwest::blocking::Client::new();
        let err = fetch_upstream_bytes(&client, &base_url, "owner/repo", "main", None).unwrap_err();
        assert!(matches!(err, HttpError::Status { status: 404, .. }));
    }

    #[test]
    fn fetch_config_json_returns_the_parsed_body() {
        let body = r#"{"model_type":"llama","hidden_size":2560}"#;
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base_url = spawn_mock_server(Box::leak(response.into_boxed_str()));
        let client = reqwest::blocking::Client::new();
        let config = fetch_config_json(&client, &base_url, "owner/repo", "main", None).unwrap();
        assert_eq!(config["model_type"], "llama");
        assert_eq!(config["hidden_size"], 2560);
    }

    #[test]
    fn fetch_config_json_errors_on_malformed_body() {
        let body = "not json";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base_url = spawn_mock_server(Box::leak(response.into_boxed_str()));
        let client = reqwest::blocking::Client::new();
        let err = fetch_config_json(&client, &base_url, "owner/repo", "main", None).unwrap_err();
        assert!(matches!(err, HttpError::Parse { .. }));
    }

    #[test]
    fn fetch_upstream_bytes_sends_bearer_auth_when_token_present() {
        // The mock just needs to respond; this test exercises the
        // `apply_auth` branch without asserting on the header (a full
        // request-inspecting mock is more machinery than this line of
        // code warrants).
        let body = "[]";
        let response = format!(
            "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{}",
            body.len(),
            body
        );
        let base_url = spawn_mock_server(Box::leak(response.into_boxed_str()));
        let client = reqwest::blocking::Client::new();
        let bytes =
            fetch_upstream_bytes(&client, &base_url, "owner/repo", "main", Some("tok")).unwrap();
        assert_eq!(bytes, 0);
    }
}
