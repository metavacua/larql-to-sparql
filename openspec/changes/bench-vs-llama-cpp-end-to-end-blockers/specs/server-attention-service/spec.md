## ADDED Requirements

### Requirement: OpenAI-compat decode endpoints SHALL respond in bounded time

`POST /v1/chat/completions` and `POST /v1/infer` SHALL return a response within a deterministic, finite time bound on any vindex that `larql serve` successfully loads (i.e. boots without panicking and reports `loaded.inference: true` via `/v1/stats`).

If the server detects that the loaded vindex is missing a structure the decode path requires (e.g. `down_features_q4k.bin` when the route depends on feature-major down), it MUST fail-fast with HTTP 4xx/5xx and a clear error message in the response body. Silent hangs at sub-2 % CPU utilisation are a contract violation.

#### Scenario: Chat completion completes or fails on a loadable Gemma 3 vindex

- **WHEN** `larql serve <vindex>` accepts a vindex (boot reports `loaded.inference: true` and `/v1/health` returns ok)
- **AND** the client POSTs to `/v1/chat/completions` with a minimal `{"messages":..., "max_tokens": 5}` body
- **THEN** the server SHALL return a non-empty response within a configurable timeout (default ≤ 60 s on a 4 B-class model)
- **AND** if it cannot, SHALL return an HTTP error with a diagnostic body explaining the missing capability or resource
<!-- test: unbacked -->

#### Scenario: Bootstrap log SHALL flag inference-blocking gaps

- **WHEN** the server boots a vindex that is missing a structure required by `/v1/chat/completions` (e.g. `down_features_q4k.bin` for routes that require feature-major down)
- **THEN** `/v1/stats.loaded.inference` SHALL report `false` (not `true`), OR the bootstrap log SHALL include a `WARN`/`ERROR` line naming the missing component
- **AND** any POST to `/v1/chat/completions` SHALL fail fast with a corresponding error
<!-- test: unbacked -->
