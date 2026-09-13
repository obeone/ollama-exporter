# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks requests per model and endpoint**, broken down by outcome (`ollama_requests_total`)
- **Covers both API schemas**: Ollama's native endpoints and its OpenAI-compatible ones (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`), which is where most clients actually land
- **Measures response time**, including time to first token for streaming replies
- **Reports requests currently in flight** (`ollama_requests_in_flight`)
- **Records model load times** (`ollama_load_duration_seconds`)
- **Tracks evaluation durations** (`ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds`)
- **Monitors token usage** (`ollama_tokens_processed_total` and `ollama_tokens_generated_total`)
- **Measures token generation rate** (`ollama_tokens_per_second`)
- **Publishes model residency**: which models are resident, their VRAM footprint, and their `keep_alive` countdown, by polling `GET /api/ps`
- **Reports upstream liveness** (`ollama_upstream_up`)
- **Transparent proxy** for all other Ollama API endpoints

## Installation

### Running Locally

#### 1. Install Dependencies
```sh
pip install fastapi uvicorn prometheus_client httpx
```

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```
By default, it connects to `http://localhost:11434` for Ollama and listens on
`[::]:8000`, a dual-stack socket reachable over both IPv6 and IPv4.

#### 3. Run the Tests (optional)
```sh
pip install -r requirements-dev.txt
pytest
```

### Running with Docker

#### 1. Build the Docker Image
```sh
docker build -t ollama-exporter .
```

#### 2. Run the Container
```sh
docker run -d --name ollama-exporter -p 8000:8000 \
  -e OLLAMA_HOST="http://192.168.1.100:11434" ollama-exporter
```

Arguments appended to `docker run` are passed straight to the exporter CLI, so
the flags of the local install work the same way in a container:
```sh
docker run -d --name ollama-exporter -p 9000:9000 ollama-exporter \
  --ollama-host http://192.168.1.100:11434 --port 9000 --log-level DEBUG
```
When overriding the port, also set `-e EXPORTER_PORT=9000` so the container
healthcheck probes the right one.

## Prometheus Integration

### Add to `prometheus.yml`
```yaml
scrape_configs:
  - job_name: 'ollama-metrics'
    static_configs:
      - targets: ['192.168.1.100:8000']
```
Restart Prometheus to apply changes:
```sh
docker restart <prometheus-container-name>
```

## Metrics

Every request-scoped metric carries an `endpoint` label: `/api/chat`, `/api/generate`, `/api/embed`, `/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`, or `other` for anything else. `other` is a bounded fallback, not a per-path breakdown; unrecognised paths never leak into the label value.

### Request metrics

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `ollama_requests_total` | Counter | `model`, `endpoint`, `status` | Generation requests counted on completion, by outcome |
| `ollama_requests_in_flight` | Gauge | `model`, `endpoint` | Generation requests currently being served |
| `ollama_response_seconds` | Histogram | `model`, `endpoint` | Total time spent for the response |
| `ollama_time_to_first_token_seconds` | Histogram | `model`, `endpoint` | Delay before the first bytes of a streaming reply (streaming only) |
| `ollama_load_duration_seconds` | Histogram | `model`, `endpoint` | Time spent loading the model |
| `ollama_prompt_eval_duration_seconds` | Histogram | `model`, `endpoint` | Time spent evaluating the prompt |
| `ollama_eval_duration_seconds` | Histogram | `model`, `endpoint` | Time spent generating the response |
| `ollama_tokens_processed_total` | Counter | `model`, `endpoint` | Number of tokens in the prompt |
| `ollama_tokens_generated_total` | Counter | `model`, `endpoint` | Number of tokens in the response |
| `ollama_prompt_tokens` | Histogram | `model`, `endpoint` | Distribution of prompt sizes in tokens |
| `ollama_generated_tokens` | Histogram | `model`, `endpoint` | Distribution of response sizes in tokens |
| `ollama_tokens_per_second` | Histogram | `model`, `endpoint` | Tokens generated per second |
| `ollama_usage_missing_total` | Counter | `model`, `endpoint` | Streaming OpenAI-compatible replies that ended without a `usage` block |

`ollama_requests_total` is incremented once the request finishes, with `status` set to one of:

| Value | Meaning |
|-------|---------|
| `success` | Upstream answered with a 2xx status |
| `client_error` | Upstream answered with a 4xx status |
| `server_error` | Upstream answered with a 5xx status |
| `aborted` | The client disconnected mid-stream |
| `upstream_error` | The exporter could not reach Ollama |

### What the OpenAI-compatible endpoints can and cannot report

The native endpoints answer with Ollama's own schema, which carries nanosecond timings for the model load, the prompt evaluation and the generation. The `/v1/*` endpoints answer with the OpenAI schema, which carries none of them: only a `usage` object with `prompt_tokens` and `completion_tokens`.

So on `/v1/*` the exporter records the request counter, the status, the in-flight gauge, the response time and the time to first token (all measured by the exporter itself), plus the token counts and a tokens-per-second figure derived from them. `ollama_load_duration_seconds`, `ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds` take **no observation at all** on this path. Feeding them a zero would be easy and wrong: it would pull every quantile and every `rate(_sum)/rate(_count)` average towards zero, including the ones the native endpoints populate correctly. A missing observation says "not measured here", which is the truth.

One more gap worth knowing about. On a streaming `/v1/*` request, Ollama only sends the `usage` object when the client sets `stream_options: {"include_usage": true}`. Most do not, and their tokens are then counted nowhere. The exporter does not rewrite the request to force the option, since that would change the event stream the client receives; it increments `ollama_usage_missing_total` instead, so the undercount is visible rather than silent. Comparing `rate(ollama_usage_missing_total[5m])` against `rate(ollama_requests_total[5m])` on the same endpoint tells you how much of your token accounting is guesswork.

### Model residency metrics

Fed by a background task that polls `GET /api/ps`, so these gauges update at most once every `OLLAMA_PS_INTERVAL_SECONDS` (see [Configuration](#configuration)) rather than on every request.

| Metric Name | Type | Labels | Description |
|------------|------|--------|-------------|
| `ollama_model_loaded` | Gauge | `model` | 1 while the model occupies VRAM; the series disappears once it is evicted |
| `ollama_model_vram_bytes` | Gauge | `model` | Bytes of VRAM the resident model occupies |
| `ollama_model_size_bytes` | Gauge | `model` | Total size of the resident model, VRAM and host memory combined |
| `ollama_model_context_length` | Gauge | `model` | Context window the model was loaded with, in tokens |
| `ollama_model_expires_seconds` | Gauge | `model` | Seconds left on the `keep_alive` countdown before eviction; goes negative once expired |
| `ollama_models_loaded` | Gauge | none | Number of models currently resident |
| `ollama_model_swaps_total` | Counter | `model` | Times a model entered the resident set, i.e. was loaded from cold |
| `ollama_model_info` | Gauge | `model`, `family`, `parameter_size`, `quantization_level` | Always 1; a join target for the other series |
| `ollama_upstream_up` | Gauge | none | 1 when the last `/api/ps` poll reached the Ollama server |

One metric sits outside both groups: `ollama_exporter_build_info` is a gauge always set to 1, whose `version` label carries the running exporter build so a dashboard can tell which one answered the scrape.

## Configuration

Every setting has an environment variable and a matching CLI flag. CLI flags always win over environment variables.

| Environment Variable | CLI Flag | Default | Description |
|----------------------|----------|---------|-------------|
| `OLLAMA_HOST` | `--ollama-host` | `http://localhost:11434` | Base URL of the upstream Ollama server |
| `EXPORTER_HOST` | `--host` | `::` | Address the exporter binds to |
| `EXPORTER_PORT` | `--port` | `8000` | Port the exporter listens on |
| `OLLAMA_PS_INTERVAL_SECONDS` | `--ps-interval` | `15` | Seconds between two `/api/ps` residency polls; `0` disables the poller |
| `LOG_LEVEL` | `--log-level` | `INFO` | Logging verbosity |
| `EXPORTER_VERSION` | none | unset | Overrides the version reported by `ollama_exporter_build_info` |

## Grafana Integration
1. Open **Grafana**.
2. Go to **Dashboards → Import**.
3. Click **Upload JSON file** and select `dashboard.json` from the project directory.
4. Select your **Prometheus data source**.
5. Click **Import** to add the dashboard.

`dashboard.json` and `dashboard_custom.json` in this repository are stale snapshots kept for bootstrapping a new install; the live Grafana instance is the canonical source of truth for both dashboards.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Exposes Prometheus metrics |
| `/api/chat` | POST | Proxies requests to Ollama and logs metrics |
| `/api/generate` | POST | Proxies requests to Ollama and logs metrics |
| `/api/embed` | POST | Proxies requests to Ollama and logs metrics |

All other endpoints are proxied to the Ollama API.

## Usage Scenario

Suppose you want to monitor your local Ollama instance with Prometheus using this exporter:

1. **Start Ollama** locally (default: `http://localhost:11434`).
2. **Run the exporter** on your machine:
   ```sh
   OLLAMA_HOST=http://localhost:11434 python ollama_exporter.py
   # or with Docker:
   # docker run -d -p 8000:8000 -e OLLAMA_HOST="http://localhost:11434" ollama-exporter
   ```
3. **Configure your application (eg Open WebUI)** to use the exporter as the API endpoint:
   - Set `OLLAMA_HOST=http://localhost:8000` (the exporter will proxy and collect metrics).
4. **Prometheus** scrapes metrics from the exporter:
   - Add `localhost:8000/metrics` to your Prometheus scrape config.

This setup allows you to transparently monitor all Ollama API usage and performance via Prometheus and Grafana dashboards.

## License
There is no spoon.
