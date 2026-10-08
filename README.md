# Ollama Prometheus Exporter

This is a **Prometheus Exporter** for **Ollama**, designed to monitor request statistics, response times, token usage, and model performance. It runs as a FastAPI service and is **Docker-ready**.

## Features
- **Tracks requests per model** (`ollama_requests_total`)
- **Measures response time** (`ollama_response_seconds`)
- **Records model load times** (`ollama_load_duration_seconds`)
- **Tracks evaluation durations** (`ollama_prompt_eval_duration_seconds` and `ollama_eval_duration_seconds`)
- **Monitors token usage** (`ollama_tokens_processed_total` and `ollama_tokens_generated_total`)
- **Measures token generation rate** (`ollama_tokens_per_second`)
- **Tracks in-flight inference requests** (`ollama_inflight_requests`)
- **Transparent streaming proxy** for every Ollama API endpoint, including the
  OpenAI-compatible `/v1/*` API
- **Cancels abandoned generations**: when a client disconnects, the upstream
  request is closed and Ollama stops generating
- **Total deadline per inference request**, so no generation can run unbounded

## Installation

### Running Locally

#### 1. Install Dependencies
```sh
pip install -r requirements.txt
```

#### 2. Run the Exporter
```sh
python ollama_exporter.py
```
By default, it connects to `http://localhost:11434` for Ollama and listens on
`[::]:8000`, a dual-stack socket reachable over both IPv6 and IPv4.

#### Configuration
Every option can be set by environment variable or CLI flag (the flag wins).

| Environment variable | Flag | Default | Description |
|----------------------|------|---------|-------------|
| `OLLAMA_HOST` | `--ollama-host` | `http://localhost:11434` | Upstream Ollama URL |
| `EXPORTER_HOST` | `--host` | `::` | Address to bind (dual-stack) |
| `EXPORTER_PORT` | `--port` | `8000` | Port to listen on |
| `EXPORTER_REQUEST_TIMEOUT` | `--request-timeout` | `1800` | Total deadline in seconds for one inference request, `0` disables it |
| `LOG_LEVEL` | `--log-level` | `INFO` | Logging verbosity |

The request deadline covers the whole exchange, from sending the request to
the last streamed byte. It applies to inference only: `POST` on `/api/chat`,
`/api/generate`, `/api/embed`, `/api/embeddings` and any `/v1/*` endpoint.
Model pulls, pushes and listings are not limited. A request that hits the
deadline before Ollama answered gets a `504`; one that is already streaming
has its stream ended. Either way the upstream connection is closed, which is
what makes Ollama stop generating.

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
| Metric Name | Description |
|------------|-------------|
| `ollama_requests_total` | Total chat and generate requests (`/api/chat`, `/api/generate`, `/v1/chat/completions`, `/v1/completions`) |
| `ollama_client_requests_total` | Inference requests per source client: chat, generate, embeddings and the OpenAI-compatible `/v1` endpoints (optional, labels `model`, `client`, `hostname`, see [Per-client metrics](#per-client-metrics)) |
| `ollama_inflight_requests` | Inference requests currently being proxied to Ollama, per model |
| `ollama_response_seconds` | Total time spent for the response |
| `ollama_load_duration_seconds` | Time spent loading the model |
| `ollama_prompt_eval_duration_seconds` | Time spent evaluating prompt |
| `ollama_eval_duration_seconds` | Time spent generating the response |
| `ollama_tokens_processed_total` | Number of tokens in the prompt |
| `ollama_tokens_generated_total` | Number of tokens in the response |
| `ollama_tokens_per_second` | Tokens generated per second |

Durations and tokens per second come from Ollama's native stats, so only
`/api/chat` and `/api/generate` feed them. The OpenAI-compatible
`/v1/chat/completions` and `/v1/completions` feed the two token counters from
their `usage` object; for streamed responses that object is only sent when the
client sets `stream_options.include_usage`.

## Per-client metrics

`ollama_client_requests_total{model, client, hostname}` counts every inference request
(chat, generate, embeddings and the OpenAI-compatible `/v1` endpoints) per source
client. It is disabled by default because a client address label has
unbounded cardinality and can bloat Prometheus on a busy or exposed instance.
Enable it only on networks where the set of clients is small and known.

| Flag | Environment variable | Description |
|------|----------------------|-------------|
| `--track-clients` | `EXPORTER_TRACK_CLIENTS` | Enable the counter (`1`, `true`, `yes` or `on`) |
| `--resolve-clients` | `EXPORTER_RESOLVE_CLIENTS` | Fill `hostname` with the client's reverse DNS name (same truthy values); needs `--track-clients` |
| `--client-hostname-ignore` | `EXPORTER_CLIENT_HOSTNAME_IGNORE` | Case-insensitive regex; resolved names matching it fall back to the IP |
| `--client-hostname-timeout` | `EXPORTER_CLIENT_HOSTNAME_TIMEOUT` | Seconds a reverse lookup may delay the counter increment, greater than 0 (default `3`); a lookup finishing later still fills the cache |
| `--forwarded-allow-ips` | `FORWARDED_ALLOW_IPS` | Comma-separated IPs/CIDRs, or `*`, allowed to set `X-Forwarded-For` (default `127.0.0.1,::1`) |

Behind a reverse proxy, the client is read from `X-Forwarded-For`, but only when
the direct peer is listed in `--forwarded-allow-ips`. IPv4 peers are matched in
their plain form even on the default dual-stack socket. Requests from any other
peer have the header ignored, so a client cannot spoof its address. Trust only
your own proxies, and avoid `*` unless the exporter is unreachable except
through them.

```sh
docker run -d --name ollama-exporter -p 8000:8000 ollama-exporter \
  --ollama-host http://192.168.1.100:11434 \
  --track-clients --forwarded-allow-ips 10.0.0.0/8
```

### Client hostnames

With `--resolve-clients`, the `hostname` label holds the reverse DNS (PTR) name
of `client`, lowercased and without the trailing dot. Lookups run in a small
background thread pool and are bounded: 3 second timeout (`--client-hostname-timeout`), a cache of 4096
addresses (least recently used evicted first), 1 hour for a found name and
5 minutes for a failure. A timeout or a temporary resolver error keeps the
previously known name, so a DNS blip does not flip an established series back
to the IP.

`--client-hostname-ignore` takes a regex, applied with `re.search` and without
case sensitivity to the resolved name. A match falls back to the IP, which is
handy for auto-generated names that say nothing, for example
`--client-hostname-ignore '\.ipv6\.obeone\.org$'`.

Requests never wait for a lookup. The first requests of an unknown client are
normally not counted under the IP: the proxied request goes through
immediately, and the counter increment is deferred until the lookup ends (3
seconds at most by default), landing directly under the final hostname. Every request
increments exactly one series. A series changes hostname if the PTR record
itself changes, or if a client that had no name gains one after the 5 minute
failure cache.

A lookup that takes longer than the timeout is not thrown away: the requests
waiting on it are counted under the IP right away, but the lookup keeps running
and, if it succeeds, its name fills the cache with the normal 1 hour lifetime.
Only the requests counted during that first lookup land under the IP. A late
failure changes nothing.

Known limitation: the timeout includes the time spent queued for one of the 4
lookup threads. If a client's first lookup times out (a burst of many new
clients, or a hung DNS server tying up the workers), the requests counted
before it ends use the IP. A lookup that never finishes leaves the client under
its IP for up to 5 minutes, until the next attempt. While such a lookup is
still running, no second lookup is started for the same IP, so a DNS hang
queues at most one job per distinct client.

The `hostname` label is always present and equals `client` when resolution is
off. Upgrading therefore starts new series once, and queries such as
`sum by (model, client)` keep working.

```sh
docker run -d --name ollama-exporter -p 8000:8000 ollama-exporter \
  --ollama-host http://192.168.1.100:11434 \
  --track-clients --resolve-clients \
  --client-hostname-ignore '\.ipv6\.obeone\.org$'
```

## Grafana Integration
1. Open **Grafana**.
2. Go to **Dashboards → Import**.
3. Click **Upload JSON file** and select `dashboard.json` from the project directory.
4. Select your **Prometheus data source**.
5. Click **Import** to add the dashboard.

## API Endpoints
| Endpoint | Method | Description |
|----------|--------|-------------|
| `/metrics` | GET | Exposes Prometheus metrics |
| `/api/chat` | POST | Proxies requests to Ollama and logs metrics |
| `/api/generate` | POST | Proxies requests to Ollama and logs metrics |
| `/v1/chat/completions`, `/v1/completions` | POST | Proxies requests to Ollama and logs request and token metrics |
| `/api/embed`, `/api/embeddings`, `/v1/embeddings` | POST | Proxies requests to Ollama and feeds the per-client counter and the in-flight gauge (not `ollama_requests_total`) |

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
