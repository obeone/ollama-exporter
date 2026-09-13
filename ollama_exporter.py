"""Prometheus exporter and metrics-extracting reverse proxy for Ollama.

The exporter sits in front of an Ollama server: every request is forwarded
verbatim, while the generation endpoints (``/api/chat``, ``/api/generate``,
``/api/embed``) additionally have their stats object mined for metrics.
"""

import os
import argparse
import asyncio
import httpx
import json
import logging
import socket
import time
from fastapi import FastAPI, Request, Response
from fastapi.responses import StreamingResponse

import uvicorn
from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

# Default values, overridable via environment variables or CLI arguments.
# CLI arguments take precedence over environment variables (see parse_args).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LISTEN_HOST = "::"  # dual-stack: binds both IPv6 and IPv4
DEFAULT_LISTEN_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"

# Fallback when the host has no usable IPv6 stack (see bind_listen_socket).
IPV4_WILDCARD = "0.0.0.0"

# Spellings of the IPv6 wildcard we bind ourselves rather than leaving to
# uvicorn; any other address is unambiguous and asyncio handles it correctly.
IPV6_WILDCARDS = frozenset({"::", "[::]", "::0"})

# Generation can legitimately run for many minutes on a large model, so the
# proxy timeout is deliberately generous.
PROXY_TIMEOUT = httpx.Timeout(900.0, read=900.0)

# Configurable Ollama host. Populated by parse_args(); the env default keeps
# backward compatibility for code paths that import this module directly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Paths this exporter instruments. Anything else reaching normalise_endpoint()
# collapses to a single "other" value: the endpoint label must stay bounded,
# since a label fed straight from request.url.path is an unbounded cardinality
# hole the moment a client probes a random URL.
INSTRUMENTED_ENDPOINTS = frozenset({"/api/chat", "/api/generate", "/api/embed"})
OTHER_ENDPOINT = "other"

# Outcome values for the `status` label on ollama_requests_total.
STATUS_SUCCESS = "success"
STATUS_CLIENT_ERROR = "client_error"
STATUS_SERVER_ERROR = "server_error"
STATUS_ABORTED = "aborted"
STATUS_UPSTREAM_ERROR = "upstream_error"


app = FastAPI()

# ---------------------------------------------------------------------------
# Bucket layouts
# ---------------------------------------------------------------------------

# Ollama durations span two very different regimes: a cached small-model reply
# lands in tens of milliseconds, while a cold load of a 23 GB model takes tens
# of seconds and a long generation runs for minutes. The prometheus_client
# defaults stop at 10s, which turns every cold load into a +Inf observation and
# makes the p95 meaningless. These buckets keep sub-second resolution and still
# reach ten minutes.
DURATION_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 7.5, 10, 15, 20, 30, 45, 60, 90, 120, 180, 300, 600,
)

# Time to first token is a user-facing latency: anything past a minute is
# already a failed interaction, so the range stops there.
TTFT_BUCKETS = (
    0.05, 0.1, 0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 7.5, 10, 15, 20, 30, 45, 60,
)

# Throughput on a dual-RTX-3090 host comfortably exceeds the old 100 tok/s top
# bucket, which saturated and reported a p95 of exactly 100. Resolution is kept
# dense across 10-200 tok/s where the real traffic sits, then coarsens.
TOKENS_PER_SECOND_BUCKETS = (
    5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 140, 160, 180, 200,
    250, 300, 400, 500, 750, 1000, 1500,
)

# Powers of two from a trivial prompt up to the 128k context some models
# advertise, so KV-cache pressure shows up as a distribution instead of an
# average derived from a counter.
TOKEN_COUNT_BUCKETS = (
    64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, 131072,
)

# ---------------------------------------------------------------------------
# Request-scoped metrics
# ---------------------------------------------------------------------------

OLLAMA_CHAT_REQUEST_COUNT = Counter(
    "ollama_requests_total",
    "Total generation requests, by outcome",
    ["model", "endpoint", "status"],
)

OLLAMA_REQUESTS_IN_FLIGHT = Gauge(
    "ollama_requests_in_flight",
    "Generation requests currently being served",
    ["model", "endpoint"],
)

OLLAMA_TOTAL_DURATION = Histogram(
    "ollama_response_seconds",
    "Total time spent for the response",
    ["model", "endpoint"],
    buckets=DURATION_BUCKETS,
)
OLLAMA_LOAD_DURATION = Histogram(
    "ollama_load_duration_seconds",
    "Time spent loading the model",
    ["model", "endpoint"],
    buckets=DURATION_BUCKETS,
)
OLLAMA_PROMPT_EVAL_DURATION = Histogram(
    "ollama_prompt_eval_duration_seconds",
    "Time spent evaluating prompt",
    ["model", "endpoint"],
    buckets=DURATION_BUCKETS,
)
OLLAMA_EVAL_DURATION = Histogram(
    "ollama_eval_duration_seconds",
    "Time spent generating the response",
    ["model", "endpoint"],
    buckets=DURATION_BUCKETS,
)

OLLAMA_TIME_TO_FIRST_TOKEN = Histogram(
    "ollama_time_to_first_token_seconds",
    "Delay between accepting a streaming request and emitting its first bytes",
    ["model", "endpoint"],
    buckets=TTFT_BUCKETS,
)

OLLAMA_PROMPT_EVAL_COUNT = Counter(
    "ollama_tokens_processed_total",
    "Number of tokens in the prompt",
    ["model", "endpoint"],
)
OLLAMA_EVAL_COUNT = Counter(
    "ollama_tokens_generated_total",
    "Number of tokens in the response",
    ["model", "endpoint"],
)

OLLAMA_PROMPT_TOKENS = Histogram(
    "ollama_prompt_tokens",
    "Distribution of prompt sizes in tokens",
    ["model", "endpoint"],
    buckets=TOKEN_COUNT_BUCKETS,
)
OLLAMA_GENERATED_TOKENS = Histogram(
    "ollama_generated_tokens",
    "Distribution of response sizes in tokens",
    ["model", "endpoint"],
    buckets=TOKEN_COUNT_BUCKETS,
)

OLLAMA_TOKENS_PER_SECOND = Histogram(
    "ollama_tokens_per_second",
    "Tokens generated per second",
    ["model", "endpoint"],
    buckets=TOKENS_PER_SECOND_BUCKETS,
)


# Headers that describe the upstream connection/framing and must never be
# forwarded verbatim. Ollama answers with `Transfer-Encoding: chunked`; once we
# buffer the body and hand it to Starlette, Starlette adds its own
# `Content-Length`. A response carrying both is illegal (RFC 9112 s6.1) and
# strict clients reject it outright - aiohttp (Open WebUI) fails the request
# with "Content-Length can't be present with Transfer-Encoding" while lenient
# ones like curl let it slide.
HOP_BY_HOP_HEADERS = frozenset({
    "connection",
    "content-encoding",
    "content-length",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
})


def make_http_client(timeout=PROXY_TIMEOUT):
    """Build the httpx client used to reach the upstream Ollama server.

    Every upstream call goes through this single factory so that tests can
    monkeypatch it and hand back a client wired to an
    :class:`httpx.MockTransport`, without the production code carrying any
    test-only plumbing.

    Parameters
    ----------
    timeout : httpx.Timeout, optional
        Timeout policy for the client. Defaults to the generous proxy timeout
        suited to long generations.

    Returns
    -------
    httpx.AsyncClient
        An unopened client; callers are expected to use it as a context
        manager.
    """
    return httpx.AsyncClient(timeout=timeout)


def sanitize_response_headers(headers):
    """Strip hop-by-hop headers from an upstream response before forwarding.

    Parameters
    ----------
    headers : Mapping[str, str]
        Headers as returned by the upstream Ollama response.

    Returns
    -------
    dict of str to str
        The same headers minus every entry in :data:`HOP_BY_HOP_HEADERS`, safe
        to hand back to Starlette which recomputes the framing itself.
    """
    return {k: v for k, v in headers.items() if k.lower() not in HOP_BY_HOP_HEADERS}


def normalise_endpoint(path):
    """Collapse a request path to a bounded `endpoint` label value.

    Parameters
    ----------
    path : str
        Raw path from the incoming request, e.g. ``request.url.path``.

    Returns
    -------
    str
        The path itself when it is one of :data:`INSTRUMENTED_ENDPOINTS`,
        otherwise :data:`OTHER_ENDPOINT`.
    """
    normalised = path.rstrip("/") or "/"
    return normalised if normalised in INSTRUMENTED_ENDPOINTS else OTHER_ENDPOINT


def classify_http_status(status_code):
    """Map an upstream HTTP status code to a `status` label value.

    Parameters
    ----------
    status_code : int
        Status code returned by Ollama.

    Returns
    -------
    str
        One of :data:`STATUS_SUCCESS`, :data:`STATUS_CLIENT_ERROR` or
        :data:`STATUS_SERVER_ERROR`. Anything below 400, including the 3xx
        range Ollama never emits, counts as a success.
    """
    if status_code >= 500:
        return STATUS_SERVER_ERROR
    if status_code >= 400:
        return STATUS_CLIENT_ERROR
    return STATUS_SUCCESS


def extract_and_record_metrics(response_data, model, endpoint):
    """Record the metrics carried by an Ollama stats object.

    Parameters
    ----------
    response_data : dict
        Final (``"done": true``) chunk of a stream, or the whole body of a
        non-streaming response. Non-dict values are ignored so a malformed
        upstream payload cannot take the request down.
    model : str
        Model name to label the samples with.
    endpoint : str
        Normalised endpoint the request was served on.
    """
    if not isinstance(response_data, dict):
        return

    # https://github.com/ollama/ollama/blob/main/docs/api.md#response
    total_duration = response_data.get("total_duration", 0) # total time spent in nanoseconds generating the response
    load_duration = response_data.get("load_duration", 0) # time spent in nanoseconds loading the model
    prompt_eval_duration = response_data.get("prompt_eval_duration", 0) # time spent in nanoseconds evaluating the prompt
    prompt_eval_count = response_data.get("prompt_eval_count", 0) # number of tokens in the prompt
    eval_duration = response_data.get("eval_duration", 0) # time spent in nanoseconds generating the response
    eval_count = response_data.get("eval_count", 0) # number of tokens in the response

    if total_duration > 0:
        total_duration_seconds = total_duration / 1_000_000_000
        OLLAMA_TOTAL_DURATION.labels(model=model, endpoint=endpoint).observe(total_duration_seconds)
        logger.debug(f"Model: {model}, Total Duration: {total_duration_seconds:.2f} seconds")
    if load_duration > 0:
        load_duration_seconds = load_duration / 1_000_000_000
        OLLAMA_LOAD_DURATION.labels(model=model, endpoint=endpoint).observe(load_duration_seconds)
        logger.debug(f"Model: {model}, Load Duration: {load_duration_seconds:.2f} seconds")
    if prompt_eval_duration > 0:
        prompt_eval_time_seconds = prompt_eval_duration / 1_000_000_000
        OLLAMA_PROMPT_EVAL_DURATION.labels(model=model, endpoint=endpoint).observe(prompt_eval_time_seconds)
        logger.debug(f"Model: {model}, Prompt Eval Duration: {prompt_eval_time_seconds:.2f} seconds")
    if prompt_eval_count > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model, endpoint=endpoint).inc(prompt_eval_count)
        OLLAMA_PROMPT_TOKENS.labels(model=model, endpoint=endpoint).observe(prompt_eval_count)
        logger.debug(f"Model: {model}, Prompt Eval Count: {prompt_eval_count}")
    if eval_duration > 0:
        eval_duration_seconds = eval_duration / 1_000_000_000
        OLLAMA_EVAL_DURATION.labels(model=model, endpoint=endpoint).observe(eval_duration_seconds)
        logger.debug(f"Model: {model}, Eval Duration: {eval_duration_seconds:.2f} seconds")
    if eval_count > 0:
        OLLAMA_EVAL_COUNT.labels(model=model, endpoint=endpoint).inc(eval_count)
        OLLAMA_GENERATED_TOKENS.labels(model=model, endpoint=endpoint).observe(eval_count)
        logger.debug(f"Model: {model}, Eval Count: {eval_count}")
    if eval_duration > 0 and eval_count > 0:
        tps = eval_count / eval_duration * 1_000_000_000
        OLLAMA_TOKENS_PER_SECOND.labels(model=model, endpoint=endpoint).observe(tps)
        logger.debug(f"Model: {model}, Tokens per Second: {tps:.2f}")


def record_request_outcome(model, endpoint, status):
    """Count a finished request under its outcome.

    The counter is incremented once, on completion, rather than on arrival:
    the outcome is only known at the end, and requests still running are
    already visible through ``ollama_requests_in_flight``.

    Parameters
    ----------
    model : str
        Model name to label the sample with.
    endpoint : str
        Normalised endpoint the request was served on.
    status : str
        One of the ``STATUS_*`` constants.
    """
    OLLAMA_CHAT_REQUEST_COUNT.labels(
        model=model, endpoint=endpoint, status=status
    ).inc()


def find_final_chunk(raw_chunk):
    """Return the terminal stats object contained in an ndjson chunk.

    Ollama streams one JSON object per line and only the last one, flagged
    ``"done": true``, carries the timing and token counts. A single network
    read may hold several lines, or a fragment of one, so every line is tried
    and undecodable ones are skipped rather than raised.

    Parameters
    ----------
    raw_chunk : bytes
        Bytes as read from the upstream stream.

    Returns
    -------
    dict or None
        The stats object when this chunk contained the final line, otherwise
        ``None``.
    """
    try:
        chunk_text = raw_chunk.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character straddling two reads: the line it belongs to
        # is not the terminal one anyway, so dropping this chunk is safe.
        return None

    final_chunk_data = None
    for line in chunk_text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            chunk_json = json.loads(line)
        except json.JSONDecodeError:
            continue
        if chunk_json.get("done", False):
            final_chunk_data = chunk_json
    return final_chunk_data


async def stream_and_record(endpoint, model, headers, body, params, started):
    """Proxy a streaming generation, yielding bytes as they arrive.

    The generator owns the whole lifecycle of its request: it raises the
    in-flight gauge on first iteration and, in a ``finally`` that also runs on
    ``GeneratorExit``, lowers it again and counts the outcome. That is what
    makes a client hanging up mid-stream show as ``status="aborted"`` instead
    of silently vanishing.

    Parameters
    ----------
    endpoint : str
        Normalised endpoint, also used to build the upstream URL.
    model : str
        Model name requested by the client.
    headers : dict of str to str
        Headers to forward upstream, already stripped of the ones we rewrite.
    body : dict
        JSON request body to forward.
    params : Mapping
        Query parameters to forward.
    started : float
        ``time.perf_counter()`` reading taken when the request entered the
        handler, used as the origin for time to first token.

    Yields
    ------
    bytes
        Upstream response chunks, forwarded untouched and in order.
    """
    in_flight = OLLAMA_REQUESTS_IN_FLIGHT.labels(model=model, endpoint=endpoint)
    in_flight.inc()

    status = STATUS_SUCCESS
    first_chunk_seen = False
    final_chunk_data = None

    try:
        async with make_http_client() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}{endpoint}",
                headers=headers,
                json=body,
                params=params,
            ) as response:
                # An error status still arrives as a stream; classify it here
                # so the body (an error message rather than ndjson) is still
                # forwarded to the client untouched.
                status = classify_http_status(response.status_code)

                async for chunk in response.aiter_bytes():
                    if chunk and not first_chunk_seen:
                        # Measured before the yield: what we want is how long
                        # Ollama took, not how long the client took to read.
                        OLLAMA_TIME_TO_FIRST_TOKEN.labels(
                            model=model, endpoint=endpoint
                        ).observe(time.perf_counter() - started)
                        first_chunk_seen = True

                    yield chunk

                    if chunk:
                        final_chunk_data = find_final_chunk(chunk) or final_chunk_data

        if final_chunk_data:
            extract_and_record_metrics(final_chunk_data, model, endpoint)
    except (asyncio.CancelledError, GeneratorExit):
        # Starlette cancels the task, then closes the generator, when the
        # client goes away. Both must be re-raised: swallowing cancellation
        # leaves the server task in an inconsistent state.
        status = STATUS_ABORTED
        logger.debug(f"Client aborted streaming request for model {model}")
        raise
    except httpx.HTTPError as exc:
        status = STATUS_UPSTREAM_ERROR
        logger.warning(f"Upstream error streaming from Ollama for model {model}: {exc}")
        raise
    finally:
        in_flight.dec()
        record_request_outcome(model, endpoint, status)


async def proxy_and_record(endpoint, model, headers, body, params):
    """Proxy a non-streaming generation and record its metrics.

    Parameters
    ----------
    endpoint : str
        Normalised endpoint, also used to build the upstream URL.
    model : str
        Model name requested by the client.
    headers : dict of str to str
        Headers to forward upstream.
    body : dict
        JSON request body to forward.
    params : Mapping
        Query parameters to forward.

    Returns
    -------
    fastapi.Response
        The upstream response, with hop-by-hop headers removed.

    Raises
    ------
    httpx.HTTPError
        Propagated after the failure has been counted as
        ``status="upstream_error"``.
    """
    in_flight = OLLAMA_REQUESTS_IN_FLIGHT.labels(model=model, endpoint=endpoint)
    in_flight.inc()

    status = STATUS_SUCCESS
    try:
        async with make_http_client() as client:
            response = await client.post(
                f"{OLLAMA_HOST}{endpoint}",
                headers=headers,
                json=body,
                params=params,
            )

        status = classify_http_status(response.status_code)

        if response.status_code == 200:
            try:
                extract_and_record_metrics(response.json(), model, endpoint)
            except (json.JSONDecodeError, TypeError, ValueError):
                # A 200 that is not JSON is odd but not fatal: forward it and
                # let the client decide what to make of it.
                pass

        return Response(
            content=response.content,
            status_code=response.status_code,
            headers=sanitize_response_headers(response.headers),
        )
    except asyncio.CancelledError:
        status = STATUS_ABORTED
        logger.debug(f"Client aborted request for model {model}")
        raise
    except httpx.HTTPError as exc:
        status = STATUS_UPSTREAM_ERROR
        logger.warning(f"Upstream error talking to Ollama for model {model}: {exc}")
        raise
    finally:
        in_flight.dec()
        record_request_outcome(model, endpoint, status)


@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/api/chat")
@app.post("/api/generate")
@app.post("/api/embed")
async def chat_with_metrics(request: Request):
    """Handle generation requests with streaming support and metrics extraction.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request on ``/api/chat``, ``/api/generate`` or ``/api/embed``.

    Returns
    -------
    fastapi.Response or fastapi.responses.StreamingResponse
        A streaming response when the client asked for one, the buffered
        upstream response otherwise. ``/api/embed`` never streams.
    """
    # Taken first thing so time to first token covers everything the client
    # waits through, body parsing and connection setup included.
    started = time.perf_counter()

    body = await request.json()
    model = body.get("model", "unknown")
    endpoint = normalise_endpoint(request.url.path)
    is_streaming = body.get("stream", False)

    headers = dict(request.headers)
    headers.pop("host", None)
    # Dropped because httpx recomputes them for the re-serialised JSON body.
    headers.pop("content-length", None)
    headers.pop("content-type", None)

    if is_streaming:
        return StreamingResponse(
            stream_and_record(
                endpoint, model, headers, body, request.query_params, started
            ),
            media_type="application/json",
        )

    return await proxy_and_record(endpoint, model, headers, body, request.query_params)


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def simple_proxy(request: Request, path: str):
    """Simple pass-through proxy for all other endpoints."""
    logger.debug(f"Proxying {request.method} request to /{path}")
    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    async with make_http_client() as client:
        response = await client.request(method=request.method, url=f"{OLLAMA_HOST}/{path}", headers=headers, content=await request.body(), params=request.query_params)

    logger.debug(f"Proxy response: {response.status_code} for {request.method} /{path}")
    return Response(content=response.content, status_code=response.status_code, headers=sanitize_response_headers(response.headers))


async def verify_ollama_connection():
    """Verify connection to Ollama server at startup."""
    logger.debug(f"Verifying connection to Ollama server at {OLLAMA_HOST}")

    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(10.0)) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/version")
            if response.status_code == 200:
                version_data = response.json()
                logger.info(f"Connected to Ollama")
            else:
                logger.error(f"Failed to connect to Ollama server. Status code: {response.status_code}")
    except Exception as e:
        logger.error(f"Failed to connect to Ollama server at {OLLAMA_HOST}: {e}")
        logger.error("Please ensure Ollama is running and accessible at the configured host")

def parse_args(argv=None):
    """Parse command-line arguments.

    Environment variables provide the defaults so that container deployments
    keep working without flags, while explicit CLI arguments always win.

    Parameters
    ----------
    argv : list of str, optional
        Argument vector to parse. Defaults to ``sys.argv[1:]`` when ``None``.

    Returns
    -------
    argparse.Namespace
        Parsed arguments with ``host``, ``port``, ``ollama_host`` and
        ``log_level`` attributes.
    """
    parser = argparse.ArgumentParser(
        description="Prometheus exporter and metrics-extracting proxy for Ollama."
    )
    parser.add_argument(
        "--host",
        default=os.getenv("EXPORTER_HOST", DEFAULT_LISTEN_HOST),
        help="Address to bind the exporter to "
             "(default: %(default)s, dual-stack IPv6/IPv4).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("EXPORTER_PORT", DEFAULT_LISTEN_PORT)),
        help="Port to listen on (default: %(default)s).",
    )
    parser.add_argument(
        "--ollama-host",
        default=os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST),
        help="Base URL of the upstream Ollama server (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging verbosity (default: %(default)s).",
    )
    return parser.parse_args(argv)


def bind_listen_socket(host, port):
    """Bind the listening socket when the requested host needs special care.

    asyncio sets ``IPV6_V6ONLY`` on every AF_INET6 socket it opens, so letting
    uvicorn bind ``::`` on its own produces an IPv6-only listener: IPv4 clients,
    including anything reaching the exporter over ``127.0.0.1``, get a
    connection refused despite the advertised dual-stack default. Binding here
    with the option cleared gives a single socket serving both families.

    Parameters
    ----------
    host : str
        Address the exporter was asked to bind. Only the IPv6 wildcard is
        handled here.
    port : int
        TCP port to listen on.

    Returns
    -------
    socket.socket or None
        A bound socket to hand over to uvicorn, or ``None`` when ``host`` is
        not the IPv6 wildcard and uvicorn can bind it itself.
    """
    if host not in IPV6_WILDCARDS:
        return None

    sock = None
    try:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        # Explicit 0 also overrides a system-wide net.ipv6.bindv6only=1.
        sock.setsockopt(socket.IPPROTO_IPV6, socket.IPV6_V6ONLY, 0)
        sock.bind(("::", port))
    except OSError as exc:
        # No usable IPv6 stack (kernel with ipv6.disable=1, IPv4-only sandbox):
        # serve IPv4 rather than refusing to start at all.
        if sock is not None:
            sock.close()
        logger.warning(
            f"Dual-stack bind on [::]:{port} failed ({exc}), "
            f"falling back to {IPV4_WILDCARD}:{port}"
        )
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind((IPV4_WILDCARD, port))

    # uvicorn passes the socket to loop.create_server(), which calls listen().
    return sock


async def main():
    """Configure runtime from CLI/env, then start the exporter server."""
    global OLLAMA_HOST
    args = parse_args()

    # CLI/env arguments override the module-level defaults set at import time.
    OLLAMA_HOST = args.ollama_host
    logger.setLevel(getattr(logging, args.log_level, logging.INFO))

    await verify_ollama_connection()
    config = uvicorn.Config(
        app, host=args.host, port=args.port, log_level=args.log_level.lower()
    )
    server = uvicorn.Server(config)

    sock = bind_listen_socket(args.host, args.port)
    if sock is None:
        await server.serve()
    else:
        # serve(sockets=...) skips uvicorn's own bind, so our socket options
        # survive.
        families = "IPv6+IPv4" if sock.family == socket.AF_INET6 else "IPv4"
        logger.info(f"Listening on {sock.getsockname()[0]}:{args.port} ({families})")
        await server.serve(sockets=[sock])

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # SIGINT (Ctrl-C, or `docker stop` via STOPSIGNAL) is a normal shutdown:
        # uvicorn already closed the server, so exit quietly instead of letting
        # the traceback surface and the process report a failure status.
        logger.info("Shutting down")
