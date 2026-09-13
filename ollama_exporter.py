"""Prometheus exporter and metrics-extracting reverse proxy for Ollama.

The exporter sits in front of an Ollama server: every request is forwarded
verbatim, while the generation endpoints additionally have their statistics
mined for metrics. Two schemas are instrumented: Ollama's native endpoints
(``/api/chat``, ``/api/generate``, ``/api/embed``), whose ndjson replies carry
a nanosecond stats object, and Ollama's OpenAI-compatible endpoints
(``/v1/chat/completions``, ``/v1/completions``, ``/v1/embeddings``), whose
JSON or SSE replies carry a ``usage`` object instead. A background task polls
``/api/ps`` so the resident-model set, its VRAM footprint and its
``keep_alive`` countdown are observable too.
"""

import os
import argparse
import asyncio
import httpx
import json
import logging
import re
import socket
import time
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timezone

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

# Reported through ollama_exporter_build_info so a dashboard can tell which
# build answered the scrape. CI may stamp a git describe / tag through the env.
__version__ = os.getenv("EXPORTER_VERSION", "2.0.0")

# Default values, overridable via environment variables or CLI arguments.
# CLI arguments take precedence over environment variables (see parse_args).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LISTEN_HOST = "::"  # dual-stack: binds both IPv6 and IPv4
DEFAULT_LISTEN_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
DEFAULT_PS_INTERVAL_SECONDS = 15.0

# Fallback when the host has no usable IPv6 stack (see bind_listen_socket).
IPV4_WILDCARD = "0.0.0.0"

# Spellings of the IPv6 wildcard we bind ourselves rather than leaving to
# uvicorn; any other address is unambiguous and asyncio handles it correctly.
IPV6_WILDCARDS = frozenset({"::", "[::]", "::0"})

# Generation can legitimately run for many minutes on a large model, so the
# proxy timeout is deliberately generous. The residency poll is a cheap local
# status call and gets a short one instead, so a wedged upstream shows up as
# ollama_upstream_up=0 within one scrape interval rather than hanging the task.
PROXY_TIMEOUT = httpx.Timeout(900.0, read=900.0)
PS_TIMEOUT = httpx.Timeout(5.0)

# Configurable Ollama host. Populated by parse_args(); the env default keeps
# backward compatibility for code paths that import this module directly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

# Seconds between two /api/ps polls. Zero or negative disables the poller,
# which is what tests and pure-proxy deployments want.
PS_INTERVAL_SECONDS = float(
    os.getenv("OLLAMA_PS_INTERVAL_SECONDS", DEFAULT_PS_INTERVAL_SECONDS)
)

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

# Paths this exporter instruments. Anything else reaching normalise_endpoint()
# collapses to a single "other" value: the endpoint label must stay bounded,
# since a label fed straight from request.url.path is an unbounded cardinality
# hole the moment a client probes a random URL.
NATIVE_ENDPOINTS = frozenset({"/api/chat", "/api/generate", "/api/embed"})

# Ollama's OpenAI-compatible surface. Clients that speak the OpenAI protocol
# (and that is most of them: Open WebUI, LangChain, the openai SDK itself) land
# here rather than on the native endpoints, so leaving these in the "other"
# catch-all made the busiest traffic path invisible: a box serving multi-minute
# generations reported no requests at all for the model doing the work.
OPENAI_ENDPOINTS = frozenset({
    "/v1/chat/completions",
    "/v1/completions",
    "/v1/embeddings",
})

INSTRUMENTED_ENDPOINTS = NATIVE_ENDPOINTS | OPENAI_ENDPOINTS
OTHER_ENDPOINT = "other"

# Server-sent events framing used by the OpenAI-compatible streaming replies:
# one `data: <json>` line per event, the stream closed by `data: [DONE]`.
SSE_DATA_PREFIX = "data:"
SSE_DONE_PAYLOAD = "[DONE]"

# Outcome values for the `status` label on ollama_requests_total.
STATUS_SUCCESS = "success"
STATUS_CLIENT_ERROR = "client_error"
STATUS_SERVER_ERROR = "server_error"
STATUS_ABORTED = "aborted"
STATUS_UPSTREAM_ERROR = "upstream_error"


@asynccontextmanager
async def lifespan(app):
    """Run the residency poller for the lifetime of the application.

    Parameters
    ----------
    app : fastapi.FastAPI
        The application being started. Unused, but required by the Starlette
        lifespan protocol.

    Yields
    ------
    None
        Control is handed back to the server while the poller runs in the
        background.
    """
    task = None
    if PS_INTERVAL_SECONDS > 0:
        task = asyncio.create_task(poll_model_residency(PS_INTERVAL_SECONDS))
    else:
        logger.info("Residency poller disabled (OLLAMA_PS_INTERVAL_SECONDS <= 0)")

    try:
        yield
    finally:
        if task is not None:
            task.cancel()
            # Awaiting the cancelled task keeps shutdown quiet: without it
            # asyncio logs "Task was destroyed but it is pending".
            with suppress(asyncio.CancelledError):
                await task


app = FastAPI(lifespan=lifespan)

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

OLLAMA_USAGE_MISSING = Counter(
    "ollama_usage_missing_total",
    "Streaming OpenAI-compatible replies that ended without a usage block",
    ["model", "endpoint"],
)

# ---------------------------------------------------------------------------
# Residency metrics, fed by the /api/ps poller
# ---------------------------------------------------------------------------

OLLAMA_MODEL_LOADED = Gauge(
    "ollama_model_loaded",
    "1 while the model occupies VRAM; the series disappears once it is evicted",
    ["model"],
)
OLLAMA_MODEL_VRAM_BYTES = Gauge(
    "ollama_model_vram_bytes",
    "Bytes of VRAM the resident model occupies",
    ["model"],
)
OLLAMA_MODEL_SIZE_BYTES = Gauge(
    "ollama_model_size_bytes",
    "Total size in bytes of the resident model, VRAM and host memory combined",
    ["model"],
)
OLLAMA_MODEL_CONTEXT_LENGTH = Gauge(
    "ollama_model_context_length",
    "Context window the resident model was loaded with, in tokens",
    ["model"],
)
OLLAMA_MODEL_EXPIRES_SECONDS = Gauge(
    "ollama_model_expires_seconds",
    "Seconds left on the keep_alive countdown before the model is evicted",
    ["model"],
)
OLLAMA_MODELS_LOADED = Gauge(
    "ollama_models_loaded",
    "Number of models currently resident",
)
OLLAMA_MODEL_SWAPS = Counter(
    "ollama_model_swaps_total",
    "Times a model entered the resident set, i.e. was loaded from cold",
    ["model"],
)
OLLAMA_MODEL_INFO = Gauge(
    "ollama_model_info",
    "Static description of a resident model; always 1, meant as a join target",
    ["model", "family", "parameter_size", "quantization_level"],
)
OLLAMA_UPSTREAM_UP = Gauge(
    "ollama_upstream_up",
    "1 when the last /api/ps poll reached the Ollama server",
)
OLLAMA_BUILD_INFO = Gauge(
    "ollama_exporter_build_info",
    "Always 1; the version label carries the running exporter build",
    ["version"],
)
OLLAMA_BUILD_INFO.labels(version=__version__).set(1)

# Resident models observed during the previous poll, mapped to the label values
# their ollama_model_info series was published with. The label values are kept
# because removing a child series requires replaying them exactly.
_RESIDENT_MODELS = {}


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


def record_openai_usage(usage, model, endpoint, generation_seconds=None):
    """Record the token counts carried by an OpenAI-compatible ``usage`` object.

    Parameters
    ----------
    usage : dict or None
        The ``usage`` object from a ``/v1/*`` reply, i.e. ``prompt_tokens``,
        ``completion_tokens`` and ``total_tokens``. Anything that is not a
        dict is ignored, so a malformed upstream payload cannot take the
        request down.
    model : str
        Model name to label the samples with.
    endpoint : str
        Normalised endpoint the request was served on.
    generation_seconds : float, optional
        Wall-clock seconds between the first streamed token and the end of the
        stream, measured by the exporter. Tokens per second is only observed
        when this is known and positive.

    Notes
    -----
    The three duration histograms (``ollama_load_duration_seconds``,
    ``ollama_prompt_eval_duration_seconds``, ``ollama_eval_duration_seconds``)
    are deliberately left unobserved on this path. The OpenAI-compatible reply
    carries no ``load_duration``, ``prompt_eval_duration`` or ``eval_duration``,
    and there is nothing to derive them from. Observing a zero instead would be
    worse than observing nothing: it drags every quantile, average and
    ``rate(_sum)/rate(_count)`` computed downstream towards zero and quietly
    corrupts the same series the native endpoints populate correctly. An absent
    observation reads as "not measured here", which is the truth.
    """
    if not isinstance(usage, dict):
        return

    # `or 0` rather than a default: Ollama sends an explicit null for the
    # fields it has nothing to report, and null does not compare with 0.
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0

    if prompt_tokens > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model, endpoint=endpoint).inc(prompt_tokens)
        OLLAMA_PROMPT_TOKENS.labels(model=model, endpoint=endpoint).observe(prompt_tokens)
        logger.debug(f"Model: {model}, Prompt Tokens: {prompt_tokens}")

    # Embeddings report prompt tokens and nothing else, so they fall out of
    # this branch on their own: no generated tokens, and no throughput sample
    # for a call that generates nothing.
    if completion_tokens > 0:
        OLLAMA_EVAL_COUNT.labels(model=model, endpoint=endpoint).inc(completion_tokens)
        OLLAMA_GENERATED_TOKENS.labels(model=model, endpoint=endpoint).observe(completion_tokens)
        logger.debug(f"Model: {model}, Completion Tokens: {completion_tokens}")

    if completion_tokens > 0 and generation_seconds and generation_seconds > 0:
        tps = completion_tokens / generation_seconds
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


def find_usage_block(raw_chunk):
    """Return the ``usage`` object contained in a server-sent-events chunk.

    The OpenAI-compatible stream is SSE, not the ndjson :func:`find_final_chunk`
    handles: events are ``data: <json>`` lines and the stream ends with
    ``data: [DONE]``. When the client asked for usage accounting, the last event
    before ``[DONE]`` carries the ``usage`` object and an empty ``choices``
    array. Every ``data:`` line is tried rather than only the last one, since a
    single network read can hold several events or a fragment of one, and
    undecodable lines are skipped rather than raised.

    Parameters
    ----------
    raw_chunk : bytes
        Bytes as read from the upstream stream.

    Returns
    -------
    dict or None
        The usage object when this chunk contained the event carrying it,
        otherwise ``None``.
    """
    try:
        chunk_text = raw_chunk.decode("utf-8")
    except UnicodeDecodeError:
        # A multi-byte character straddling two reads. The usage event is pure
        # ASCII, so the line it belongs to is never the one being truncated.
        return None

    usage = None
    for line in chunk_text.splitlines():
        line = line.strip()
        if not line.startswith(SSE_DATA_PREFIX):
            continue
        payload = line[len(SSE_DATA_PREFIX):].strip()
        if not payload or payload == SSE_DONE_PAYLOAD:
            continue
        try:
            event = json.loads(payload)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and isinstance(event.get("usage"), dict):
            usage = event["usage"]
    return usage


def chunk_carries_sse_data(raw_chunk):
    """Tell whether a raw chunk contains at least one SSE ``data:`` line.

    Used to time the first token: the upstream may open the stream with SSE
    comments or blank lines, and an error reply is not SSE at all. Neither
    should be mistaken for the model having produced something.

    Parameters
    ----------
    raw_chunk : bytes
        Bytes as read from the upstream stream.

    Returns
    -------
    bool
        ``True`` when a ``data:`` line is present in this chunk.
    """
    try:
        chunk_text = raw_chunk.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return any(line.strip().startswith(SSE_DATA_PREFIX) for line in chunk_text.splitlines())


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


async def stream_openai_and_record(endpoint, model, headers, body, params, started):
    """Proxy a streaming OpenAI-compatible generation, yielding bytes as they arrive.

    Mirrors :func:`stream_and_record` on the ``/v1/*`` surface: same in-flight
    accounting, same ``finally`` that turns a client hang-up into
    ``status="aborted"``, same unbuffered pass-through. What differs is the
    payload, which is SSE carrying a ``usage`` object rather than ndjson
    carrying a nanosecond stats object, and therefore what can honestly be
    recorded from it.

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
        handler, used as the origin for time to first token and for the
        exporter-measured response time.

    Yields
    ------
    bytes
        Upstream response chunks, forwarded untouched and in order.
    """
    in_flight = OLLAMA_REQUESTS_IN_FLIGHT.labels(model=model, endpoint=endpoint)
    in_flight.inc()

    status = STATUS_SUCCESS
    first_token_time = None
    usage = None

    try:
        async with make_http_client() as client:
            async with client.stream(
                "POST",
                f"{OLLAMA_HOST}{endpoint}",
                headers=headers,
                json=body,
                params=params,
            ) as response:
                # An error status still arrives as a stream; classify it here so
                # the body (an error message rather than SSE) is still forwarded
                # to the client untouched.
                status = classify_http_status(response.status_code)

                async for chunk in response.aiter_bytes():
                    if first_token_time is None and chunk and chunk_carries_sse_data(chunk):
                        # Measured before the yield: what we want is how long
                        # Ollama took, not how long the client took to read.
                        first_token_time = time.perf_counter()
                        OLLAMA_TIME_TO_FIRST_TOKEN.labels(
                            model=model, endpoint=endpoint
                        ).observe(first_token_time - started)

                    yield chunk

                    if chunk:
                        usage = find_usage_block(chunk) or usage

        ended = time.perf_counter()
        OLLAMA_TOTAL_DURATION.labels(model=model, endpoint=endpoint).observe(ended - started)

        if usage is not None:
            # Throughput is measured from the first token rather than from the
            # start of the request: the prompt evaluation and a cold model load
            # happen before it, and folding them in would understate generation
            # speed by an order of magnitude on a large model.
            generation_seconds = None if first_token_time is None else ended - first_token_time
            record_openai_usage(usage, model, endpoint, generation_seconds)
        elif status == STATUS_SUCCESS:
            # No usage block on a stream that completed normally: the client
            # did not send `stream_options: {"include_usage": true}`, so its
            # tokens are simply not counted anywhere. The exporter will not
            # rewrite the request to force the option, since that would change
            # the event stream the client receives. Counting the gap instead
            # keeps the undercount visible rather than silent.
            OLLAMA_USAGE_MISSING.labels(model=model, endpoint=endpoint).inc()
            logger.debug(f"No usage block in streaming {endpoint} reply for model {model}")
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


async def proxy_openai_and_record(endpoint, model, headers, body, params, started):
    """Proxy a non-streaming OpenAI-compatible request and record its metrics.

    Covers ``/v1/embeddings``, which never streams, as well as the chat and
    completion endpoints when the client did not ask for a stream. No time to
    first token is recorded: there is no first token to time, only a single
    buffered reply. No tokens-per-second either, for the same reason, since the
    generation window cannot be separated from the prompt evaluation and the
    model load that precede it inside one opaque wait.

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
    started : float
        ``time.perf_counter()`` reading taken when the request entered the
        handler, used as the origin for the exporter-measured response time.

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
            OLLAMA_TOTAL_DURATION.labels(model=model, endpoint=endpoint).observe(
                time.perf_counter() - started
            )
            try:
                payload = response.json()
            except (json.JSONDecodeError, TypeError, ValueError):
                # A 200 that is not JSON is odd but not fatal: forward it and
                # let the client decide what to make of it.
                payload = None
            if isinstance(payload, dict):
                record_openai_usage(payload.get("usage"), model, endpoint)

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


@app.post("/v1/chat/completions")
@app.post("/v1/completions")
@app.post("/v1/embeddings")
async def openai_with_metrics(request: Request):
    """Handle OpenAI-compatible requests with streaming support and metrics extraction.

    Parameters
    ----------
    request : fastapi.Request
        Incoming request on ``/v1/chat/completions``, ``/v1/completions`` or
        ``/v1/embeddings``.

    Returns
    -------
    fastapi.Response or fastapi.responses.StreamingResponse
        A server-sent-events response when the client asked for a stream, the
        buffered upstream response otherwise. ``/v1/embeddings`` never streams,
        since its request body carries no ``stream`` field.
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
            stream_openai_and_record(
                endpoint, model, headers, body, request.query_params, started
            ),
            # SSE, not ndjson: an OpenAI client parses the framing, not just
            # the payload, and mislabelling it breaks the stream client-side.
            media_type="text/event-stream",
        )

    return await proxy_openai_and_record(
        endpoint, model, headers, body, request.query_params, started
    )


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


# ---------------------------------------------------------------------------
# Residency poller
# ---------------------------------------------------------------------------

# Go's time formatting emits up to nanosecond precision, while
# datetime.fromisoformat tops out at microseconds. Trim the surplus digits
# instead of failing the whole parse over sub-microsecond noise.
_SURPLUS_FRACTIONAL_DIGITS = re.compile(r"(\.\d{6})\d+")


def parse_expires_at(raw):
    """Parse the ``expires_at`` timestamp returned by ``/api/ps``.

    Parameters
    ----------
    raw : str or None
        RFC 3339 timestamp as emitted by Ollama, e.g.
        ``"2026-09-13T14:38:31.83753294-07:00"``.

    Returns
    -------
    datetime.datetime or None
        A timezone-aware datetime, or ``None`` when the field is missing or
        unparseable.
    """
    if not raw:
        return None

    text = _SURPLUS_FRACTIONAL_DIGITS.sub(r"\1", str(raw).strip())
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        logger.debug(f"Unparseable expires_at from /api/ps: {raw!r}")
        return None

    # Ollama always sends an offset, but a naive value would otherwise blow up
    # the subtraction against an aware "now".
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _remove_child(metric, *label_values):
    """Drop one labelled child series, tolerating an already-absent child.

    Parameters
    ----------
    metric : prometheus_client.metrics.MetricWrapperBase
        The labelled metric family to prune.
    *label_values : str
        Label values identifying the child, in declaration order.
    """
    try:
        metric.remove(*label_values)
    except KeyError:
        pass


def clear_model_residency(model, info_labels):
    """Remove every residency series belonging to an evicted model.

    Gauges are removed rather than zeroed: a stale ``ollama_model_loaded=1``
    for a model that left VRAM half an hour ago is worse than no sample at
    all, and an absent series makes ``keep_alive`` eviction visible as a gap.

    Parameters
    ----------
    model : str
        Name of the model that left the resident set.
    info_labels : tuple of str
        The ``(family, parameter_size, quantization_level)`` values its
        ``ollama_model_info`` series was published with.
    """
    _remove_child(OLLAMA_MODEL_LOADED, model)
    _remove_child(OLLAMA_MODEL_VRAM_BYTES, model)
    _remove_child(OLLAMA_MODEL_SIZE_BYTES, model)
    _remove_child(OLLAMA_MODEL_CONTEXT_LENGTH, model)
    _remove_child(OLLAMA_MODEL_EXPIRES_SECONDS, model)
    _remove_child(OLLAMA_MODEL_INFO, model, *info_labels)


def update_residency_metrics(models, now=None):
    """Reconcile the residency gauges with one ``/api/ps`` payload.

    Parameters
    ----------
    models : list of dict
        The ``models`` array from ``/api/ps``.
    now : datetime.datetime, optional
        Reference instant for the ``expires_at`` countdown. Defaults to the
        current UTC time; injectable so tests are not time-dependent.

    Notes
    -----
    The first poll after an exporter restart counts every resident model as a
    swap, since nothing was known about the previous state. That is a
    restart artifact, not a thrash signal, and is why dashboards should read
    ``ollama_model_swaps_total`` as a rate rather than an absolute.
    """
    global _RESIDENT_MODELS

    reference = now or datetime.now(timezone.utc)

    current = {}
    for entry in models or []:
        if not isinstance(entry, dict):
            continue
        # /api/ps calls the field "name"; some versions also carry "model".
        name = entry.get("name") or entry.get("model")
        if name:
            current[name] = entry

    for name, info_labels in _RESIDENT_MODELS.items():
        if name not in current:
            clear_model_residency(name, info_labels)

    resident = {}
    for name, entry in current.items():
        if name not in _RESIDENT_MODELS:
            OLLAMA_MODEL_SWAPS.labels(model=name).inc()

        details = entry.get("details") or {}
        info_labels = (
            details.get("family", "unknown"),
            details.get("parameter_size", "unknown"),
            details.get("quantization_level", "unknown"),
        )

        previous_info = _RESIDENT_MODELS.get(name)
        if previous_info is not None and previous_info != info_labels:
            # Same name, different details. Re-pulling a tag can change the
            # quantization or the parameter size under an unchanged name, and
            # the ollama_model_info child published with the old label values
            # would otherwise linger next to the new one forever.
            _remove_child(OLLAMA_MODEL_INFO, name, *previous_info)

        OLLAMA_MODEL_LOADED.labels(model=name).set(1)
        OLLAMA_MODEL_VRAM_BYTES.labels(model=name).set(entry.get("size_vram", 0))
        OLLAMA_MODEL_SIZE_BYTES.labels(model=name).set(entry.get("size", 0))
        if entry.get("context_length"):
            OLLAMA_MODEL_CONTEXT_LENGTH.labels(model=name).set(entry["context_length"])

        expires_at = parse_expires_at(entry.get("expires_at"))
        if expires_at is not None:
            OLLAMA_MODEL_EXPIRES_SECONDS.labels(model=name).set(
                (expires_at - reference).total_seconds()
            )

        OLLAMA_MODEL_INFO.labels(model=name, family=info_labels[0],
                                 parameter_size=info_labels[1],
                                 quantization_level=info_labels[2]).set(1)

        resident[name] = info_labels

    OLLAMA_MODELS_LOADED.set(len(resident))
    _RESIDENT_MODELS = resident


async def refresh_model_residency():
    """Poll ``/api/ps`` once and reconcile the residency gauges.

    Any failure, network or payload, is contained here: it flips
    ``ollama_upstream_up`` to 0 and returns, leaving the residency gauges at
    their last known values rather than clearing them on a transient blip.
    """
    try:
        async with make_http_client(PS_TIMEOUT) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/ps")
            response.raise_for_status()
            payload = response.json()
    except Exception as exc:
        # Deliberately broad: httpx errors, JSON decoding and any surprise the
        # upstream throws must all degrade to "upstream down", never kill the
        # poller. CancelledError is a BaseException and still propagates, so
        # shutdown keeps working.
        OLLAMA_UPSTREAM_UP.set(0)
        logger.warning(f"Residency poll of {OLLAMA_HOST}/api/ps failed: {exc}")
        return

    OLLAMA_UPSTREAM_UP.set(1)
    update_residency_metrics(payload.get("models") if isinstance(payload, dict) else [])


async def poll_model_residency(interval):
    """Refresh the residency gauges forever, one poll every ``interval``.

    Parameters
    ----------
    interval : float
        Seconds to wait between two polls.
    """
    logger.info(f"Polling {OLLAMA_HOST}/api/ps every {interval}s for model residency")
    while True:
        await refresh_model_residency()
        await asyncio.sleep(interval)


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
        Parsed arguments with ``host``, ``port``, ``ollama_host``,
        ``ps_interval`` and ``log_level`` attributes.
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
        "--ps-interval",
        type=float,
        default=float(os.getenv("OLLAMA_PS_INTERVAL_SECONDS", DEFAULT_PS_INTERVAL_SECONDS)),
        help="Seconds between two /api/ps residency polls; 0 disables the "
             "poller (default: %(default)s).",
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
    global OLLAMA_HOST, PS_INTERVAL_SECONDS
    args = parse_args()

    # CLI/env arguments override the module-level defaults set at import time.
    # The residency poller reads these globals when the lifespan starts, which
    # happens after this point, so assigning them here is enough.
    OLLAMA_HOST = args.ollama_host
    PS_INTERVAL_SECONDS = args.ps_interval
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
