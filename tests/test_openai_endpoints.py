"""Tests for the OpenAI-compatible surface: outcomes, TTFT, usage accounting.

Same approach as ``test_metrics.py``: ``stream_openai_and_record`` and
``proxy_openai_and_record`` are driven directly with ``asyncio.run`` inside
plain sync tests, and the upstream is an ``httpx.MockTransport`` swapped in
through the ``make_http_client`` seam. Response bodies are always async
generators, since a plain sync iterator makes httpx's async request path raise
inside ``MockTransport`` on this version.
"""

import asyncio
import time

import httpx
from prometheus_client import REGISTRY

import ollama_exporter as oe

CHAT_ENDPOINT = "/v1/chat/completions"
EMBEDDINGS_ENDPOINT = "/v1/embeddings"


def metric_value(name, labels=None):
    """Read one Prometheus sample, treating an absent series as zero.

    Parameters
    ----------
    name : str
        Fully qualified sample name, e.g. ``"ollama_usage_missing_total"`` or a
        histogram's ``_bucket``/``_count``/``_sum`` suffix.
    labels : dict of str to str, optional
        Label values identifying the sample. Defaults to no labels.

    Returns
    -------
    float
        The sample's current value, or ``0.0`` when the series does not exist
        yet, so a before/after delta never has to special-case a metric's
        first use.
    """
    value = REGISTRY.get_sample_value(name, labels or {})
    return value if value is not None else 0.0


def _patch_client(monkeypatch, handler):
    """Point ``ollama_exporter.make_http_client`` at a mocked transport.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used so the patch is undone automatically after the test.
    handler : Callable[[httpx.Request], httpx.Response]
        Synchronous handler invoked for every request made through the patched
        client; see :class:`httpx.MockTransport`.
    """

    def factory(timeout=None):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oe, "make_http_client", factory)


async def _drain(agen):
    """Fully consume an async generator, discarding its yielded values.

    Parameters
    ----------
    agen : AsyncGenerator
        Generator to exhaust, e.g. the return value of
        ``stream_openai_and_record``.
    """
    async for _ in agen:
        pass


def _sse(*events):
    """Build an async body yielding the given SSE payloads, then ``[DONE]``.

    Parameters
    ----------
    *events : str
        JSON payloads, one per ``data:`` event, in the order the upstream
        emits them.

    Returns
    -------
    Callable[[], AsyncGenerator]
        A factory producing a fresh async generator over the encoded events.
    """

    async def body():
        for event in events:
            yield f"data: {event}\n\n".encode("utf-8")
        yield b"data: [DONE]\n\n"

    return body


def _run_stream(model, endpoint=CHAT_ENDPOINT):
    """Drive one streaming request through ``stream_openai_and_record``.

    Parameters
    ----------
    model : str
        Model name to label the samples with.
    endpoint : str, optional
        Endpoint being exercised. Defaults to ``/v1/chat/completions``.
    """
    asyncio.run(
        _drain(
            oe.stream_openai_and_record(
                endpoint, model, {}, {"model": model, "stream": True}, {},
                time.perf_counter(),
            )
        )
    )


# ---------------------------------------------------------------------------
# Request outcomes
# ---------------------------------------------------------------------------


def test_successful_chat_completion_counts_as_success(monkeypatch):
    """A 200 on /v1/chat/completions lands under status="success"."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "choices": [{"message": {"content": "hello"}}],
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
        )

    _patch_client(monkeypatch, handler)

    labels = {"model": "ok-model", "endpoint": CHAT_ENDPOINT, "status": "success"}
    before = metric_value("ollama_requests_total", labels)

    asyncio.run(
        oe.proxy_openai_and_record(
            CHAT_ENDPOINT, "ok-model", {}, {"model": "ok-model"}, {}, time.perf_counter()
        )
    )

    assert metric_value("ollama_requests_total", labels) - before == 1


def test_upstream_5xx_counts_as_server_error(monkeypatch):
    """An upstream 500 is forwarded and counted under status="server_error"."""

    def handler(request):
        return httpx.Response(500, json={"error": "model runner crashed"})

    _patch_client(monkeypatch, handler)

    labels = {"model": "broken-model", "endpoint": CHAT_ENDPOINT, "status": "server_error"}
    before = metric_value("ollama_requests_total", labels)

    response = asyncio.run(
        oe.proxy_openai_and_record(
            CHAT_ENDPOINT, "broken-model", {}, {"model": "broken-model"}, {},
            time.perf_counter(),
        )
    )

    assert response.status_code == 500
    assert metric_value("ollama_requests_total", labels) - before == 1


# ---------------------------------------------------------------------------
# Time to first token
# ---------------------------------------------------------------------------


def test_ttft_is_recorded_from_the_first_sse_event(monkeypatch):
    """Draining a streamed reply adds exactly one TTFT observation.

    The stream deliberately opens with an SSE comment: the first token has not
    been produced yet when it arrives, so it must not start the clock.
    """

    async def body():
        yield b": keep-alive\n\n"
        yield b'data: {"id":"1","choices":[{"delta":{"content":"Hi"}}]}\n\n'
        yield b"data: [DONE]\n\n"

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)

    labels = {"model": "ttft-openai", "endpoint": CHAT_ENDPOINT}
    before = metric_value("ollama_time_to_first_token_seconds_count", labels)

    _run_stream("ttft-openai")

    assert metric_value("ollama_time_to_first_token_seconds_count", labels) - before == 1


# ---------------------------------------------------------------------------
# Usage accounting on the streaming path
# ---------------------------------------------------------------------------


def test_streaming_usage_block_feeds_the_token_metrics(monkeypatch):
    """A stream carrying usage records prompt, completion and throughput."""

    def handler(request):
        return httpx.Response(
            200,
            content=_sse(
                '{"id":"1","choices":[{"delta":{"content":"Hi"}}]}',
                '{"id":"1","choices":[],'
                '"usage":{"prompt_tokens":11,"completion_tokens":7,"total_tokens":18}}',
            )(),
        )

    _patch_client(monkeypatch, handler)

    labels = {"model": "usage-model", "endpoint": CHAT_ENDPOINT}
    before_prompt = metric_value("ollama_tokens_processed_total", labels)
    before_generated = metric_value("ollama_tokens_generated_total", labels)
    before_tps = metric_value("ollama_tokens_per_second_count", labels)
    before_missing = metric_value("ollama_usage_missing_total", labels)

    _run_stream("usage-model")

    assert metric_value("ollama_tokens_processed_total", labels) - before_prompt == 11
    assert metric_value("ollama_tokens_generated_total", labels) - before_generated == 7
    assert metric_value("ollama_tokens_per_second_count", labels) - before_tps == 1
    assert metric_value("ollama_usage_missing_total", labels) - before_missing == 0


def test_streaming_without_usage_is_counted_as_missing(monkeypatch):
    """A stream that never sends usage bumps ollama_usage_missing_total.

    This is the common case: a client that does not set
    ``stream_options: {"include_usage": true}`` gets no token accounting at
    all, and the counter is what makes that undercount visible.
    """

    def handler(request):
        return httpx.Response(
            200,
            content=_sse('{"id":"1","choices":[{"delta":{"content":"Hi"}}]}')(),
        )

    _patch_client(monkeypatch, handler)

    labels = {"model": "no-usage-model", "endpoint": CHAT_ENDPOINT}
    before_missing = metric_value("ollama_usage_missing_total", labels)
    before_generated = metric_value("ollama_tokens_generated_total", labels)

    _run_stream("no-usage-model")

    assert metric_value("ollama_usage_missing_total", labels) - before_missing == 1
    assert metric_value("ollama_tokens_generated_total", labels) - before_generated == 0


def test_duration_histograms_stay_unobserved_on_the_openai_path(monkeypatch):
    """The three nanosecond histograms take no sample from a /v1/* reply.

    The OpenAI-compatible schema carries no load, prompt eval or eval duration.
    Observing a zero would be worse than observing nothing, so the series must
    stay empty for this endpoint.
    """

    def handler(request):
        return httpx.Response(
            200,
            content=_sse(
                '{"id":"1","choices":[{"delta":{"content":"Hi"}}]}',
                '{"id":"1","choices":[],'
                '"usage":{"prompt_tokens":3,"completion_tokens":2,"total_tokens":5}}',
            )(),
        )

    _patch_client(monkeypatch, handler)

    labels = {"model": "no-durations-model", "endpoint": CHAT_ENDPOINT}

    _run_stream("no-durations-model")

    assert metric_value("ollama_load_duration_seconds_count", labels) == 0
    assert metric_value("ollama_prompt_eval_duration_seconds_count", labels) == 0
    assert metric_value("ollama_eval_duration_seconds_count", labels) == 0
    # The exporter measures this one itself, so it is the one that must exist.
    assert metric_value("ollama_response_seconds_count", labels) == 1


# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------


def test_embeddings_record_prompt_tokens_but_no_throughput(monkeypatch):
    """An embedding call counts its prompt tokens and nothing about speed."""

    def handler(request):
        return httpx.Response(
            200,
            json={
                "object": "list",
                "data": [{"object": "embedding", "index": 0, "embedding": [0.1, 0.2]}],
                "usage": {"prompt_tokens": 42, "total_tokens": 42},
            },
        )

    _patch_client(monkeypatch, handler)

    labels = {"model": "embed-model", "endpoint": EMBEDDINGS_ENDPOINT}
    before_prompt = metric_value("ollama_tokens_processed_total", labels)

    asyncio.run(
        oe.proxy_openai_and_record(
            EMBEDDINGS_ENDPOINT, "embed-model", {}, {"model": "embed-model", "input": "hi"},
            {}, time.perf_counter(),
        )
    )

    assert metric_value("ollama_tokens_processed_total", labels) - before_prompt == 42
    assert metric_value("ollama_tokens_per_second_count", labels) == 0
    assert metric_value("ollama_generated_tokens_count", labels) == 0
    assert metric_value("ollama_time_to_first_token_seconds_count", labels) == 0
