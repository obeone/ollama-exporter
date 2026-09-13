"""Tests for the request-scoped metrics: outcomes, TTFT, tokens/second, embed.

These exercise ``stream_and_record`` and ``proxy_and_record`` directly rather
than through the FastAPI app, so no ASGI test client or event loop fixture is
needed: coroutines are driven with ``asyncio.run`` inside plain sync tests, and
the upstream is a ``httpx.MockTransport`` swapped in through the
``make_http_client`` seam.
"""

import asyncio
import time

import httpx
import pytest
from prometheus_client import REGISTRY

import ollama_exporter as oe


def metric_value(name, labels=None):
    """Read one Prometheus sample, treating an absent series as zero.

    Parameters
    ----------
    name : str
        Fully qualified sample name, e.g. ``"ollama_requests_total"`` or a
        histogram's ``_bucket``/``_count``/``_sum`` suffix.
    labels : dict of str to str, optional
        Label values identifying the sample. Defaults to no labels.

    Returns
    -------
    float
        The sample's current value, or ``0.0`` when the series does not
        exist yet. Callers computing a before/after delta can therefore
        subtract two readings without special-casing a metric's first use.
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
        Synchronous handler invoked for every request made through the
        patched client; see :class:`httpx.MockTransport`.
    """

    def factory(timeout=None):
        return httpx.AsyncClient(transport=httpx.MockTransport(handler))

    monkeypatch.setattr(oe, "make_http_client", factory)


async def _drain(agen):
    """Fully consume an async generator, discarding its yielded values.

    Parameters
    ----------
    agen : AsyncGenerator
        Generator to exhaust, e.g. the return value of ``stream_and_record``.
    """
    async for _ in agen:
        pass


# ---------------------------------------------------------------------------
# Time to first token
# ---------------------------------------------------------------------------


def test_ttft_is_recorded_once_a_streaming_response_is_exhausted(monkeypatch):
    """Draining a streamed reply adds exactly one TTFT observation.

    A multi-line ndjson body is served over an async generator (a plain sync
    iterator makes httpx's async request path raise inside MockTransport on
    this httpx version, so the mocked upstream body is always an async
    generator in this file).
    """

    async def body():
        yield b'{"model": "ttft-model", "done": false}\n'
        yield (
            b'{"model": "ttft-model", "done": true, '
            b'"eval_count": 3, "eval_duration": 1000000000}\n'
        )

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)

    labels = {"model": "ttft-model", "endpoint": "/api/chat"}
    before = metric_value("ollama_time_to_first_token_seconds_count", labels)

    asyncio.run(
        _drain(
            oe.stream_and_record(
                "/api/chat", "ttft-model", {}, {"model": "ttft-model"}, {}, time.perf_counter()
            )
        )
    )

    after = metric_value("ollama_time_to_first_token_seconds_count", labels)
    assert after - before == 1


# ---------------------------------------------------------------------------
# In-flight gauge returns to zero
# ---------------------------------------------------------------------------


def test_in_flight_gauge_returns_to_zero_after_normal_completion(monkeypatch):
    """A fully drained stream leaves no requests marked in flight."""

    async def body():
        yield b'{"done": true, "eval_count": 1, "eval_duration": 1}\n'

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)
    labels = {"model": "inflight-normal", "endpoint": "/api/chat"}

    asyncio.run(
        _drain(oe.stream_and_record("/api/chat", "inflight-normal", {}, {}, {}, time.perf_counter()))
    )

    assert metric_value("ollama_requests_in_flight", labels) == 0


def test_in_flight_gauge_returns_to_zero_after_upstream_exception_mid_stream(monkeypatch):
    """An upstream error partway through a stream still lowers the gauge."""

    async def body():
        yield b'{"done": false}\n'
        raise httpx.ReadError("connection reset mid-stream")

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)
    labels = {"model": "inflight-upstream-error", "endpoint": "/api/chat"}

    async def drive():
        with pytest.raises(httpx.HTTPError):
            await _drain(
                oe.stream_and_record(
                    "/api/chat", "inflight-upstream-error", {}, {}, {}, time.perf_counter()
                )
            )

    asyncio.run(drive())

    assert metric_value("ollama_requests_in_flight", labels) == 0


def test_in_flight_gauge_returns_to_zero_after_client_abort(monkeypatch):
    """A client hanging up mid-stream (``aclose()``) still lowers the gauge."""

    async def body():
        yield b'{"done": false}\n'
        yield b'{"done": true, "eval_count": 1, "eval_duration": 1}\n'

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)
    labels = {"model": "inflight-abort", "endpoint": "/api/chat"}

    async def drive():
        gen = oe.stream_and_record("/api/chat", "inflight-abort", {}, {}, {}, time.perf_counter())
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(drive())

    assert metric_value("ollama_requests_in_flight", labels) == 0


# ---------------------------------------------------------------------------
# Client abort is counted as aborted
# ---------------------------------------------------------------------------


def test_client_abort_is_counted_as_aborted_outcome(monkeypatch):
    """Closing the generator early (as Starlette does on disconnect) counts once.

    ``aclose()`` throws ``GeneratorExit`` into the generator at its suspended
    ``yield``; the nested ``async with`` blocks unwind through that exception
    without any special handling needed here, confirming the technique note
    in the task brief: no ``athrow(asyncio.CancelledError)`` fallback required.
    """

    async def body():
        yield b'{"done": false}\n'
        yield b'{"done": true, "eval_count": 1, "eval_duration": 1}\n'

    def handler(request):
        return httpx.Response(200, content=body())

    _patch_client(monkeypatch, handler)
    labels = {"model": "abort-outcome", "endpoint": "/api/chat", "status": oe.STATUS_ABORTED}
    before = metric_value("ollama_requests_total", labels)

    async def drive():
        gen = oe.stream_and_record("/api/chat", "abort-outcome", {}, {}, {}, time.perf_counter())
        await gen.__anext__()
        await gen.aclose()

    asyncio.run(drive())

    after = metric_value("ollama_requests_total", labels)
    assert after - before == 1


# ---------------------------------------------------------------------------
# Status label correctness: 2xx / 4xx / 5xx
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "status_code,expected_status",
    [
        (200, oe.STATUS_SUCCESS),
        (404, oe.STATUS_CLIENT_ERROR),
        (500, oe.STATUS_SERVER_ERROR),
    ],
)
def test_streaming_status_label_matches_upstream_status_code(monkeypatch, status_code, expected_status):
    """The counted outcome tracks the upstream status code on the streaming path."""

    async def body():
        yield b'{"done": true}\n' if status_code == 200 else b"error body\n"

    def handler(request):
        return httpx.Response(status_code, content=body())

    _patch_client(monkeypatch, handler)
    model = f"stream-status-{status_code}"
    labels = {"model": model, "endpoint": "/api/chat", "status": expected_status}
    before = metric_value("ollama_requests_total", labels)

    asyncio.run(_drain(oe.stream_and_record("/api/chat", model, {}, {}, {}, time.perf_counter())))

    after = metric_value("ollama_requests_total", labels)
    assert after - before == 1


@pytest.mark.parametrize(
    "status_code,expected_status",
    [
        (200, oe.STATUS_SUCCESS),
        (404, oe.STATUS_CLIENT_ERROR),
        (500, oe.STATUS_SERVER_ERROR),
    ],
)
def test_buffered_status_label_matches_upstream_status_code(monkeypatch, status_code, expected_status):
    """The counted outcome tracks the upstream status code on the buffered path."""
    body = b'{"done": true}' if status_code == 200 else b'{"error": "boom"}'

    def handler(request):
        return httpx.Response(status_code, content=body)

    _patch_client(monkeypatch, handler)
    model = f"buffered-status-{status_code}"
    labels = {"model": model, "endpoint": "/api/generate", "status": expected_status}
    before = metric_value("ollama_requests_total", labels)

    response = asyncio.run(oe.proxy_and_record("/api/generate", model, {}, {}, {}))

    assert response.status_code == status_code
    after = metric_value("ollama_requests_total", labels)
    assert after - before == 1


# ---------------------------------------------------------------------------
# upstream_error status
# ---------------------------------------------------------------------------


def test_streaming_connect_error_is_counted_as_upstream_error(monkeypatch):
    """A transport-level failure before any bytes arrive is an upstream_error."""

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_client(monkeypatch, handler)
    model = "stream-connect-error"
    labels = {"model": model, "endpoint": "/api/chat", "status": oe.STATUS_UPSTREAM_ERROR}
    before = metric_value("ollama_requests_total", labels)

    async def drive():
        with pytest.raises(httpx.ConnectError):
            await _drain(oe.stream_and_record("/api/chat", model, {}, {}, {}, time.perf_counter()))

    asyncio.run(drive())

    after = metric_value("ollama_requests_total", labels)
    assert after - before == 1
    assert metric_value("ollama_requests_in_flight", {"model": model, "endpoint": "/api/chat"}) == 0


def test_buffered_connect_error_is_counted_as_upstream_error(monkeypatch):
    """A transport-level failure on the buffered path is also an upstream_error."""

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_client(monkeypatch, handler)
    model = "buffered-connect-error"
    labels = {"model": model, "endpoint": "/api/generate", "status": oe.STATUS_UPSTREAM_ERROR}
    before = metric_value("ollama_requests_total", labels)

    async def drive():
        with pytest.raises(httpx.ConnectError):
            await oe.proxy_and_record("/api/generate", model, {}, {}, {})

    asyncio.run(drive())

    after = metric_value("ollama_requests_total", labels)
    assert after - before == 1


# ---------------------------------------------------------------------------
# Tokens per second no longer clips at 100
# ---------------------------------------------------------------------------


def test_tokens_per_second_lands_in_the_400_bucket_not_the_100_bucket():
    """~350 tok/s now has bucket resolution instead of saturating at le=100."""
    response_data = {
        "eval_count": 350,
        "eval_duration": 1_000_000_000,  # exactly one second, so tok/s == eval_count
    }
    model = "tps-model"
    endpoint = "/api/generate"

    # prometheus_client renders bucket boundaries as Go-style floats
    # ("100.0", not "100"), so the le label must match that spelling.
    labels_100 = {"model": model, "endpoint": endpoint, "le": "100.0"}
    labels_400 = {"model": model, "endpoint": endpoint, "le": "400.0"}
    before_100 = metric_value("ollama_tokens_per_second_bucket", labels_100)
    before_400 = metric_value("ollama_tokens_per_second_bucket", labels_400)

    oe.extract_and_record_metrics(response_data, model, endpoint)

    after_100 = metric_value("ollama_tokens_per_second_bucket", labels_100)
    after_400 = metric_value("ollama_tokens_per_second_bucket", labels_400)

    assert after_100 - before_100 == 0
    assert after_400 - before_400 == 1


# ---------------------------------------------------------------------------
# /api/embed is instrumented
# ---------------------------------------------------------------------------


def test_embed_endpoint_records_prompt_metrics_and_success_status(monkeypatch):
    """An embed-shaped payload (no eval_count) still records prompt-side metrics."""
    payload = {
        "total_duration": 2_000_000_000,
        "load_duration": 500_000_000,
        "prompt_eval_count": 12,
    }

    def handler(request):
        return httpx.Response(200, json=payload)

    _patch_client(monkeypatch, handler)
    model = "embed-model"
    endpoint = "/api/embed"

    labels_status = {"model": model, "endpoint": endpoint, "status": oe.STATUS_SUCCESS}
    labels_model_endpoint = {"model": model, "endpoint": endpoint}
    before_status = metric_value("ollama_requests_total", labels_status)
    before_prompt_tokens = metric_value("ollama_tokens_processed_total", labels_model_endpoint)

    response = asyncio.run(oe.proxy_and_record(endpoint, model, {}, {"model": model}, {}))

    assert response.status_code == 200
    after_status = metric_value("ollama_requests_total", labels_status)
    after_prompt_tokens = metric_value("ollama_tokens_processed_total", labels_model_endpoint)
    assert after_status - before_status == 1
    assert after_prompt_tokens - before_prompt_tokens == 12


# ---------------------------------------------------------------------------
# parse_expires_at
# ---------------------------------------------------------------------------


def test_parse_expires_at_accepts_zulu_suffix():
    """A ``Z``-suffixed timestamp parses as UTC."""
    result = oe.parse_expires_at("2026-09-13T14:38:31Z")
    assert result is not None
    assert result.utcoffset().total_seconds() == 0


def test_parse_expires_at_accepts_a_numeric_offset():
    """A explicit numeric UTC offset is preserved."""
    result = oe.parse_expires_at("2026-09-13T14:38:31+02:00")
    assert result is not None
    assert result.utcoffset().total_seconds() == 7200


def test_parse_expires_at_truncates_nanosecond_precision():
    """Go's nine-digit fractional seconds are trimmed to the microsecond Python supports.

    ``datetime.fromisoformat`` tops out at six fractional digits, so the
    surplus three (nanosecond) digits must be dropped rather than causing the
    whole parse to fail.
    """
    result = oe.parse_expires_at("2026-09-13T14:38:31.837532940-07:00")
    assert result is not None
    assert result.microsecond == 837532


def test_parse_expires_at_returns_none_for_missing_value():
    """A ``None`` input (field absent from the /api/ps payload) yields ``None``."""
    assert oe.parse_expires_at(None) is None


def test_parse_expires_at_returns_none_for_unparseable_garbage():
    """Text that isn't a timestamp at all is reported unparseable, not raised."""
    assert oe.parse_expires_at("not-a-timestamp") is None
