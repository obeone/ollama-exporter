"""Tests for the /api/ps residency poller: gauges, eviction, swaps, upstream health.

Each test uses a model name unique to itself so leftover Prometheus series
from earlier tests in this module (or in ``test_metrics.py``, sharing the
same global registry) cannot be mistaken for this test's own state.
"""

import asyncio
import time
from datetime import datetime, timezone

import httpx
import pytest
from fastapi.testclient import TestClient
from prometheus_client import REGISTRY

import ollama_exporter as oe


def metric_value(name, labels=None):
    """Read one Prometheus sample, treating an absent series as zero.

    Parameters
    ----------
    name : str
        Fully qualified sample name.
    labels : dict of str to str, optional
        Label values identifying the sample. Defaults to no labels.

    Returns
    -------
    float
        The sample's current value, or ``0.0`` when the series does not
        exist. Use :func:`prometheus_client.REGISTRY.get_sample_value`
        directly instead of this helper when the distinction between "zero"
        and "absent" actually matters to the assertion.
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


def _ps_entry(name, **overrides):
    """Build one ``/api/ps`` model entry with sane defaults.

    Parameters
    ----------
    name : str
        Model name for the entry's ``"name"`` field.
    **overrides
        Fields to override on top of the defaults, e.g. ``expires_at`` or a
        nested ``details`` dict.

    Returns
    -------
    dict
        One element of the ``/api/ps`` response's ``"models"`` array.
    """
    entry = {
        "name": name,
        "size": 5_000_000_000,
        "size_vram": 4_000_000_000,
        "context_length": 4096,
        "expires_at": "2026-09-13T15:00:00Z",
        "details": {
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": "Q4_0",
        },
    }
    entry.update(overrides)
    return entry


@pytest.fixture(autouse=True)
def _reset_resident_state():
    """Clear the previous-poll residency state before and after each test.

    ``_RESIDENT_MODELS`` is module-level and shared with production code;
    leaving it dirty between tests would make one test's model look
    already-resident to the next test, corrupting the swap-counter
    assertions in particular.
    """
    oe._RESIDENT_MODELS.clear()
    yield
    oe._RESIDENT_MODELS.clear()


def test_residency_poll_populates_then_clears_model_gauges():
    """A poll publishes every residency series; an empty poll removes them all."""
    model = "residency-populate"
    entry = _ps_entry(model)
    info_labels = {
        "model": model,
        "family": "llama",
        "parameter_size": "8B",
        "quantization_level": "Q4_0",
    }

    oe.update_residency_metrics([entry])

    assert REGISTRY.get_sample_value("ollama_model_loaded", {"model": model}) == 1
    assert REGISTRY.get_sample_value("ollama_model_vram_bytes", {"model": model}) == entry["size_vram"]
    assert REGISTRY.get_sample_value("ollama_model_size_bytes", {"model": model}) == entry["size"]
    assert (
        REGISTRY.get_sample_value("ollama_model_context_length", {"model": model})
        == entry["context_length"]
    )
    assert REGISTRY.get_sample_value("ollama_model_expires_seconds", {"model": model}) is not None
    assert REGISTRY.get_sample_value("ollama_model_info", info_labels) == 1
    assert metric_value("ollama_models_loaded") == 1

    # An empty poll means the model is no longer resident: its per-model
    # series must disappear entirely rather than read as zero.
    oe.update_residency_metrics([])

    assert REGISTRY.get_sample_value("ollama_model_loaded", {"model": model}) is None
    assert REGISTRY.get_sample_value("ollama_model_vram_bytes", {"model": model}) is None
    assert REGISTRY.get_sample_value("ollama_model_size_bytes", {"model": model}) is None
    assert REGISTRY.get_sample_value("ollama_model_context_length", {"model": model}) is None
    assert REGISTRY.get_sample_value("ollama_model_expires_seconds", {"model": model}) is None
    assert REGISTRY.get_sample_value("ollama_model_info", info_labels) is None
    assert metric_value("ollama_models_loaded") == 0


def test_swap_counter_increments_only_on_genuine_entry():
    """Staying resident across polls counts one swap; leaving and returning counts two."""
    model = "residency-swap"
    entry = _ps_entry(model)
    labels = {"model": model}
    before = metric_value("ollama_model_swaps_total", labels)

    # Three consecutive polls with the same resident model: only the first
    # one is a genuine entry into the resident set.
    oe.update_residency_metrics([entry])
    oe.update_residency_metrics([entry])
    oe.update_residency_metrics([entry])
    assert metric_value("ollama_model_swaps_total", labels) - before == 1

    # Evicted, then reloaded from cold: a second genuine entry.
    oe.update_residency_metrics([])
    oe.update_residency_metrics([entry])
    assert metric_value("ollama_model_swaps_total", labels) - before == 2


def test_refresh_model_residency_marks_upstream_up_on_success(monkeypatch):
    """A healthy /api/ps poll sets ollama_upstream_up to 1 and updates the gauges."""
    model = "residency-refresh-ok"
    entry = _ps_entry(model)

    def handler(request):
        return httpx.Response(200, json={"models": [entry]})

    _patch_client(monkeypatch, handler)

    asyncio.run(oe.refresh_model_residency())

    assert metric_value("ollama_upstream_up") == 1
    assert REGISTRY.get_sample_value("ollama_model_loaded", {"model": model}) == 1


def test_refresh_model_residency_marks_upstream_down_without_raising(monkeypatch):
    """A failed /api/ps poll sets ollama_upstream_up to 0 and swallows the error.

    The poller runs forever in a background task; letting an exception
    escape here would kill it silently, so the failure must be contained.
    """

    def handler(request):
        raise httpx.ConnectError("connection refused", request=request)

    _patch_client(monkeypatch, handler)

    asyncio.run(oe.refresh_model_residency())  # must not raise

    assert metric_value("ollama_upstream_up") == 0


def test_repulled_model_replaces_its_info_series():
    """Changed model details drop the previous ollama_model_info child.

    Re-pulling a tag can swap the quantization under an unchanged model name.
    The info series is a join target, so two children for one model would make
    any dashboard join return duplicate rows.
    """
    model = "residency-repull"
    first = _ps_entry(model)
    second = _ps_entry(model, details=dict(first["details"], quantization_level="Q8_0"))

    def info_labels(quantization):
        return {
            "model": model,
            "family": "llama",
            "parameter_size": "8B",
            "quantization_level": quantization,
        }

    oe.update_residency_metrics([first])
    oe.update_residency_metrics([second])

    assert REGISTRY.get_sample_value("ollama_model_info", info_labels("Q4_0")) is None
    assert REGISTRY.get_sample_value("ollama_model_info", info_labels("Q8_0")) == 1


def test_expires_seconds_counts_down_from_the_injected_reference():
    """The keep_alive countdown is expires_at minus the reference instant.

    ``now`` is injected rather than read from the clock so the assertion is an
    exact number of seconds instead of a tolerance around wall time.
    """
    model = "residency-countdown"
    entry = _ps_entry(model, expires_at="2026-09-13T12:05:00Z")
    reference = datetime(2026, 9, 13, 12, 0, 0, tzinfo=timezone.utc)

    oe.update_residency_metrics([entry], now=reference)

    assert REGISTRY.get_sample_value("ollama_model_expires_seconds", {"model": model}) == 300

    # Past its expiry the countdown goes negative rather than vanishing, which
    # is what makes an overdue eviction visible on a graph.
    oe.update_residency_metrics(
        [entry], now=datetime(2026, 9, 13, 12, 6, 0, tzinfo=timezone.utc)
    )
    assert REGISTRY.get_sample_value("ollama_model_expires_seconds", {"model": model}) == -60


def test_the_lifespan_starts_a_poller_that_populates_gauges(monkeypatch):
    """Booting the app runs the residency poller as a background task.

    This is the only test that goes through the real ASGI lifespan, because
    the wiring between ``lifespan`` and ``poll_model_residency`` is exactly
    what unit tests on ``refresh_model_residency`` cannot reach. The interval
    is shortened so the first poll lands well inside the wait below.
    """
    model = "residency-lifespan"

    def handler(request):
        assert request.url.path == "/api/ps"
        return httpx.Response(200, json={"models": [_ps_entry(model)]})

    _patch_client(monkeypatch, handler)
    monkeypatch.setattr(oe, "PS_INTERVAL_SECONDS", 0.01)

    # Entering the context manager is what runs the lifespan; TestClient
    # drives the app on its own thread, so the poll happens while we wait.
    with TestClient(oe.app) as client:
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            if REGISTRY.get_sample_value("ollama_model_loaded", {"model": model}) == 1:
                break
            time.sleep(0.01)

        assert metric_value("ollama_upstream_up") == 1
        assert REGISTRY.get_sample_value("ollama_model_loaded", {"model": model}) == 1
        # The app still serves while the poller runs.
        assert client.get("/metrics").status_code == 200
