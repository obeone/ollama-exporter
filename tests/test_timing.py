"""Proxy-measured timing of ``/v1`` responses, and the native path left alone.

``MetricsTap`` is driven directly with a fake clock so every duration is exact;
one end-to-end test checks that the relay really calls ``start()``.
"""

import asyncio
import itertools
import json

from prometheus_client import REGISTRY

import ollama_exporter
from ollama_exporter import MetricsTap
from tests.test_proxy import FakeOllama, PROPAGATION_TIMEOUT, raw_post, run_scenario

_models = itertools.count()


class FakeClock:
    """Manually advanced stand-in for ``time.monotonic``."""

    def __init__(self):
        """Start at time zero."""
        self.now = 0.0

    def __call__(self):
        """Return the current fake time in seconds."""
        return self.now


def new_model():
    """Return a model label unused by any other test.

    Returns
    -------
    str
        Fresh label, so registry reads never see other tests' samples.
    """
    return f"timing-{next(_models)}"


def sse(obj):
    """Encode one object as an SSE ``data:`` event.

    Parameters
    ----------
    obj : dict or str
        Payload, JSON-encoded unless already a string (e.g. ``[DONE]``).

    Returns
    -------
    bytes
        The event with its blank-line terminator.
    """
    payload = obj if isinstance(obj, str) else json.dumps(obj)
    return b"data: " + payload.encode() + b"\n\n"


def chat_chunk(content=None, **delta):
    """Build a chat completion chunk.

    Parameters
    ----------
    content : str, optional
        ``delta.content`` value.
    **delta
        Extra delta fields (``role``, ``reasoning``...).

    Returns
    -------
    dict
        Chunk with one choice.
    """
    if content is not None:
        delta["content"] = content
    return {"choices": [{"index": 0, "delta": delta}]}


def usage_chunk(completion_tokens):
    """Build the usage-only final chunk.

    Parameters
    ----------
    completion_tokens : int
        Completion token count to report.

    Returns
    -------
    dict
        Chunk with empty ``choices`` and a ``usage`` object.
    """
    return {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": completion_tokens}}


def hist(name, model, api=None):
    """Read the observation count and sum of a histogram.

    Parameters
    ----------
    name : str
        Histogram name without suffix.
    model : str
        Model label.
    api : str, optional
        ``api`` label, when the histogram has one.

    Returns
    -------
    tuple of (float, float)
        ``(count, sum)``, zeros when never observed.
    """
    labels = {"model": model}
    if api:
        labels["api"] = api
    return (
        REGISTRY.get_sample_value(name + "_count", labels) or 0.0,
        REGISTRY.get_sample_value(name + "_sum", labels) or 0.0,
    )


def run_tap(path, steps, model, status_ok=True):
    """Replay timed chunks through a tap.

    Parameters
    ----------
    path : str
        Request path.
    steps : list of (float, bytes)
        ``(clock value, chunk)`` pairs fed in order; the clock is set before
        each feed. The tap starts at time 0.
    model : str
        Model label.
    status_ok : bool, optional
        When false ``finish()`` is not called, as for a non-200 response.
    """
    clock = FakeClock()
    tap = MetricsTap(path, model, clock=clock)
    tap.start()
    for at, chunk in steps:
        clock.now = at
        tap.feed(chunk)
    if status_ok:
        tap.finish()


def stream_steps():
    """Return a four-token chat stream with timestamps.

    Returns
    -------
    list of (float, bytes)
        Role chunk at 1, content at 3 and 5, usage at 6, ``[DONE]`` at 6.
    """
    return [
        (1.0, sse(chat_chunk(role="assistant", content=""))),
        (3.0, sse(chat_chunk("a"))),
        (4.0, sse(chat_chunk("b"))),
        (5.0, sse(chat_chunk("c"))),
        (6.0, sse(usage_chunk(4))),
        (6.0, sse("[DONE]")),
    ]


def test_openai_stream_observes_response_seconds_with_openai_label():
    """A streamed /v1 response observes wall-clock duration under api=openai."""
    m = new_model()
    run_tap("/v1/chat/completions", stream_steps(), m)
    assert hist("ollama_response_seconds", m, "openai") == (1, 6.0)
    assert hist("ollama_response_seconds", m, "native") == (0, 0)


def test_openai_non_streaming_observes_response_seconds_without_usage():
    """A plain JSON /v1 body observes duration even with no usage object."""
    m = new_model()
    body = json.dumps({"choices": [{"message": {"content": "hi"}}]}).encode()
    run_tap("/v1/completions", [(2.5, body)], m)
    assert hist("ollama_response_seconds", m, "openai") == (1, 2.5)


def test_openai_non_streaming_does_not_observe_tokens_per_second_or_ttft():
    """Non-streaming duration includes load and prefill, so no rate or TTFT."""
    m = new_model()
    body = json.dumps({"choices": [{"message": {"content": "hi"}}],
                       "usage": {"prompt_tokens": 1, "completion_tokens": 50}}).encode()
    run_tap("/v1/chat/completions", [(2.5, body)], m)
    assert hist("ollama_tokens_per_second", m, "openai")[0] == 0
    assert hist("ollama_time_to_first_token_seconds", m)[0] == 0


def test_openai_stream_observes_time_to_first_token_at_first_content_chunk():
    """TTFT runs from start() to the first chunk with generated content."""
    m = new_model()
    run_tap("/v1/chat/completions", stream_steps(), m)
    assert hist("ollama_time_to_first_token_seconds", m) == (1, 3.0)


def test_openai_stream_tokens_per_second_uses_n_minus_one_over_content_interval():
    """Rate is (completion_tokens - 1) / (last content - first content)."""
    m = new_model()
    run_tap("/v1/chat/completions", stream_steps(), m)
    # 4 tokens, content from t=3 to t=5: 3 / 2.
    assert hist("ollama_tokens_per_second", m, "openai") == (1, 1.5)


def test_openai_stream_counts_reasoning_deltas_as_content():
    """Thinking models emit reasoning first; it ends the wait for a token."""
    m = new_model()
    for key in ("reasoning", "reasoning_content"):
        model = f"{m}-{key}"
        run_tap("/v1/chat/completions", [
            (2.0, sse(chat_chunk(**{key: "hmm"}))),
            (3.0, sse("[DONE]")),
        ], model)
        assert hist("ollama_time_to_first_token_seconds", model) == (1, 2.0)


def test_openai_completions_text_counts_as_content():
    """/v1/completions chunks carry generated text in choices[].text."""
    m = new_model()
    run_tap("/v1/completions", [
        (1.0, sse({"choices": [{"text": ""}]})),
        (2.0, sse({"choices": [{"text": "a"}]})),
        (4.0, sse({"choices": [{"text": "b"}]})),
        (4.0, sse(usage_chunk(3))),
    ], m)
    assert hist("ollama_time_to_first_token_seconds", m) == (1, 2.0)
    assert hist("ollama_tokens_per_second", m, "openai") == (1, 1.0)


def test_openai_stream_ignores_role_only_empty_and_usage_chunks_for_timing():
    """Role-only, empty-delta, usage-only chunks and [DONE] are not content."""
    m = new_model()
    run_tap("/v1/chat/completions", [
        (1.0, sse(chat_chunk(role="assistant"))),
        (2.0, sse(chat_chunk(""))),
        (9.0, sse(usage_chunk(5))),
        (9.0, sse("[DONE]")),
    ], m)
    assert hist("ollama_response_seconds", m, "openai") == (1, 9.0)
    assert hist("ollama_time_to_first_token_seconds", m)[0] == 0
    assert hist("ollama_tokens_per_second", m, "openai")[0] == 0


def test_openai_non_200_response_observes_nothing():
    """The relay skips finish() for non-200 responses, so nothing is recorded."""
    m = new_model()
    run_tap("/v1/chat/completions", stream_steps(), m, status_ok=False)
    for name in ("ollama_response_seconds", "ollama_time_to_first_token_seconds",
                 "ollama_tokens_per_second"):
        assert REGISTRY.get_sample_value(name + "_count", {"model": m, "api": "openai"}) is None
        assert REGISTRY.get_sample_value(name + "_count", {"model": m}) is None


def test_openai_stream_without_usage_observes_response_and_ttft_but_not_rate():
    """Without stream_options.include_usage there is no token count for a rate."""
    m = new_model()
    run_tap("/v1/chat/completions", stream_steps()[:4] + [(6.0, sse("[DONE]"))], m)
    assert hist("ollama_response_seconds", m, "openai") == (1, 6.0)
    assert hist("ollama_time_to_first_token_seconds", m) == (1, 3.0)
    assert hist("ollama_tokens_per_second", m, "openai")[0] == 0


def test_openai_single_content_chunk_stream_observes_no_rate():
    """One content chunk gives a zero interval, so no tokens per second."""
    m = new_model()
    run_tap("/v1/chat/completions", [
        (2.0, sse(chat_chunk("a"))),
        (3.0, sse(usage_chunk(1))),
    ], m)
    assert hist("ollama_time_to_first_token_seconds", m) == (1, 2.0)
    assert hist("ollama_tokens_per_second", m, "openai")[0] == 0


def test_openai_stream_timing_survives_json_split_across_chunks():
    """An SSE event cut mid-JSON is stamped when its line completes."""
    m = new_model()
    event = sse(chat_chunk("a"))
    run_tap("/v1/chat/completions", [
        (1.0, event[:10]),
        (2.0, event[10:]),
        (3.0, sse("[DONE]")),
    ], m)
    assert hist("ollama_time_to_first_token_seconds", m) == (1, 2.0)


def test_native_response_uses_ollama_durations_not_proxy_timing():
    """Native responses keep Ollama's own total_duration and api=native label."""
    m = new_model()
    final = json.dumps({"done": True, "total_duration": 4 * 10**9,
                        "eval_count": 10, "eval_duration": 2 * 10**9}).encode() + b"\n"
    run_tap("/api/chat", [(100.0, final)], m)
    assert hist("ollama_response_seconds", m, "native") == (1, 4.0)
    assert hist("ollama_response_seconds", m, "openai") == (0, 0)
    assert hist("ollama_tokens_per_second", m, "native") == (1, 5.0)
    assert hist("ollama_time_to_first_token_seconds", m)[0] == 0


def test_relay_starts_the_clock_so_proxied_v1_response_is_timed(monkeypatch):
    """End to end, a /v1 reply through the proxy observes response_seconds."""
    m = new_model()
    body = json.dumps({"model": m, "choices": [{"message": {"content": "x"}}]}).encode()

    async def scenario(fake, port):
        reader, writer = await asyncio.open_connection("127.0.0.1", port)
        writer.write(raw_post("/v1/chat/completions", {"model": m}))
        await writer.drain()
        data = b""
        while body not in data:
            data += await asyncio.wait_for(reader.read(4096), PROPAGATION_TIMEOUT)
        writer.close()
        for _ in range(100):
            if hist("ollama_response_seconds", m, "openai")[0]:
                break
            await asyncio.sleep(0.02)

    run_scenario(monkeypatch, lambda: FakeOllama("reply", reply_body=body), scenario)
    count, total = hist("ollama_response_seconds", m, "openai")
    assert count == 1 and total >= 0
