"""Tests for the optional per-client request counter.

The counter relies on uvicorn's proxy-headers middleware to turn
``X-Forwarded-For`` into ``request.client``, but only for trusted peers. These
tests pin the CLI/env wiring and drive the real app through that middleware to
check that a spoofed header from an untrusted peer is never used as a label.
"""

import asyncio

import httpx
import pytest
from prometheus_client import REGISTRY
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import ollama_exporter

PEER = "10.0.0.5"


def test_parse_args_defaults(monkeypatch):
    """Tracking is off and only the loopback is trusted when nothing is set."""
    monkeypatch.delenv("EXPORTER_TRACK_CLIENTS", raising=False)
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    args = ollama_exporter.parse_args([])
    assert args.track_clients is False
    assert args.forwarded_allow_ips == "127.0.0.1"


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
def test_parse_args_env_enables_tracking(monkeypatch, value):
    """Truthy spellings of EXPORTER_TRACK_CLIENTS enable tracking."""
    monkeypatch.setenv("EXPORTER_TRACK_CLIENTS", value)
    assert ollama_exporter.parse_args([]).track_clients is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "banana"])
def test_parse_args_env_falsy_values_keep_tracking_off(monkeypatch, value):
    """Anything that is not a truthy spelling leaves tracking disabled."""
    monkeypatch.setenv("EXPORTER_TRACK_CLIENTS", value)
    assert ollama_exporter.parse_args([]).track_clients is False


def test_parse_args_env_sets_allow_ips(monkeypatch):
    """FORWARDED_ALLOW_IPS provides the default trusted proxy list."""
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    assert ollama_exporter.parse_args([]).forwarded_allow_ips == "10.0.0.0/8"


def test_parse_args_cli_overrides_env(monkeypatch):
    """Explicit CLI flags win over the environment."""
    monkeypatch.setenv("EXPORTER_TRACK_CLIENTS", "0")
    monkeypatch.setenv("FORWARDED_ALLOW_IPS", "10.0.0.0/8")
    args = ollama_exporter.parse_args(
        ["--track-clients", "--forwarded-allow-ips", "*"]
    )
    assert args.track_clients is True
    assert args.forwarded_allow_ips == "*"


class _FakeUpstream:
    """Stand-in for ``httpx.AsyncClient`` returning a canned Ollama reply."""

    def __init__(self, *args, **kwargs):
        """Accept and ignore the real client's constructor arguments."""

    async def __aenter__(self):
        """Enter the async context, mirroring httpx.AsyncClient."""
        return self

    async def __aexit__(self, *exc):
        """Leave the async context without suppressing exceptions."""
        return False

    async def post(self, *args, **kwargs):
        """Return a minimal successful non-streaming chat response."""
        return httpx.Response(200, json={"message": {"content": "hi"}, "done": True})


def _send_chat(monkeypatch, trusted_hosts, forwarded_for=None):
    """Post one chat request to the app as seen from ``PEER``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub the upstream Ollama call.
    trusted_hosts : str
        Value for the middleware's ``trusted_hosts``, same format as
        ``--forwarded-allow-ips``.
    forwarded_for : str, optional
        Value of the ``X-Forwarded-For`` header to send, if any.
    """
    monkeypatch.setattr(ollama_exporter.httpx, "AsyncClient", _FakeUpstream)
    wrapped = ProxyHeadersMiddleware(ollama_exporter.app, trusted_hosts=trusted_hosts)

    async def run():
        """Drive the request through the ASGI stack with a fixed peer."""
        transport = httpx.ASGITransport(app=wrapped, client=(PEER, 1234))
        # The stub replaces httpx.AsyncClient globally, so the test client
        # must be the class captured before any patching.
        async with _REAL_CLIENT(transport=transport, base_url="http://test") as client:
            headers = {"X-Forwarded-For": forwarded_for} if forwarded_for else {}
            response = await client.post(
                "/api/chat", json={"model": "m", "stream": False}, headers=headers
            )
            assert response.status_code == 200

    asyncio.run(run())


# Captured at import, before any test replaces ollama_exporter.httpx.AsyncClient
# (the module attribute is the shared httpx module, so the patch is global).
_REAL_CLIENT = httpx.AsyncClient


def _count(client):
    """Read the per-client counter for model ``m`` and the given label.

    Parameters
    ----------
    client : str
        Value of the ``client`` label.

    Returns
    -------
    float
        Current sample value, ``0.0`` when the series does not exist yet.
    """
    value = REGISTRY.get_sample_value(
        "ollama_client_requests_total", {"model": "m", "client": client}
    )
    return value or 0.0


def test_trusted_peer_forwarded_address_is_the_label(monkeypatch):
    """A trusted proxy's X-Forwarded-For becomes the client label."""
    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", True)
    before = _count("203.0.113.7")
    _send_chat(monkeypatch, PEER, "203.0.113.7")
    assert _count("203.0.113.7") == before + 1


def test_untrusted_peer_cannot_spoof_the_label(monkeypatch):
    """A spoofed header from an untrusted peer is ignored: the peer is used."""
    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", True)
    before_spoof = _count("203.0.113.7")
    before_peer = _count(PEER)
    _send_chat(monkeypatch, "192.0.2.1", "203.0.113.7")
    assert _count("203.0.113.7") == before_spoof
    assert _count(PEER) == before_peer + 1


def test_multi_hop_chain_skips_trusted_hops(monkeypatch):
    """With a trusted CIDR, the first untrusted hop from the right is the client."""
    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", True)
    before = _count("203.0.113.7")
    _send_chat(monkeypatch, "10.0.0.0/8", "203.0.113.7, 10.0.0.9")
    assert _count("203.0.113.7") == before + 1


def test_tracking_disabled_does_not_increment(monkeypatch):
    """With tracking off no per-client series is created or incremented."""
    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", False)
    before = _count("203.0.113.7")
    before_peer = _count(PEER)
    _send_chat(monkeypatch, PEER, "203.0.113.7")
    assert _count("203.0.113.7") == before
    assert _count(PEER) == before_peer
