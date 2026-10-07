"""Tests for the reverse DNS ``hostname`` label of the per-client counter.

Every lookup is an injected fake: no test touches real DNS. The resolver is
exercised directly, then through the production ASGI stack to prove the request
path never waits for a lookup.
"""

import asyncio
import re
import socket
import threading
import time

import httpx
import pytest
from prometheus_client import REGISTRY

import ollama_exporter
from ollama_exporter import ClientNameResolver

# Captured before any test replaces httpx.AsyncClient with a stub.
_REAL_CLIENT = httpx.AsyncClient

IP = "192.0.2.10"


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self):
        """Start at an arbitrary non-zero instant."""
        self.now = 1000.0

    def __call__(self):
        """Return the current fake time."""
        return self.now


class FakeLookup:
    """Counting lookup returning canned names or raising canned errors.

    Parameters
    ----------
    result : str or BaseException
        Name to return, or exception to raise.
    """

    def __init__(self, result):
        """Store the outcome and zero the call counter."""
        self.result = result
        self.calls = []

    def __call__(self, ip):
        """Record the call, then return or raise the configured outcome."""
        self.calls.append(ip)
        if isinstance(self.result, BaseException):
            raise self.result
        return self.result


def _resolve(resolver, ip=IP):
    """Resolve ``ip`` once and wait for the callback.

    Parameters
    ----------
    resolver : ClientNameResolver
        Resolver under test.
    ip : str, optional
        Address to resolve.

    Returns
    -------
    str
        The hostname handed to the callback.
    """

    async def run():
        """Register a callback and wait until it fires."""
        done = asyncio.get_running_loop().create_future()
        resolver.with_hostname(ip, done.set_result)
        return await asyncio.wait_for(done, 2)

    return asyncio.run(run())


def test_success_normalizes_trailing_dot_and_case():
    """A PTR name loses its root dot and is lowercased."""
    resolver = ClientNameResolver(lookup=FakeLookup("Gpu-Box.Lan."))
    assert _resolve(resolver) == "gpu-box.lan"


def test_timeout_falls_back_to_ip():
    """A lookup slower than the timeout yields the IP."""
    release = threading.Event()

    def slow(ip):
        """Block until the test releases the worker thread."""
        release.wait(5)
        return "late.example"

    resolver = ClientNameResolver(lookup=slow, timeout=0.05)
    try:
        assert _resolve(resolver) == IP
    finally:
        release.set()


def test_nxdomain_uses_ip_and_negative_ttl():
    """A definitive failure gives the IP, cached until negative_ttl elapses."""
    clock = FakeClock()
    lookup = FakeLookup(socket.herror(1, "Unknown host"))
    resolver = ClientNameResolver(lookup=lookup, clock=clock, negative_ttl=300, ttl=3600)
    assert _resolve(resolver) == IP
    clock.now += 299
    assert _resolve(resolver) == IP
    assert len(lookup.calls) == 1
    clock.now += 2
    lookup.result = "now-named.lan"
    assert _resolve(resolver) == "now-named.lan"
    assert len(lookup.calls) == 2


def test_gaierror_is_definitive():
    """gaierror falls back to the IP like any definitive failure."""
    resolver = ClientNameResolver(lookup=FakeLookup(socket.gaierror(-5, "no name")))
    assert _resolve(resolver) == IP


def test_ignore_regex_matches_and_non_matching_is_kept():
    """Names matching the ignore pattern become the IP; others are kept."""
    ignore = re.compile(r"\.ipv6\.example\.org$", re.IGNORECASE)
    ignored = ClientNameResolver(lookup=FakeLookup("a-b.IPv6.example.org."), ignore=ignore)
    kept = ClientNameResolver(lookup=FakeLookup("host.example.org"), ignore=ignore)
    assert _resolve(ignored) == IP
    assert _resolve(kept) == "host.example.org"


def test_cache_hit_is_synchronous_and_skips_lookup():
    """A second call is answered from the cache without a new lookup."""
    lookup = FakeLookup("cached.lan")
    resolver = ClientNameResolver(lookup=lookup)

    async def run():
        """Resolve twice; the second callback must fire before returning."""
        first = asyncio.get_running_loop().create_future()
        resolver.with_hostname(IP, first.set_result)
        await first
        seen = []
        resolver.with_hostname(IP, seen.append)
        assert seen == ["cached.lan"]

    asyncio.run(run())
    assert len(lookup.calls) == 1


def test_positive_ttl_expiry_resolves_again():
    """Once the TTL passes the address is looked up again."""
    clock = FakeClock()
    lookup = FakeLookup("one.lan")
    resolver = ClientNameResolver(lookup=lookup, clock=clock, ttl=10)
    assert _resolve(resolver) == "one.lan"
    clock.now += 11
    lookup.result = "two.lan"
    assert _resolve(resolver) == "two.lan"
    assert len(lookup.calls) == 2


def test_lru_bound_evicts_oldest():
    """Past max_size the least recently used entry is dropped."""
    lookup = FakeLookup("n.lan")
    resolver = ClientNameResolver(lookup=lookup, max_size=2)

    async def run():
        """Resolve three addresses, touching the first one in between."""
        for ip in ("192.0.2.1", "192.0.2.2"):
            done = asyncio.get_running_loop().create_future()
            resolver.with_hostname(ip, done.set_result)
            await done
        resolver.with_hostname("192.0.2.1", lambda name: None)  # LRU touch
        done = asyncio.get_running_loop().create_future()
        resolver.with_hostname("192.0.2.3", done.set_result)
        await done

    asyncio.run(run())
    assert set(resolver._cache) == {"192.0.2.1", "192.0.2.3"}


def test_stale_on_error_keeps_previous_name():
    """A timeout during refresh keeps the expired but known name."""
    clock = FakeClock()
    release = threading.Event()
    state = {"slow": False}

    def lookup(ip):
        """Answer normally until the test makes the resolver hang."""
        if state["slow"]:
            release.wait(5)
        return "known.lan"

    resolver = ClientNameResolver(lookup=lookup, clock=clock, ttl=10, timeout=0.05)
    try:
        assert _resolve(resolver) == "known.lan"
        clock.now += 11
        state["slow"] = True
        assert _resolve(resolver) == "known.lan"
    finally:
        release.set()


def test_try_again_is_transient():
    """herror TRY_AGAIN keeps the previous name instead of flipping to the IP."""
    clock = FakeClock()
    lookup = FakeLookup("known.lan")
    resolver = ClientNameResolver(lookup=lookup, clock=clock, ttl=10)
    assert _resolve(resolver) == "known.lan"
    clock.now += 11
    lookup.result = socket.herror(2, "Host name lookup failure")
    assert _resolve(resolver) == "known.lan"


def test_concurrent_calls_share_one_lookup():
    """Callbacks for the same uncached IP all fire after a single lookup."""
    release = threading.Event()
    calls = []

    def lookup(ip):
        """Hold the lookup open so the calls overlap."""
        calls.append(ip)
        release.wait(5)
        return "shared.lan"

    resolver = ClientNameResolver(lookup=lookup)

    async def run():
        """Register three callbacks, then let the lookup finish."""
        seen = []
        done = asyncio.get_running_loop().create_future()

        def record(name):
            """Collect the name and resolve ``done`` after the third call."""
            seen.append(name)
            if len(seen) == 3:
                done.set_result(None)

        for _ in range(3):
            resolver.with_hostname(IP, record)
        await asyncio.sleep(0.05)
        release.set()
        await asyncio.wait_for(done, 2)
        return seen

    try:
        assert asyncio.run(run()) == ["shared.lan"] * 3
    finally:
        release.set()
    assert calls == [IP]


def test_non_ip_input_is_answered_immediately():
    """A placeholder such as 'unknown' is returned as is, with no lookup."""
    lookup = FakeLookup("never.lan")
    resolver = ClientNameResolver(lookup=lookup)

    async def run():
        """Call and check the callback ran before with_hostname returned."""
        seen = []
        resolver.with_hostname("unknown", seen.append)
        return seen

    assert asyncio.run(run()) == ["unknown"]
    assert lookup.calls == []


def test_failing_callback_does_not_starve_others():
    """An exception in one waiting callback is logged, the rest still run."""
    resolver = ClientNameResolver(lookup=FakeLookup("ok.lan"))

    async def run():
        """Queue a raising callback before a normal one."""
        seen = []

        def boom(name):
            """Fail on purpose."""
            raise RuntimeError("boom")

        done = asyncio.get_running_loop().create_future()

        def record(name):
            """Collect the name and signal that the last callback ran."""
            seen.append(name)
            done.set_result(None)

        resolver.with_hostname(IP, boom)
        resolver.with_hostname(IP, record)
        await asyncio.wait_for(done, 2)
        return seen

    assert asyncio.run(run()) == ["ok.lan"]


class _FakeUpstream:
    """Stand-in for ``httpx.AsyncClient`` returning a canned Ollama reply."""

    def __init__(self, *args, **kwargs):
        """Accept and ignore the real client's constructor arguments."""

    def build_request(self, method, url, **kwargs):
        """Build a plain request, mirroring httpx.AsyncClient."""
        return httpx.Request(method, url)

    async def send(self, request, stream=False):
        """Return a minimal successful non-streaming chat response."""
        return httpx.Response(200, json={"message": {"content": "hi"}, "done": True})

    async def aclose(self):
        """Nothing to release."""


def _value(client, hostname):
    """Read the per-client counter for model ``hostmodel``.

    Parameters
    ----------
    client : str
        Value of the ``client`` label.
    hostname : str
        Value of the ``hostname`` label.

    Returns
    -------
    float
        Sample value, ``0.0`` when the series does not exist.
    """
    return (
        REGISTRY.get_sample_value(
            "ollama_client_requests_total",
            {"model": "hostmodel", "client": client, "hostname": hostname},
        )
        or 0.0
    )


def test_request_path_does_not_wait_for_the_lookup(monkeypatch):
    """The 200 comes back while the lookup hangs; the count lands afterwards."""
    release = threading.Event()

    def lookup(ip):
        """Block until released, then answer."""
        release.wait(5)
        return "Blocked.Lan."

    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", True)
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", ClientNameResolver(lookup=lookup, timeout=2))
    monkeypatch.setattr(ollama_exporter.httpx, "AsyncClient", _FakeUpstream)
    wrapped = ollama_exporter.build_asgi_app("127.0.0.1")
    peer = "198.51.100.9"
    before_name = _value(peer, "blocked.lan")
    before_ip = _value(peer, peer)

    async def run():
        """Post a chat request, check the counter, release, check again."""
        transport = httpx.ASGITransport(app=wrapped, client=(peer, 1234))
        async with _REAL_CLIENT(transport=transport, base_url="http://test") as client:
            response = await client.post("/api/chat", json={"model": "hostmodel", "stream": False})
            assert response.status_code == 200
            assert _value(peer, "blocked.lan") == before_name
            release.set()
            # Poll rather than sleep a fixed time: a loaded CI runner may take
            # a while to hand the thread's result back to the loop.
            for _ in range(200):
                if _value(peer, "blocked.lan") > before_name:
                    break
                await asyncio.sleep(0.01)
        assert _value(peer, "blocked.lan") == before_name + 1
        assert _value(peer, peer) == before_ip

    try:
        asyncio.run(run())
    finally:
        release.set()


def test_parse_args_defaults(monkeypatch):
    """Resolution is off and there is no ignore pattern by default."""
    monkeypatch.delenv("EXPORTER_RESOLVE_CLIENTS", raising=False)
    monkeypatch.delenv("EXPORTER_CLIENT_HOSTNAME_IGNORE", raising=False)
    args = ollama_exporter.parse_args([])
    assert args.resolve_clients is False
    assert args.client_hostname_ignore is None


@pytest.mark.parametrize("value", ["1", "true", "TRUE", "Yes", "on"])
def test_parse_args_env_enables_resolution(monkeypatch, value):
    """Truthy spellings of EXPORTER_RESOLVE_CLIENTS enable resolution."""
    monkeypatch.setenv("EXPORTER_RESOLVE_CLIENTS", value)
    assert ollama_exporter.parse_args([]).resolve_clients is True


def test_parse_args_empty_ignore_env_is_none(monkeypatch):
    """An empty ignore variable means no filter, not match-everything."""
    monkeypatch.setenv("EXPORTER_CLIENT_HOSTNAME_IGNORE", "")
    assert ollama_exporter.parse_args([]).client_hostname_ignore is None


def test_parse_args_valid_ignore_is_compiled_case_insensitive(monkeypatch):
    """A valid pattern is compiled with IGNORECASE."""
    monkeypatch.setenv("EXPORTER_CLIENT_HOSTNAME_IGNORE", r"\.ipv6\.example\.org$")
    pattern = ollama_exporter.parse_args([]).client_hostname_ignore
    assert pattern.search("A.IPV6.example.ORG")


def test_parse_args_invalid_ignore_exits():
    """An invalid regex makes argparse exit with an error."""
    with pytest.raises(SystemExit):
        ollama_exporter.parse_args(["--client-hostname-ignore", "("])


def test_parse_args_cli_overrides_env(monkeypatch):
    """CLI flags win over the environment."""
    monkeypatch.delenv("EXPORTER_RESOLVE_CLIENTS", raising=False)
    monkeypatch.setenv("EXPORTER_CLIENT_HOSTNAME_IGNORE", "env")
    args = ollama_exporter.parse_args(["--resolve-clients", "--client-hostname-ignore", "cli"])
    assert args.resolve_clients is True
    assert args.client_hostname_ignore.pattern == "cli"


def _run_main(monkeypatch, argv, serve_hook=None):
    """Run main() against stubbed uvicorn and return nothing.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Used to stub uvicorn and the network-facing helpers.
    argv : list of str
        Command-line arguments passed to main().
    serve_hook : callable, optional
        Called from the fake ``serve()``, to observe state while "serving".
    """
    # main() writes these module globals and the logger level; monkeypatch
    # restores them at teardown so nothing leaks into other tests.
    monkeypatch.setattr(ollama_exporter, "OLLAMA_HOST", ollama_exporter.OLLAMA_HOST)
    monkeypatch.setattr(ollama_exporter, "REQUEST_TIMEOUT", ollama_exporter.REQUEST_TIMEOUT)
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", ollama_exporter.CLIENT_RESOLVER)
    monkeypatch.setattr(ollama_exporter.logger, "level", ollama_exporter.logger.level)

    class FakeServer:
        """Server stub whose serve() returns immediately."""

        def __init__(self, config):
            """Ignore the config."""

        async def serve(self, sockets=None):
            """Pretend to serve."""
            if serve_hook is not None:
                serve_hook()

    async def noop():
        """Skip the upstream connectivity check."""

    monkeypatch.setattr(ollama_exporter.uvicorn, "Config", lambda app, **kw: object())
    monkeypatch.setattr(ollama_exporter.uvicorn, "Server", FakeServer)
    monkeypatch.setattr(ollama_exporter, "verify_ollama_connection", noop)
    monkeypatch.setattr(ollama_exporter, "bind_listen_socket", lambda h, p: None)
    monkeypatch.setattr(ollama_exporter, "TRACK_CLIENTS", False)
    monkeypatch.setattr("sys.argv", ["ollama_exporter", *argv])
    asyncio.run(ollama_exporter.main())


def test_main_sets_resolver_with_track_and_resolve(monkeypatch):
    """Both flags together install a ClientNameResolver with the ignore regex."""
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", None)
    _run_main(monkeypatch, ["--track-clients", "--resolve-clients", "--client-hostname-ignore", "x$"])
    resolver = ollama_exporter.CLIENT_RESOLVER
    assert isinstance(resolver, ClientNameResolver)
    assert resolver.ignore.pattern == "x$"
    # Restore before teardown so the resolver does not leak into other tests.
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", None)


def test_main_closes_resolver_after_serve(monkeypatch):
    """The resolver is closed once serve() returns, not before."""
    closed = []
    monkeypatch.setattr(ClientNameResolver, "close", lambda self: closed.append(True))
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", None)
    _run_main(
        monkeypatch,
        ["--track-clients", "--resolve-clients"],
        serve_hook=lambda: closed.append(False),
    )
    assert closed == [False, True]


def test_close_does_not_wait_for_a_stuck_lookup():
    """close() returns at once and drops queued lookups."""
    release = threading.Event()
    resolver = ClientNameResolver(lookup=lambda ip: release.wait(5) or "x.lan")

    async def run():
        """Start a lookup that hangs, then close."""
        resolver.with_hostname(IP, lambda name: None)
        await asyncio.sleep(0.05)
        started = time.monotonic()
        resolver.close()
        return time.monotonic() - started

    try:
        assert asyncio.run(run()) < 1
    finally:
        release.set()


def test_main_resolve_without_track_leaves_resolver_unset(monkeypatch):
    """--resolve-clients alone is ignored (with a warning)."""
    monkeypatch.setattr(ollama_exporter, "CLIENT_RESOLVER", None)
    _run_main(monkeypatch, ["--resolve-clients"])
    assert ollama_exporter.CLIENT_RESOLVER is None
