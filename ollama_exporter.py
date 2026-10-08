import os
import argparse
import asyncio
import concurrent.futures
import httpx
import ipaddress
import json
import logging
import re
import socket
import time
from collections import OrderedDict
from urllib.parse import urlsplit

import anyio
from fastapi import FastAPI, Request, Response
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

import uvicorn
from prometheus_client import Counter, Gauge, Histogram, generate_latest, CONTENT_TYPE_LATEST

# Default values, overridable via environment variables or CLI arguments.
# CLI arguments take precedence over environment variables (see parse_args).
DEFAULT_OLLAMA_HOST = "http://localhost:11434"
DEFAULT_LISTEN_HOST = "::"  # dual-stack: binds both IPv6 and IPv4
DEFAULT_LISTEN_PORT = 8000
DEFAULT_LOG_LEVEL = "INFO"
# Total deadline, in seconds, for one inference request (0 disables it).
DEFAULT_REQUEST_TIMEOUT = 1800.0

# Only the loopback (both families, like uvicorn's own default) is trusted to
# set X-Forwarded-For, so a client reaching the exporter directly cannot spoof
# its address.
DEFAULT_FORWARDED_ALLOW_IPS = "127.0.0.1,::1"

# Values accepted as "true" for boolean environment variables.
TRUTHY_VALUES = frozenset({"1", "true", "yes", "on"})

# Reverse DNS (PTR) resolution of client addresses, see ClientNameResolver.
# A lookup slower than this stops holding back the metric increment (the
# request itself never waits); its result still fills the cache when it ends.
# A cold PTR lookup of an IPv6 client was measured at 1.13 s in production.
DEFAULT_RESOLVE_TIMEOUT = 3.0
# How long a resolved name (or an ignored one) is trusted.
DEFAULT_RESOLVE_TTL = 3600.0
# How long a failed lookup is remembered, so an address without a PTR record
# (or a dead resolver) is not queried again on every request.
DEFAULT_RESOLVE_NEGATIVE_TTL = 300.0
# Upper bound on cached addresses, so a scan cannot grow the cache forever.
DEFAULT_RESOLVE_MAX_SIZE = 4096
# Concurrent blocking lookups: gethostbyaddr has no async form, and a small
# dedicated pool keeps a slow resolver from starving asyncio's default executor.
RESOLVE_WORKERS = 4

# Fallback when the host has no usable IPv6 stack (see bind_listen_socket).
IPV4_WILDCARD = "0.0.0.0"

# Spellings of the IPv6 wildcard we bind ourselves rather than leaving to
# uvicorn; any other address is unambiguous and asyncio handles it correctly.
IPV6_WILDCARDS = frozenset({"::", "[::]", "::0"})

# Configurable Ollama host. Populated by parse_args(); the env default keeps
# backward compatibility for code paths that import this module directly.
OLLAMA_HOST = os.getenv("OLLAMA_HOST", DEFAULT_OLLAMA_HOST)

# Whether to count requests per source client. Populated by parse_args(); off
# by default because a client address label has unbounded cardinality.
TRACK_CLIENTS = False

# Reverse DNS resolver for the `hostname` label. Populated by main() only when
# --track-clients and --resolve-clients are both set; None means the label
# simply repeats the client address.
CLIENT_RESOLVER = None

# Total deadline applied to inference requests, populated the same way.
REQUEST_TIMEOUT = float(os.getenv("EXPORTER_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT))

# Native endpoints that run a model. Any POST under /v1/ (Ollama's OpenAI
# compatible API) counts as well, see is_inference().
INFERENCE_PATHS = frozenset({"/api/chat", "/api/generate", "/api/embed", "/api/embeddings"})

# Endpoints whose final record carries Ollama's native stats (NDJSON).
NATIVE_METRICS_PATHS = frozenset({"/api/chat", "/api/generate"})
# OpenAI-compatible endpoints whose responses carry a `usage` object.
OPENAI_METRICS_PATHS = frozenset({"/v1/chat/completions", "/v1/completions"})
# Endpoints counted in ollama_requests_total.
COUNTED_PATHS = NATIVE_METRICS_PATHS | OPENAI_METRICS_PATHS

logging.basicConfig()
logger = logging.getLogger(__name__)
LOG_LEVEL = os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper()
logger.setLevel(getattr(logging, LOG_LEVEL, logging.INFO))

app = FastAPI()

OLLAMA_CHAT_REQUEST_COUNT = Counter("ollama_requests_total", "Total chat requests", ["model"])
# Optional: only incremented, for every inference request (see is_inference()),
# when --track-clients is set. A labelled counter
# emits no samples until used, so defining it here costs nothing when disabled.
OLLAMA_CLIENT_REQUEST_COUNT = Counter(
    "ollama_client_requests_total",
    "Inference requests (chat, generate, embeddings, OpenAI-compatible) per source client",
    # `hostname` is always part of the schema (equal to `client` when
    # resolution is off) so the series shape does not depend on a flag.
    ["model", "client", "hostname"],
)
OLLAMA_INFLIGHT = Gauge("ollama_inflight_requests", "Inference requests currently proxied to Ollama", ["model"])

OLLAMA_TOTAL_DURATION =       Histogram("ollama_response_seconds", "Total time spent for the response", ["model"])
OLLAMA_LOAD_DURATION =        Histogram("ollama_load_duration_seconds", "Time spent loading the model", ["model"])
OLLAMA_PROMPT_EVAL_DURATION = Histogram("ollama_prompt_eval_duration_seconds", "Time spent evaluating prompt", ["model"])
OLLAMA_EVAL_DURATION =        Histogram("ollama_eval_duration_seconds", "Time spent generating the response", ["model"])

OLLAMA_PROMPT_EVAL_COUNT = Counter("ollama_tokens_processed_total", "Number of tokens in the prompt", ["model"])
OLLAMA_EVAL_COUNT =        Counter("ollama_tokens_generated_total", "Number of tokens in the response", ["model"])

OLLAMA_TOKENS_PER_SECOND = Histogram(
    "ollama_tokens_per_second",
    "Tokens generated per second",
    ["model"],
    # Use buckets with suitable ranges for tokens/s measurements
    buckets=[5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
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

# Headers uvicorn always adds itself; forwarding Ollama's too would send them
# twice (`date: X, X`, `server: uvicorn, uvicorn`).
SERVER_OWNED_HEADERS = frozenset({"date", "server"})


class UnmapIPv4Middleware:
    """Rewrite IPv4-mapped IPv6 peers (``::ffff:a.b.c.d``) to plain IPv4.

    On the dual-stack ``::`` socket the kernel reports IPv4 peers in mapped
    form. uvicorn's trusted-host check compares that string against the
    configured IPv4 addresses and CIDRs, so it would never match and
    ``X-Forwarded-For`` would be silently ignored behind any IPv4 proxy. This
    middleware must run before :class:`ProxyHeadersMiddleware`.

    Parameters
    ----------
    app : callable
        The wrapped ASGI application.
    """

    def __init__(self, app):
        """Store the wrapped ASGI application."""
        self.app = app

    async def __call__(self, scope, receive, send):
        """Unmap the client address in ``scope``, then delegate to the app.

        Parameters
        ----------
        scope : dict
            ASGI connection scope.
        receive : callable
            ASGI receive channel.
        send : callable
            ASGI send channel.
        """
        client = scope.get("client")
        if client:
            try:
                # Only IPv6Address has ipv4_mapped; plain IPv4 yields None here.
                mapped = getattr(ipaddress.ip_address(client[0]), "ipv4_mapped", None)
            except ValueError:
                # Not an IP literal (e.g. a unix socket path): leave untouched.
                mapped = None
            if mapped is not None:
                scope["client"] = (str(mapped), client[1])
        await self.app(scope, receive, send)


def default_ptr_lookup(ip):
    """Return the reverse DNS name of an IP address.

    Parameters
    ----------
    ip : str
        IP address literal.

    Returns
    -------
    str
        First name returned by ``socket.gethostbyaddr``.

    Raises
    ------
    OSError
        ``socket.herror`` or ``socket.gaierror`` when no name can be found.
    """
    return socket.gethostbyaddr(ip)[0]


class ClientNameResolver:
    """Resolve client IPs to hostnames without ever blocking a request.

    Counting decision: the first request(s) of an unknown client are NOT
    counted under the IP and do NOT wait. The proxied request proceeds
    immediately; the counter increment is deferred until the lookup ends (at
    most ``timeout`` seconds later) and lands directly under the final
    hostname. In the normal case a client's series is therefore born with its
    final hostname, and every request increments exactly one series (no double
    count). A series can still change hostname when the PTR record itself
    changes, or when a client that had no name gains one after
    ``negative_ttl``. A transient failure (timeout, ``TRY_AGAIN``) keeps the
    previously known name, so a DNS outage does not flip an established series
    back to the IP.

    Late results: a lookup that outlives ``timeout`` is not discarded. The
    waiting callbacks are released with the fallback right away, but the
    lookup thread keeps running and, if it eventually succeeds, its name
    replaces the negative cache entry with the positive TTL. A late failure
    changes nothing.

    Known limitation: ``timeout`` includes the time spent queued for one of the
    ``RESOLVE_WORKERS`` lookup threads. If a client's first lookup times out
    (a burst of many new clients, or a hung DNS server tying up the workers),
    only the requests counted while that first lookup was pending land under
    the IP; once the lookup finishes, later requests use the name. A lookup
    that never finishes still leaves the client under its IP for up to
    ``negative_ttl``.

    Parameters
    ----------
    ignore : re.Pattern, optional
        Resolved names matching it (``re.search`` on the normalized name) fall
        back to the IP, e.g. auto-generated reverse names that carry no
        information.
    timeout : float, optional
        Seconds before a lookup is abandoned.
    ttl : float, optional
        Seconds a resolved (or ignored) name stays cached.
    negative_ttl : float, optional
        Seconds a failed lookup stays cached.
    max_size : int, optional
        Maximum number of cached addresses; the least recently used is
        evicted first.
    lookup : callable, optional
        Blocking ``lookup(ip) -> name`` function, replaceable in tests.
    clock : callable, optional
        Monotonic time source, replaceable in tests.
    """

    def __init__(
        self,
        ignore=None,
        timeout=DEFAULT_RESOLVE_TIMEOUT,
        ttl=DEFAULT_RESOLVE_TTL,
        negative_ttl=DEFAULT_RESOLVE_NEGATIVE_TTL,
        max_size=DEFAULT_RESOLVE_MAX_SIZE,
        lookup=default_ptr_lookup,
        clock=time.monotonic,
    ):
        """Store the settings and create the empty cache and worker pool."""
        self.ignore = ignore
        self.timeout = timeout
        self.ttl = ttl
        self.negative_ttl = negative_ttl
        self.max_size = max_size
        self._lookup = lookup
        self._clock = clock
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=RESOLVE_WORKERS, thread_name_prefix="ptr"
        )
        # ip -> (name, expires_at), least recently used first. Expired entries
        # are kept on purpose: they are the "stale" value used on transient
        # errors, and are overwritten on refresh.
        self._cache = OrderedDict()
        # ip -> callbacks waiting for the single lookup running for that ip.
        self._pending = {}
        # ip -> executor future of a lookup that outlived its timeout and is
        # still running. While present, no new lookup is submitted for that ip:
        # it bounds the executor queue to one job per address during a DNS hang
        # and stops an old late result from racing a newer lookup.
        self._late = {}
        # Strong references: the loop only keeps weak ones to running tasks.
        self._tasks = set()

    def with_hostname(self, ip, callback):
        """Call ``callback(hostname)`` exactly once, never blocking.

        The callback runs synchronously for a non-IP input or a fresh cache
        hit, otherwise once the background lookup for ``ip`` has finished.
        Must be called from a running event loop.

        Parameters
        ----------
        ip : str
            Client address, or a placeholder such as ``"unknown"``.
        callback : callable
            Receives the hostname, or ``ip`` itself when none is usable.
        """
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            # Nothing to reverse: do not waste a lookup on "unknown".
            callback(ip)
            return

        entry = self._cache.get(ip)
        if entry is not None and entry[1] > self._clock():
            self._cache.move_to_end(ip)
            callback(entry[0])
            return

        waiting = self._pending.get(ip)
        if waiting is not None:
            # One lookup per address, however many requests arrive meanwhile.
            waiting.append(callback)
            return
        # Created before registering, so a failure here leaves no orphan entry
        # in _pending that would swallow every later callback for this ip.
        task = asyncio.get_running_loop().create_task(self._resolve(ip))
        self._pending[ip] = [callback]
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)

    def close(self):
        """Shut the lookup thread pool down without waiting for its threads.

        A stuck ``gethostbyaddr`` cannot be interrupted, so waiting for it
        would delay process exit (and, on Kubernetes, eat the termination
        grace period). Queued lookups are cancelled; running ones are left to
        finish in the background.
        """
        self._executor.shutdown(wait=False, cancel_futures=True)

    def _normalize(self, name):
        """Strip the root dot and lowercase a DNS name.

        Parameters
        ----------
        name : str
            Name as returned by the resolver.

        Returns
        -------
        str
            Canonical form, so ``Host.Example.`` and ``host.example`` share a
            series.
        """
        return name.rstrip(".").lower()

    def _store(self, ip, name, ttl):
        """Cache ``name`` for ``ip`` and evict the oldest entry past the bound.

        Parameters
        ----------
        ip : str
            Address the name belongs to.
        name : str
            Hostname, or ``ip`` for a negative entry.
        ttl : float
            Seconds the entry stays fresh.
        """
        self._cache[ip] = (name, self._clock() + ttl)
        self._cache.move_to_end(ip)
        while len(self._cache) > self.max_size:
            self._cache.popitem(last=False)

    async def _resolve(self, ip):
        """Look ``ip`` up, cache the outcome and release the waiting callbacks.

        Parameters
        ----------
        ip : str
            Address to resolve.
        """
        # Falls back to the IP unless a branch below settles on a name; the
        # finally block guarantees the callbacks run whatever happens.
        final = ip
        try:
            if ip in self._late:
                # A previous lookup of this ip is still stuck in a thread:
                # serve the usual transient fallback instead of piling up
                # another job behind it.
                final = self._on_transient_failure(ip, "late lookup still running")
                return
            loop = asyncio.get_running_loop()
            lookup_future = loop.run_in_executor(self._executor, self._lookup, ip)
            try:
                # shield: wait_for must only abandon the wait, not cancel the
                # lookup, so its result can still be used if it arrives late.
                raw = await asyncio.wait_for(
                    asyncio.shield(lookup_future), self.timeout
                )
            except (TimeoutError, asyncio.TimeoutError):
                final = self._on_transient_failure(ip, "timed out")
                self._late[ip] = lookup_future
                lookup_future.add_done_callback(
                    lambda fut: self._on_late_result(ip, fut)
                )
            except socket.herror as exc:
                if exc.errno == 2:  # TRY_AGAIN: the resolver is flaky, not definitive
                    final = self._on_transient_failure(ip, f"temporary failure ({exc})")
                else:
                    logger.debug(f"No PTR record for {ip}: {exc}")
                    self._store(ip, ip, self.negative_ttl)
            except OSError as exc:
                logger.debug(f"Reverse lookup of {ip} failed: {exc}")
                self._store(ip, ip, self.negative_ttl)
            else:
                name = self._normalize(raw)
                if self.ignore is not None and self.ignore.search(name):
                    logger.debug(f"Hostname {name} of {ip} matches the ignore pattern")
                    self._store(ip, ip, self.ttl)
                else:
                    logger.debug(f"Resolved {ip} to {name}")
                    final = name
                    self._store(ip, name, self.ttl)
        except Exception:
            logger.exception(f"Unexpected error resolving {ip}")
            self._store(ip, ip, self.negative_ttl)
        finally:
            for callback in self._pending.pop(ip, []):
                try:
                    callback(final)
                except Exception:
                    # One bad callback must not starve the others.
                    logger.exception(f"Client hostname callback failed for {ip}")

    def _on_late_result(self, ip, future):
        """Cache the outcome of a lookup that finished after its timeout.

        Runs on the event loop thread (asyncio future callback). It always
        releases the ``_late`` guard first, whatever the outcome, so the next
        lookup of the address is allowed again. Only a success is used: a failure must not overwrite the stale name or the
        negative entry already in place, and a cancelled lookup (``close()``)
        is ignored.

        Parameters
        ----------
        ip : str
            Address that was being resolved.
        future : asyncio.Future
            The finished lookup future.
        """
        if self._late.get(ip) is future:
            del self._late[ip]
        if future.cancelled():
            return
        # Retrieving the exception also silences "never retrieved" warnings.
        exc = future.exception()
        if exc is not None:
            logger.debug(f"Late reverse lookup of {ip} failed: {exc}")
            return
        try:
            name = self._normalize(future.result())
            if self.ignore is not None and self.ignore.search(name):
                logger.debug(f"Late hostname {name} of {ip} matches the ignore pattern")
                self._store(ip, ip, self.ttl)
            else:
                logger.debug(f"Late lookup resolved {ip} to {name}")
                self._store(ip, name, self.ttl)
        except Exception:
            logger.exception(f"Unexpected error caching late lookup of {ip}")

    def _on_transient_failure(self, ip, reason):
        """Handle a retryable failure, keeping any previously known name.

        Parameters
        ----------
        ip : str
            Address that failed to resolve.
        reason : str
            Short description for the debug log.

        Returns
        -------
        str
            The previous name when one exists (stale on error), else ``ip``.
        """
        previous = self._cache.get(ip)
        name = previous[0] if previous is not None else ip
        logger.debug(f"Reverse lookup of {ip} {reason}, using {name}")
        self._store(ip, name, self.negative_ttl)
        return name


def build_asgi_app(forwarded_allow_ips):
    """Wrap the FastAPI app with the production proxy-header handling.

    Parameters
    ----------
    forwarded_allow_ips : str
        Comma-separated IPs/CIDRs, or ``*``, allowed to set
        ``X-Forwarded-For``.

    Returns
    -------
    callable
        ASGI app: IPv4 unmapping first, then uvicorn's proxy-headers handling.
        Pass it to uvicorn with ``proxy_headers=False`` to avoid a second layer.
    """
    return UnmapIPv4Middleware(
        ProxyHeadersMiddleware(app, trusted_hosts=forwarded_allow_ips)
    )


def sanitize_response_headers(headers):
    """Strip hop-by-hop headers from an upstream response before forwarding.

    ``Content-Length`` is kept when the upstream body has no
    ``Content-Encoding``: httpx then hands the bytes over untouched, so the
    length still matches and clients keep download sizes, HEAD answers and a
    way to detect a truncated body. Uvicorn frames the response with it
    instead of switching to chunked encoding.

    Parameters
    ----------
    headers : httpx.Headers
        Headers as returned by the upstream Ollama response.

    Returns
    -------
    list of tuple of (bytes, bytes)
        Raw ASGI header pairs, lower-cased names, repeated headers preserved,
        minus every entry in :data:`HOP_BY_HOP_HEADERS` and
        :data:`SERVER_OWNED_HEADERS`.
    """
    dropped = HOP_BY_HOP_HEADERS | SERVER_OWNED_HEADERS
    if "content-encoding" not in headers:
        dropped = dropped - {"content-length"}
    # .raw keeps the bytes as received, so no header can fail to re-encode.
    return [(k.lower(), v) for k, v in headers.raw if k.lower().decode("latin-1") not in dropped]


def extract_and_record_metrics(response_data, model):
    """Extract and record metrics from Ollama response data.

    Parameters
    ----------
    response_data : dict
        Final native response record (the one with ``"done": true``), holding
        Ollama's durations in nanoseconds and its token counts.
    model : str
        Model label, taken from the request body.
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
        OLLAMA_TOTAL_DURATION.labels(model=model).observe(total_duration_seconds)
        logger.debug(f"Model: {model}, Total Duration: {total_duration_seconds:.2f} seconds")
    if load_duration > 0:
        load_duration_seconds = load_duration / 1_000_000_000
        OLLAMA_LOAD_DURATION.labels(model=model).observe(load_duration_seconds)
        logger.debug(f"Model: {model}, Load Duration: {load_duration_seconds:.2f} seconds")
    if prompt_eval_duration > 0:
        prompt_eval_time_seconds = prompt_eval_duration / 1_000_000_000
        OLLAMA_PROMPT_EVAL_DURATION.labels(model=model).observe(prompt_eval_time_seconds)
        logger.debug(f"Model: {model}, Prompt Eval Duration: {prompt_eval_time_seconds:.2f} seconds")
    if prompt_eval_count > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_eval_count)
        logger.debug(f"Model: {model}, Prompt Eval Count: {prompt_eval_count}")
    if eval_duration > 0:
        eval_duration_seconds = eval_duration / 1_000_000_000
        OLLAMA_EVAL_DURATION.labels(model=model).observe(eval_duration_seconds)
        logger.debug(f"Model: {model}, Eval Duration: {eval_duration_seconds:.2f} seconds")
    if eval_count > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(eval_count)
        logger.debug(f"Model: {model}, Eval Count: {eval_count}")
    if eval_duration > 0 and eval_count > 0:
        tps = eval_count / eval_duration * 1_000_000_000
        OLLAMA_TOKENS_PER_SECOND.labels(model=model).observe(tps)
        logger.debug(f"Model: {model}, Tokens per Second: {tps:.2f}")

@app.get("/metrics")
def metrics():
    """Expose Prometheus metrics.

    Returns
    -------
    fastapi.Response
        Every registered metric in the Prometheus text format.
    """
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

def record_openai_usage(response_data, model):
    """Record token counts from an OpenAI-compatible (``/v1``) response.

    Ollama's OpenAI layer reports no durations, only a ``usage`` object, so
    only the token counters can be fed from it.

    Parameters
    ----------
    response_data : dict
        Decoded completion object, or the last streamed chunk carrying
        ``usage`` (sent when the client asks for ``stream_options.include_usage``).
    model : str
        Model label, taken from the request body.
    """
    usage = response_data.get("usage") if isinstance(response_data, dict) else None
    if not isinstance(usage, dict):
        return
    prompt_tokens = usage.get("prompt_tokens") or 0
    completion_tokens = usage.get("completion_tokens") or 0
    if prompt_tokens > 0:
        OLLAMA_PROMPT_EVAL_COUNT.labels(model=model).inc(prompt_tokens)
    if completion_tokens > 0:
        OLLAMA_EVAL_COUNT.labels(model=model).inc(completion_tokens)
    logger.debug(f"Model: {model}, OpenAI usage: {prompt_tokens} prompt / {completion_tokens} completion tokens")


class MetricsTap:
    """Watch a relayed response body for the record that carries the stats.

    The body is fed chunk by chunk as it flows to the client. Lines are
    reassembled across chunk boundaries, so a JSON record split by the network
    is still parsed; only the last matching record is kept and recorded once
    the response completes.

    Parameters
    ----------
    path : str
        Request path, which decides the body format: native NDJSON for
        :data:`NATIVE_METRICS_PATHS`, OpenAI JSON or SSE for
        :data:`OPENAI_METRICS_PATHS`, nothing at all for anything else.
    model : str
        Model label to record the metrics under.
    """

    def __init__(self, path, model):
        """Initialise an empty tap for one response (see class docstring)."""
        if path in NATIVE_METRICS_PATHS:
            self.kind = "native"
        elif path in OPENAI_METRICS_PATHS:
            self.kind = "openai"
        else:
            self.kind = None
        self.model = model
        self._buffer = b""
        self._final = None

    def feed(self, chunk):
        """Consume one body chunk.

        Parameters
        ----------
        chunk : bytes
            Decoded body bytes, exactly as forwarded to the client.
        """
        if self.kind is None:
            return
        self._buffer += chunk
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            self._parse(line)

    def finish(self):
        """Parse what is left of the body and record the metrics found."""
        if self.kind is None:
            return
        # A non-streaming body is usually a single line with no trailing newline.
        self._parse(self._buffer)
        self._buffer = b""
        if self._final is None:
            return
        if self.kind == "native":
            extract_and_record_metrics(self._final, self.model)
        else:
            record_openai_usage(self._final, self.model)

    def _parse(self, line):
        """Remember ``line`` if it is the record holding the final stats.

        Parameters
        ----------
        line : bytes
            One body line, an NDJSON record or an SSE ``data:`` field.
        """
        line = line.strip()
        if self.kind == "openai" and line.startswith(b"data:"):
            line = line[len(b"data:"):].strip()
        if not line or line == b"[DONE]":
            return
        try:
            data = json.loads(line)
        except ValueError:  # JSONDecodeError and UnicodeDecodeError alike
            return
        if not isinstance(data, dict):
            return
        if self.kind == "native" and data.get("done"):
            self._final = data
        elif self.kind == "openai" and isinstance(data.get("usage"), dict):
            self._final = data


def is_inference(method, path):
    """Tell whether a request makes Ollama run a model.

    Parameters
    ----------
    method : str
        HTTP method of the request.
    path : str
        Request path, with its leading slash.

    Returns
    -------
    bool
        ``True`` for POSTs to the native inference endpoints and to any
        OpenAI-compatible ``/v1`` endpoint. Those get the in-flight gauge and
        the total deadline; model pulls, pushes and listings do not.
    """
    return method == "POST" and (path in INFERENCE_PATHS or path.startswith("/v1/"))


def model_from_body(body):
    """Pull the ``model`` field out of a raw JSON request body.

    Parameters
    ----------
    body : bytes
        Request body as received from the client.

    Returns
    -------
    str
        The model name, or ``"unknown"`` when the body is not a JSON object
        naming one (Ollama will reject such a request itself).
    """
    try:
        data = json.loads(body) if body else None
    except ValueError:
        return "unknown"
    model = data.get("model") if isinstance(data, dict) else None
    return model if isinstance(model, str) and model else "unknown"


class UpstreamProxyResponse(Response):
    """ASGI response that relays one request to Ollama as a live stream.

    Every proxied request goes through this class, streaming or not. It owns
    the whole upstream exchange so that its lifetime is bound to the client's:

    - the body is forwarded chunk by chunk as Ollama produces it, nothing is
      buffered to the end;
    - a task listens for ``http.disconnect`` the whole time, including while
      Ollama has not answered yet (a non-streaming generation sends nothing
      until it is over, so a failed write would never reveal the hang-up);
      on disconnect the upstream request is cancelled and its connection
      closed, which is what makes Ollama stop generating;
    - an optional total deadline bounds the exchange end to end, unlike
      httpx's read timeout which a steadily streaming generation never hits.

    Parameters
    ----------
    method : str
        HTTP method to use upstream.
    url : str
        Full upstream URL.
    headers : dict of str to str
        Request headers to forward.
    params : starlette.datastructures.QueryParams
        Query string to forward.
    body : bytes
        Request body to forward.
    model : str
        Model label for metrics.
    deadline : float or None
        Seconds after which the exchange is aborted, ``None`` for no limit.
    track_inflight : bool
        Whether this request counts in ``ollama_inflight_requests``.
    """

    def __init__(self, method, url, headers, params, body, model, deadline, track_inflight):
        """Store the upstream request to perform (see class docstring)."""
        super().__init__()
        self.method = method
        self.url = url
        self.forward_headers = headers
        self.params = params
        self.forward_body = body
        self.model = model
        self.deadline = deadline
        self.track_inflight = track_inflight
        self.tap = MetricsTap(urlsplit(url).path, model)
        # Set right before the last body message: from then on, uvicorn's
        # receive() answers http.disconnect for a completed response, which
        # is not the client hanging up.
        self.finished = False

    async def __call__(self, scope, receive, send):
        """Run the upstream exchange until it ends or the client leaves.

        Parameters
        ----------
        scope : dict
            ASGI connection scope.
        receive : callable
            ASGI receive channel; the request body was already consumed, so
            the only message left to come is ``http.disconnect``.
        send : callable
            ASGI send channel.
        """
        if self.track_inflight:
            OLLAMA_INFLIGHT.labels(model=self.model).inc()
        try:
            async with anyio.create_task_group() as task_group:

                async def watch_disconnect():
                    """Cancel the exchange as soon as the client hangs up."""
                    while True:
                        message = await receive()
                        if message["type"] == "http.disconnect":
                            if self.finished:
                                return
                            logger.info(
                                f"Client disconnected, cancelling {self.method} "
                                f"{urlsplit(self.url).path} (model {self.model})"
                            )
                            task_group.cancel_scope.cancel()
                            return

                task_group.start_soon(watch_disconnect)
                await self._relay(send)
                # The exchange is over: stop listening for a disconnect.
                task_group.cancel_scope.cancel()
        finally:
            if self.track_inflight:
                OLLAMA_INFLIGHT.labels(model=self.model).dec()

    async def _relay(self, send):
        """Perform the upstream request and forward its response.

        Parameters
        ----------
        send : callable
            ASGI send channel to the client.
        """
        path = urlsplit(self.url).path
        client = httpx.AsyncClient(timeout=httpx.Timeout(900.0, read=900.0))
        upstream = None
        started = False
        try:
            # fail_after(None) imposes no limit.
            with anyio.fail_after(self.deadline):
                request = client.build_request(
                    self.method, self.url, headers=self.forward_headers,
                    params=self.params, content=self.forward_body,
                )
                upstream = await client.send(request, stream=True)
                await send({
                    "type": "http.response.start",
                    "status": upstream.status_code,
                    "headers": sanitize_response_headers(upstream.headers),
                })
                started = True
                # aiter_bytes() undoes any content-encoding, which is why that
                # header is stripped by sanitize_response_headers().
                async for chunk in upstream.aiter_bytes():
                    self.tap.feed(chunk)
                    await send({"type": "http.response.body", "body": chunk, "more_body": True})
            # Record before the final send: once the response is complete the
            # watcher may wake up and cancel this task.
            if upstream.status_code == 200:
                self.tap.finish()
            self.finished = True
            await send({"type": "http.response.body", "body": b"", "more_body": False})
            logger.debug(f"Proxy response: {upstream.status_code} for {self.method} {path}")
        except TimeoutError:
            logger.warning(
                f"{self.method} {path} (model {self.model}) exceeded the "
                f"{self.deadline:g}s request deadline, aborting it upstream"
            )
            await self._fail(send, started, 504, "request deadline exceeded")
        except httpx.HTTPError as exc:
            logger.error(f"Upstream error on {self.method} {path}: {exc!r}")
            await self._fail(send, started, 502, f"upstream error: {exc}")
        except Exception:
            # Anything else would escape the task group as an ExceptionGroup
            # and reach the client as a bare 500 or a stream ended cleanly.
            logger.exception(f"Proxy failure on {self.method} {path}")
            await self._fail(send, started, 502, "proxy error")
        finally:
            # Closing the connection is what tells Ollama to stop generating.
            # Shielded, because on a client disconnect we run under a
            # cancelled scope where any unshielded await would bail out first.
            with anyio.CancelScope(shield=True):
                if upstream is not None:
                    await upstream.aclose()
                await client.aclose()

    @staticmethod
    async def _fail(send, started, status, message):
        """Report an aborted exchange to the client.

        Parameters
        ----------
        send : callable
            ASGI send channel to the client.
        started : bool
            Whether the response status line was already sent. If so, the
            status cannot change any more, so the body is deliberately left
            unterminated: returning without the final message makes uvicorn
            drop the connection, and the client sees a truncated response
            instead of a 200 that looks complete.
        status : int
            HTTP status to answer with when nothing was sent yet.
        message : str
            Error text, returned in Ollama's ``{"error": ...}`` shape.
        """
        if not started:
            payload = json.dumps({"error": message}).encode()
            await send({
                "type": "http.response.start",
                "status": status,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode()),
                ],
            })
            await send({"type": "http.response.body", "body": payload})


@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE", "PATCH", "HEAD", "OPTIONS"])
async def proxy(request: Request, path: str):
    """Relay any request to Ollama, recording metrics on the way.

    Parameters
    ----------
    request : fastapi.Request
        Incoming client request.
    path : str
        Request path without its leading slash.

    Returns
    -------
    UpstreamProxyResponse
        Response that performs the upstream exchange when served.
    """
    endpoint = f"/{path}"
    body = await request.body()
    inference = is_inference(request.method, endpoint)
    model = model_from_body(body) if request.method == "POST" else "unknown"
    logger.debug(f"Proxying {request.method} request to {endpoint}")

    headers = dict(request.headers)
    headers.pop("host", None)
    headers.pop("content-length", None)

    if request.method == "POST" and endpoint in COUNTED_PATHS:
        OLLAMA_CHAT_REQUEST_COUNT.labels(model=model).inc()

    if TRACK_CLIENTS and inference:
        # uvicorn's proxy-headers middleware has already replaced
        # request.client with the X-Forwarded-For address when the direct peer
        # is trusted, so no header parsing is needed (or wanted) here.
        client_host = request.client.host if request.client else "unknown"
        if CLIENT_RESOLVER is None:
            OLLAMA_CLIENT_REQUEST_COUNT.labels(
                model=model, client=client_host, hostname=client_host
            ).inc()
        else:
            # The increment may be deferred by up to the lookup timeout so
            # that it lands directly under the final hostname; the request
            # itself is not delayed (see ClientNameResolver).
            def count(hostname):
                """Increment the per-client counter under ``hostname``."""
                OLLAMA_CLIENT_REQUEST_COUNT.labels(
                    model=model, client=client_host, hostname=hostname
                ).inc()

            CLIENT_RESOLVER.with_hostname(client_host, count)

    return UpstreamProxyResponse(
        method=request.method,
        url=f"{OLLAMA_HOST}{endpoint}",
        headers=headers,
        params=request.query_params,
        body=body,
        model=model,
        deadline=REQUEST_TIMEOUT if inference and REQUEST_TIMEOUT > 0 else None,
        track_inflight=inference,
    )

async def verify_ollama_connection():
    """Verify connection to Ollama server at startup.

    Only logs the outcome: the exporter starts either way, so a late Ollama
    does not keep it in a crash loop.
    """
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

def positive_float(value):
    """Parse a CLI/env value as a strictly positive float.

    argparse also applies ``type`` to string defaults, so a bad environment
    variable is rejected the same way as a bad flag.

    Parameters
    ----------
    value : str or float
        Value to convert.

    Returns
    -------
    float
        The parsed number, greater than zero.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` is not a number or is not greater than zero.
    """
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(f"invalid number: {value!r}")
    # NaN fails this comparison too, which is what we want.
    if not number > 0:
        raise argparse.ArgumentTypeError(f"must be greater than 0: {value!r}")
    return number


def regex_or_none(value):
    """Compile a CLI/env regex, mapping an empty string to ``None``.

    argparse also applies ``type`` to string defaults, and an empty pattern
    would match every name, so empty must mean "no filter".

    Parameters
    ----------
    value : str
        Regular expression source.

    Returns
    -------
    re.Pattern or None
        Case-insensitive compiled pattern, or ``None`` when ``value`` is empty.

    Raises
    ------
    argparse.ArgumentTypeError
        If ``value`` is not a valid regular expression.
    """
    if not value:
        return None
    try:
        return re.compile(value, re.IGNORECASE)
    except re.error as exc:
        raise argparse.ArgumentTypeError(f"invalid regular expression {value!r}: {exc}")


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
        ``request_timeout``, ``log_level``, ``track_clients`` (bool),
        ``resolve_clients`` (bool), ``client_hostname_ignore``
        (``re.Pattern`` or ``None``), ``client_hostname_timeout`` (float,
        seconds, > 0) and ``forwarded_allow_ips`` (str) attributes.
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
        "--request-timeout",
        type=float,
        default=float(os.getenv("EXPORTER_REQUEST_TIMEOUT", DEFAULT_REQUEST_TIMEOUT)),
        help="Total deadline in seconds for one inference request, after "
             "which it is aborted upstream; 0 disables it (default: %(default)s).",
    )
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", DEFAULT_LOG_LEVEL).upper(),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        type=str.upper,
        help="Logging verbosity (default: %(default)s).",
    )
    parser.add_argument(
        "--track-clients",
        action="store_true",
        default=os.getenv("EXPORTER_TRACK_CLIENTS", "").strip().lower() in TRUTHY_VALUES,
        help="Count chat/generate requests per source client "
             "(ollama_client_requests_total). Off by default: the client label "
             "has unbounded cardinality.",
    )
    parser.add_argument(
        "--resolve-clients",
        action="store_true",
        default=os.getenv("EXPORTER_RESOLVE_CLIENTS", "").strip().lower() in TRUTHY_VALUES,
        help="Add the reverse DNS name of each client as the hostname label of "
             "ollama_client_requests_total. Requires --track-clients.",
    )
    parser.add_argument(
        "--client-hostname-ignore",
        type=regex_or_none,
        default=os.getenv("EXPORTER_CLIENT_HOSTNAME_IGNORE", ""),
        help="Case-insensitive regex; resolved hostnames matching it (re.search) "
             "fall back to the client IP.",
    )
    parser.add_argument(
        "--client-hostname-timeout",
        type=positive_float,
        default=os.getenv("EXPORTER_CLIENT_HOSTNAME_TIMEOUT", DEFAULT_RESOLVE_TIMEOUT),
        help="Seconds after which a reverse lookup stops delaying the counter "
             "increment; a lookup finishing later still fills the cache "
             "(default: %(default)s).",
    )
    parser.add_argument(
        "--forwarded-allow-ips",
        default=os.getenv("FORWARDED_ALLOW_IPS", DEFAULT_FORWARDED_ALLOW_IPS),
        help="Comma-separated IPs/CIDRs, or '*', allowed to set X-Forwarded-For. "
             "The header is only trusted when the direct peer is in this list; "
             "from anyone else it is ignored (default: %(default)s).",
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
    """Configure runtime from CLI/env, then start the exporter server.

    Sets the module-level ``OLLAMA_HOST``, ``REQUEST_TIMEOUT`` and
    ``TRACK_CLIENTS`` from the parsed arguments before serving, since request
    handlers read them there. ``CLIENT_RESOLVER`` is created only when both
    ``--track-clients`` and ``--resolve-clients`` are set.
    """
    global OLLAMA_HOST, REQUEST_TIMEOUT, TRACK_CLIENTS, CLIENT_RESOLVER
    args = parse_args()

    # CLI/env arguments override the module-level defaults set at import time.
    OLLAMA_HOST = args.ollama_host
    REQUEST_TIMEOUT = args.request_timeout
    TRACK_CLIENTS = args.track_clients
    logger.setLevel(getattr(logging, args.log_level, logging.INFO))
    if args.resolve_clients and not args.track_clients:
        logger.warning("--resolve-clients has no effect without --track-clients")
    elif args.resolve_clients:
        CLIENT_RESOLVER = ClientNameResolver(
            ignore=args.client_hostname_ignore,
            timeout=args.client_hostname_timeout,
        )

    await verify_ollama_connection()
    config = uvicorn.Config(
        build_asgi_app(args.forwarded_allow_ips),
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower(),
        # build_asgi_app already adds the proxy-headers layer (after unmapping
        # IPv4-mapped peers); uvicorn's own would be a redundant second one.
        proxy_headers=False,
    )
    server = uvicorn.Server(config)

    sock = bind_listen_socket(args.host, args.port)
    try:
        if sock is None:
            await server.serve()
        else:
            # serve(sockets=...) skips uvicorn's own bind, so our socket
            # options survive.
            families = "IPv6+IPv4" if sock.family == socket.AF_INET6 else "IPv4"
            logger.info(f"Listening on {sock.getsockname()[0]}:{args.port} ({families})")
            await server.serve(sockets=[sock])
    finally:
        if CLIENT_RESOLVER is not None:
            CLIENT_RESOLVER.close()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # SIGINT (Ctrl-C, or `docker stop` via STOPSIGNAL) is a normal shutdown:
        # uvicorn already closed the server, so exit quietly instead of letting
        # the traceback surface and the process report a failure status.
        logger.info("Shutting down")
