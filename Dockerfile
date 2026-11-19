# syntax=docker/dockerfile:1

###############################################################################
# Builder stage — install dependencies into an isolated virtualenv so the
# runtime image carries only what it needs (no pip cache, no build cruft).
###############################################################################
FROM python:3.11-slim AS builder

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Self-contained virtualenv we can copy wholesale into the runtime stage.
RUN python -m venv "$VIRTUAL_ENV"

# Install deps first, in their own layer: it stays cached until
# requirements.txt changes, so code edits don't trigger a reinstall.
COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip \
    pip install -r requirements.txt

# pip is not needed at runtime, and its vendored certifi bundle still ships
# roots that were distrusted upstream (emSign ECC Root CA G3, Comodo AAA).
# Drop it from the venv before the venv is copied into the runtime stage.
RUN pip uninstall -y pip

###############################################################################
# Runtime stage — minimal Debian slim image, non-root, app + venv only.
###############################################################################
FROM python:3.11-slim AS runtime

# OCI metadata (image source, description, license).
LABEL org.opencontainers.image.source="https://github.com/frcooper/ollama-exporter" \
      org.opencontainers.image.description="Prometheus exporter and transparent proxy for Ollama" \
      org.opencontainers.image.licenses="Unlicense"

ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH" \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OLLAMA_HOST="http://localhost:11434"

WORKDIR /app

# Bring in the pre-built virtualenv from the builder stage.
COPY --from=builder /opt/venv /opt/venv

# Unprivileged runtime user; high UID/GID avoids collisions with host users.
# The base image's own pip goes too, for the same certifi reason as above.
# Called through the system interpreter, since the venv comes first in PATH.
RUN groupadd -r -g 10001 app && useradd -r -u 10001 -g app app \
    && /usr/local/bin/python -m pip uninstall -y pip

# Keep the two distrusted roots out of the system bundle even if the floating
# base tag ever ships a ca-certificates that still has them: match them by
# SHA-256 fingerprint (names change between releases), deselect and delete
# their files, and rebuild the bundle. A no-op on a current base.
RUN <<'EOF'
set -eu
/usr/local/bin/python - <<'PY'
import hashlib, pathlib, ssl
BAD = {
    "d7a7a0fb5d7e2731d771e9484ebcdef71d5f0c3e0a2948782bc83ee0ea699ef4",
    "86a1ecba089c4a8d3bbe2734c612ba341d813e043cf9e8a862cd5c57a36bbe6b",
}
root = pathlib.Path("/usr/share/ca-certificates")
drop = {
    str(p.relative_to(root)) for p in root.rglob("*.crt")
    if hashlib.sha256(ssl.PEM_cert_to_DER_cert(p.read_text())).hexdigest() in BAD
}
conf = pathlib.Path("/etc/ca-certificates.conf")
conf.write_text("".join(
    f"!{line}" if line.strip() in drop else line
    for line in conf.read_text().splitlines(keepends=True)
))
# Delete the files too, so a scan of the image does not flag them.
for name in drop:
    (root / name).unlink()
print("deselected:", sorted(drop) or "none")
PY
update-ca-certificates --fresh > /dev/null
EOF

# Application code, owned by the unprivileged user.
COPY --chown=app:app ollama_exporter.py .

USER app

EXPOSE 8000

# /metrics is served locally; every other path is proxied to Ollama, so it is
# the only endpoint that reflects this process being healthy on its own.
# Uses python (always present) rather than curl/wget (absent from slim).
# Reads EXPORTER_PORT so the probe follows a port override; a port set only
# via --port on the command line still needs this env var to stay in sync.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
  CMD python -c "import os,urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:%s/metrics' % os.getenv('EXPORTER_PORT', '8000'), timeout=2).status == 200 else 1)"

# uvicorn shuts down gracefully on SIGINT.
STOPSIGNAL SIGINT

# Go through the module's own CLI instead of calling uvicorn directly: it is
# what reads the environment, applies the argparse defaults and checks the
# upstream Ollama connection before serving. Splitting it as ENTRYPOINT/CMD
# lets `docker run <image> --port 9000 --log-level DEBUG` reach argparse.
ENTRYPOINT ["python", "ollama_exporter.py"]
CMD []
