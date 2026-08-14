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
RUN groupadd -r -g 10001 app && useradd -r -u 10001 -g app app

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
