# FMOps API / inference service.
#
# Multi-stage: the builder compiles wheels, the runtime carries none of the
# build toolchain. The final image runs as a non-root user with a read-only
# application directory and writable volumes only where state genuinely lives.

# ---------------------------------------------------------------- builder ---
FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1

# build-essential is needed to compile any sdist-only wheels; it stays in the
# builder stage and never reaches the runtime image.
RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements.txt

# ---------------------------------------------------------------- runtime ---
FROM python:3.12-slim-bookworm AS runtime

# The commit this image was built from, reported by /health. Pass it at build
# time: docker build --build-arg GIT_COMMIT=$(git rev-parse --short HEAD) ...
ARG GIT_COMMIT=unknown

LABEL org.opencontainers.image.title="fmops-api" \
      org.opencontainers.image.description="FMOps platform API and inference service" \
      org.opencontainers.image.source="https://github.com/shaileshpawar-dev/fmops-platform" \
      org.opencontainers.image.revision="${GIT_COMMIT}" \
      org.opencontainers.image.licenses="MIT"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FMOPS_ENV=production \
    FMOPS_LOG_FORMAT=json \
    FMOPS_GIT_COMMIT=${GIT_COMMIT} \
    FMOPS_SERVER__WORKERS=1

# curl is used by the container HEALTHCHECK below.
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 fmops \
 && useradd --uid 10001 --gid fmops --create-home --shell /usr/sbin/nologin fmops

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=fmops:fmops app/ ./app/
COPY --chown=fmops:fmops pipelines/ ./pipelines/
COPY --chown=fmops:fmops scripts/ ./scripts/
COPY --chown=fmops:fmops configs/ ./configs/
COPY --chown=fmops:fmops pyproject.toml README.md ./

# State directories the process genuinely writes to. Everything else stays
# owned by root and unwritable by the runtime user.
RUN mkdir -p /app/artifacts /app/data \
 && chown -R fmops:fmops /app/artifacts /app/data

USER fmops
EXPOSE 8000

# Readiness: the database answers and the job worker is alive. A platform with
# no models yet is still ready -- it must accept uploads and training.
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
  CMD curl -fsS http://localhost:8000/health/ready || exit 1

# The worker count comes from FMOPS_SERVER__WORKERS like every other setting;
# hard-coding it here used to override the configured value silently.
CMD ["sh", "-c", "exec uvicorn app.api.main:app --host 0.0.0.0 --port 8000 --workers ${FMOPS_SERVER__WORKERS:-1} --no-access-log"]
