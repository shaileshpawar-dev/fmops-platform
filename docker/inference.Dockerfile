# Slim inference-only image.
#
# Differs from api.Dockerfile deliberately: it carries only what the scoring
# path needs, and it is the image deployed behind a SageMaker endpoint (which
# invokes the container with `serve` and health-checks /ping).
#
# Keeping this separate from the full API image means the thing exposed to
# production traffic does not ship the training, tuning or retraining code
# paths at all -- smaller attack surface, faster cold start.

FROM python:3.12-slim-bookworm AS builder

ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
 && apt-get install -y --no-install-recommends build-essential gcc \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /build
COPY requirements.txt ./
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip setuptools wheel \
 && /opt/venv/bin/pip install -r requirements.txt

FROM python:3.12-slim-bookworm AS runtime

LABEL org.opencontainers.image.title="fmops-inference" \
      org.opencontainers.image.description="FMOps real-time inference endpoint"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FMOPS_ENV=production \
    FMOPS_LOG_FORMAT=json \
    FMOPS_SERVER__DOCS_ENABLED=false \
    FMOPS_TUNING__ENABLED=false

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && groupadd --gid 10001 fmops \
 && useradd --uid 10001 --gid fmops --create-home --shell /usr/sbin/nologin fmops

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=fmops:fmops app/ ./app/
COPY --chown=fmops:fmops configs/ ./configs/
COPY --chown=fmops:fmops pyproject.toml README.md ./

RUN mkdir -p /app/artifacts /app/data /opt/ml/model \
 && chown -R fmops:fmops /app/artifacts /app/data /opt/ml

USER fmops
EXPOSE 8080

HEALTHCHECK --interval=20s --timeout=5s --start-period=30s --retries=3 \
  CMD curl -fsS http://localhost:8080/health/ready || exit 1

# Port 8080 and the `serve` convention match what SageMaker expects from a
# bring-your-own inference container.
CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2", "--no-access-log"]
