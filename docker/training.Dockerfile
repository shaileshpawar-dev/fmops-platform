# FMOps training image.
#
# Runs the training / evaluation / retraining pipelines. This is also the image
# referenced by FMOPS_AWS__SAGEMAKER_TRAINING_IMAGE when training on SageMaker:
# SageMaker invokes the container with the argument `train`, which the
# entrypoint below maps onto the training pipeline.

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

LABEL org.opencontainers.image.title="fmops-training" \
      org.opencontainers.image.description="FMOps training and retraining pipelines"

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    FMOPS_LOG_FORMAT=json

RUN groupadd --gid 10001 fmops \
 && useradd --uid 10001 --gid fmops --create-home --shell /usr/sbin/nologin fmops

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=fmops:fmops app/ ./app/
COPY --chown=fmops:fmops pipelines/ ./pipelines/
COPY --chown=fmops:fmops scripts/ ./scripts/
COPY --chown=fmops:fmops configs/ ./configs/
COPY --chown=fmops:fmops docker/entrypoint-training.sh /usr/local/bin/entrypoint-training.sh
COPY --chown=fmops:fmops pyproject.toml README.md ./

RUN chmod +x /usr/local/bin/entrypoint-training.sh \
 && mkdir -p /app/artifacts /app/data /opt/ml \
 && chown -R fmops:fmops /app/artifacts /app/data /opt/ml

USER fmops

ENTRYPOINT ["/usr/local/bin/entrypoint-training.sh"]
CMD ["train"]
