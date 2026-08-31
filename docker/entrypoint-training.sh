#!/usr/bin/env bash
# Training container entrypoint.
#
# Understands both the plain FMOps commands and the two argument conventions
# SageMaker uses ("train" and "serve"), so the same image works locally, in CI,
# and as a SageMaker training image.
set -euo pipefail

COMMAND="${1:-train}"
shift || true

# SageMaker mounts channels under /opt/ml/input/data/<channel> and expects
# artifacts written to /opt/ml/model. When those paths exist we are running on
# SageMaker, so point the platform at them.
if [ -d "/opt/ml/input/data/training" ]; then
  echo "detected SageMaker input channel; using /opt/ml paths"
  export FMOPS_PATHS__RAW_DIR="/opt/ml/input/data/training"
  export FMOPS_PATHS__MODELS_DIR="/opt/ml/model"
  export FMOPS_PATHS__ARTIFACTS_DIR="/opt/ml/output"
fi

case "${COMMAND}" in
  train)
    echo "running the training pipeline"
    exec python -m pipelines.training_pipeline "$@"
    ;;
  evaluate)
    exec python -m pipelines.evaluation_pipeline "$@"
    ;;
  deploy)
    exec python -m pipelines.deployment_pipeline "$@"
    ;;
  retrain)
    exec python -m pipelines.retraining_pipeline "$@"
    ;;
  data)
    exec python -m app.cli data generate "$@"
    ;;
  fmops)
    exec python -m app.cli "$@"
    ;;
  serve)
    # SageMaker uses "serve" for inference containers.
    exec uvicorn app.api.main:app --host 0.0.0.0 --port 8080
    ;;
  bash|sh)
    exec /bin/bash "$@"
    ;;
  *)
    echo "unknown command: ${COMMAND}" >&2
    echo "expected one of: train, evaluate, deploy, retrain, data, fmops, serve" >&2
    exit 64
    ;;
esac
