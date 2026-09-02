# Deployment image for the public demo.
#
# The API image is stateless: a fresh container has an empty registry, so
# /health/ready correctly reports not_ready and /predict has nothing to serve.
# That is right for a real deployment, where a trained model arrives from the
# training pipeline through S3 and the registry. It is wrong for a portfolio
# demo, where the URL must show a working prediction the moment it is opened.
#
# So this image bakes in a seed: the training image generates a dataset,
# validates it, trains, evaluates against the approval gate, registers and
# promotes a model, then deploys it. The resulting artifacts and SQLite state
# are copied into the API image.
#
# The seed is real output from the real pipeline -- not a fixture. What it is
# not is production state: the database is baked into the image, so anything
# written at runtime (predictions, drift scans, feedback) lives only for the
# lifetime of that task.
#
#   docker build -f docker/demo-seed.Dockerfile \
#     --build-arg API_IMAGE=fmops/api:local \
#     --build-arg TRAINING_IMAGE=fmops/training:local \
#     -t fmops/api:demo .

ARG API_IMAGE=fmops/api:local
ARG TRAINING_IMAGE=fmops/training:local

FROM ${TRAINING_IMAGE} AS seed

WORKDIR /app

# Local everything: the seed runs at build time with no AWS credentials and no
# network access to any service. Explicit rather than relying on defaults.
ENV FMOPS_ENV=development \
    FMOPS_AWS__ENABLED=false \
    FMOPS_TRACKING__BACKEND=local \
    FMOPS_TRACKING__REGISTRY_BACKEND=local \
    FMOPS_LLM__PROVIDER=mock \
    FMOPS_LOG_FORMAT=console

RUN python -m app.cli data generate \
 && python -m app.cli data validate \
 && python -m pipelines.training_pipeline --promote \
 && python -m pipelines.deployment_pipeline --strategy direct \
 && python -m app.cli status

FROM ${API_IMAGE}

COPY --from=seed --chown=fmops:fmops /app/artifacts /app/artifacts
COPY --from=seed --chown=fmops:fmops /app/data /app/data
