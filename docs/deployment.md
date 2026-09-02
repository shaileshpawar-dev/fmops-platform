# Deployment

## Local (no Docker)

```bash
make setup
make demo          # the whole lifecycle, ~5 minutes
make serve         # then open http://localhost:8000/dashboard
```

Everything runs on the filesystem, SQLite and a local MLflow store. No AWS, no
API keys, no network.

## Local stack (Docker Compose)

```bash
docker compose up -d
docker compose --profile bootstrap up bootstrap    # data + train + promote + deploy
```

| Service | URL | Purpose |
|---|---|---|
| API + dashboard | http://localhost:8000/dashboard | the platform |
| API docs | http://localhost:8000/docs | OpenAPI |
| MLflow | http://localhost:5000 | experiments and registry |
| Prometheus | http://localhost:9090 | metrics + alert rules |
| Grafana | http://localhost:3000 | dashboards (admin/admin) |
| MinIO | http://localhost:9001 | S3-compatible, `--profile s3` |

MinIO is worth calling out: it lets the `S3ArtifactStore` code path be genuinely
exercised locally rather than only asserted about.

```bash
docker compose --profile s3 up -d minio
export FMOPS_AWS__ENABLED=true \
       FMOPS_AWS__S3_BUCKET=fmops \
       AWS_ACCESS_KEY_ID=fmopsadmin \
       AWS_SECRET_ACCESS_KEY=fmopsadmin123 \
       AWS_ENDPOINT_URL=http://localhost:9000
```

## Container images

Three images, deliberately separate:

| Image | Contains | Runs as | Port |
|---|---|---|---|
| `api` | full platform: serving, training, monitoring, LLMOps | uid 10001 | 8000 |
| `training` | pipelines only; also the SageMaker training image | uid 10001 | — |
| `inference` | scoring path only | uid 10001 | 8080 |

`inference` is separate on purpose: the thing exposed to production traffic does
not ship the training, tuning or retraining code paths at all. Smaller attack
surface, faster cold start.

### Image tags

**Images are referenced by commit SHA. There is no `:latest` tag, by design.**

The ECR repositories are created with `IMMUTABLE` tag mutability, so a tag that
has been pushed cannot be repointed at a new image. A moving tag therefore
cannot be republished at all -- a second push is rejected by the registry and
fails the pipeline.

That is the mechanical reason. The operational one matters more: a moving tag
makes the running revision ambiguous. `:latest` answers "what is deployed?"
with "whatever was pushed most recently", which is not an answer you can roll
back to. A commit SHA is.

So:

- `cd.yml` publishes exactly one tag per image, `sha-<short commit>`, and every
  consumer -- the trivy scan, the ECS deploy, the SageMaker training image --
  references it.
- The Terraform `container_image` variable rejects `:latest` and other moving
  tags with a validation rule, so a mutable reference cannot reach a task
  definition even by hand.
- Rolling back means pointing `container_image` at an earlier SHA and applying.
  Every previously deployed image stays pullable by its own SHA tag, subject to
  the ECR lifecycle policy (`ecr_image_retention_count`, default 5).

All three are multi-stage (no build toolchain in the runtime layer), run as a
non-root user, and declare a `HEALTHCHECK` against `/health/ready` — **readiness,
not liveness**, so an orchestrator does not route traffic to a container that has
not loaded a model yet.

## Deployment strategies

```bash
fmops deploy --version 3 --strategy blue_green
fmops deploy --version 3 --strategy canary
fmops deploy --version 3 --strategy shadow
fmops rollback
```

### Blue/green

Green is brought up **at zero traffic** and health-checked (which loads the
artifact, so a version that cannot load fails before any traffic moves), then
traffic cuts over in one step. A failed post-cutover check returns traffic to
blue immediately.

Trade-off: the cutover is instant, so a defect that only appears under real
traffic hits 100% of requests for the duration of one health check. In exchange
there is no period of split-version traffic — which matters when two versions
writing to the same downstream store would be inconsistent.

### Canary

```yaml
deployment:
  canary_steps: [10, 25, 50, 100]
  canary_step_seconds: 300
  canary_min_requests_per_step: 200
  canary_max_error_rate: 0.02
  canary_max_latency_ms: 200.0
```

After each step the strategy reads the **observed** error rate and p95 latency
for the candidate from the persisted inference log, and either advances or aborts.

**The minimum-sample guard matters.** A step that saw fewer than
`canary_min_requests_per_step` requests is recorded as *passed with insufficient
evidence*:

```
[PASS] step 0 @10%: 3 reqs, err=33.33%, p95=41.2ms
       -- only 3 requests observed (minimum 200); advancing without
          statistically meaningful evidence
```

With three requests, one error is 33% and means nothing. Failing the deployment
on that would be superstition; passing it silently would be a lie. Recording it
is the honest option.

### Shadow

The candidate receives mirrored traffic; its predictions are logged and **never
returned**. Zero blast radius.

What shadow mode can tell you: latency under real traffic shapes, error rates on
real inputs, how far the candidate's prediction distribution diverges from the
incumbent's. What it **cannot** tell you: whether the candidate is more accurate
— shadow traffic is unlabelled at scoring time.

Shadow scoring runs inside the request path but is fully isolated: any exception
is logged and swallowed, so a broken candidate cannot degrade production.

> The SageMaker provider **refuses** shadow deployments rather than silently
> serving the shadow version real traffic. SageMaker has native shadow tests;
> wire those up, or run the shadow strategy against the local provider.

## Rollback

```bash
fmops rollback
fmops rollback --to-version 5 --reason "latency regression"
curl -X POST localhost:8000/api/v1/deployments/rollback
```

Target selection is explicit and ordered: explicit `to_version` → the recorded
`previous_version` → the registry's most recently archived ex-production version.
If none exists, rollback **raises** rather than guessing.

Rollback also repairs the registry, demoting the failed version and walking the
restored one back to Production, so the registry never claims a version is live
when it is not.

Automatic rollback fires from three places: a strategy failing its health gate,
a post-deployment smoke test failing, and the CD workflow's failure handler.

## AWS

### 1. Provision

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # edit
terraform init
terraform plan  -var environment=staging
terraform apply -var environment=staging
```

Creates S3, ECR, IAM roles, CloudWatch log groups/alarms/dashboard, SNS, and a
SageMaker model package group. **It does not create a SageMaker endpoint** —
endpoints bill per instance-hour and are created by the deployment pipeline
against an approved model version. Infrastructure and model rollout are separate
lifecycles.

### 2. Point the platform at it

```bash
eval "$(terraform output -raw fmops_env_exports)"
fmops aws status
```

### 3. Push images and deploy

```bash
aws ecr get-login-password | docker login --username AWS --password-stdin "$ECR_REGISTRY"
docker build -f docker/inference.Dockerfile -t "$ECR_REGISTRY/inference:v1" .
docker push "$ECR_REGISTRY/inference:v1"

export FMOPS_DEPLOYMENT__PROVIDER=sagemaker
export FMOPS_AWS__SAGEMAKER_TRAINING_IMAGE="$ECR_REGISTRY/inference:v1"
python -m pipelines.deployment_pipeline --strategy canary
```

Canary on SageMaker maps to production-variant weights, so shifting traffic is an
`UpdateEndpointWeightsAndCapacities` call rather than a reprovision — the same
step sequence, a different mechanism.

### Cost warning

| Resource | Idle cost |
|---|---|
| S3, ECR, IAM, SNS | pennies |
| CloudWatch alarms | ~$0.10/alarm/month |
| **SageMaker endpoint (`ml.m5.large`)** | **~$70/month, billed while it exists** |
| SageMaker training job | per second, only while running |

`terraform destroy` removes the infrastructure but **not** endpoints created by
the deployment pipeline:

```bash
fmops deploy --help          # see terminate
curl -X POST localhost:8000/api/v1/deployments/terminate
```

## Production checklist

**Configuration**
- [ ] `FMOPS_ENV=production` (strict gates, manual approval, canary, docs off)
- [ ] `FMOPS_SECURITY__AUTH_BACKEND=api_key` with generated keys
- [ ] `FMOPS_SECURITY__CORS_ALLOW_ORIGINS` set to real origins, not `["*"]`
- [ ] `FMOPS_LOG_FORMAT=json`
- [ ] Secrets from a secret manager, never a file

**Infrastructure**
- [ ] Terraform state in S3 with DynamoDB locking
- [ ] TLS terminated at the load balancer (the app assumes it is behind one and
      does not terminate TLS itself)
- [ ] Security groups restrict `/metrics` to the Prometheus scraper
- [ ] `force_destroy_bucket = false`

**Before the first real deploy**
- [ ] Replace SQLite with Postgres if running more than one API replica
- [ ] Point MLflow at a tracking server with an RDS backend and S3 artifacts
- [ ] Subscribe a real address to the alerts SNS topic
- [ ] Configure required reviewers on the `production` GitHub Environment
- [ ] Establish a labelling loop — without it, live performance and concept
      drift are permanently unavailable and the `performance` retraining trigger
      never fires

## Security posture

**Implemented**

- No secrets in git; `.gitignore` and a CI check both enforce it
- Every config dump and log line redacts secrets
- API key auth with `secrets.compare_digest`; keys are logged only as an
  8-character hash tag
- Request validation via pydantic with explicit ranges; unknown fields rejected
- Batch size cap
- Non-root containers, multi-stage builds, `no-new-privileges`
- IAM least privilege — the inference role has `GetObject` under `models/` and no
  `PutObject`, so a compromised serving container cannot tamper with the registry
- GitHub OIDC, no long-lived AWS keys; the CI role cannot delete endpoints
- S3: public access blocked, SSE enforced, versioned, non-TLS denied
- ECR immutable tags with scan-on-push
- Audit log for every state-changing action
- Dependency audit (`pip-audit`), static analysis (`bandit`), secret scanning
  (`gitleaks`), container scanning (`trivy`) in CI

**Explicitly not implemented**

- **No user identity or RBAC.** `AuthBackend` is a shared-secret interface with a
  clean seam for OIDC/JWT; there are no users, roles or per-principal scopes.
- **No rate limiting.** Put an API gateway or reverse proxy in front.
- **No TLS termination.** The app assumes a load balancer in front of it.
- **No encryption at rest for the local SQLite database.** On AWS, use RDS with
  encryption enabled.
- **No signed model artifacts.** Artifact integrity relies on S3 versioning and
  bucket policy, not on cryptographic signing.

These are honest gaps, not oversights — each is a deliberate scope boundary, and
each is a real requirement before this handles regulated production traffic.
