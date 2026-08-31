# FMOps infrastructure

Terraform for the AWS backend. The platform runs fully locally without any of
this; applying it turns on the production backends (S3 artifacts, SageMaker,
CloudWatch, Bedrock, SNS alerts).

## What gets created

| Resource | Purpose | Cost when idle |
|---|---|---|
| S3 bucket | datasets, model artifacts, reports | storage only (pennies) |
| ECR repositories x3 | api / training / inference images | storage only |
| IAM roles | SageMaker execution, inference task, ECS execution, GitHub OIDC | free |
| CloudWatch log groups x3 | api / training / pipeline | ingestion + storage |
| CloudWatch alarms x5 | drift, rollback, errors, latency, 5xx | ~$0.10/alarm/month |
| CloudWatch dashboard | one-glance overview | first 3 free |
| SNS topic | alert fan-out | per-message |
| SageMaker model package group | AWS-native model registry | free |

**No SageMaker endpoint is created.** Endpoints bill per instance-hour
(~$70/month for a single `ml.m5.large`), so they are created by the deployment
pipeline against an approved model version, not by `terraform apply`.
Infrastructure and model rollout are separate lifecycles on purpose: otherwise
every model promotion would need a terraform run, and every terraform run would
risk touching live traffic.

## Usage

```bash
cd terraform
cp terraform.tfvars.example terraform.tfvars   # then edit
cp backend.tf.example backend.tf               # optional but recommended

terraform init
terraform plan  -var environment=dev
terraform apply -var environment=dev
```

Wire the platform to what was created:

```bash
eval "$(terraform output -raw fmops_env_exports)"
fmops aws status
```

Tear down:

```bash
terraform destroy -var environment=dev
```

`destroy` will refuse to delete a non-empty artifact bucket unless
`force_destroy_bucket = true`. That is deliberate — it is the guard against
deleting the only copy of a production model.

## Security posture

- **No IAM users, no access keys.** CI authenticates with GitHub OIDC and
  short-lived credentials, scoped by `sub` to a single repository.
- **Least privilege.** The inference role can `GetObject` under `models/` and
  nothing else; it has no `PutObject`, so a compromised serving container
  cannot tamper with the registry. Wildcards appear only where the AWS API
  requires them (`ecr:GetAuthorizationToken`, `cloudwatch:PutMetricData`, the
  latter constrained by a namespace condition).
- **Deleting an endpoint is not in the CI role**, so a runaway workflow cannot
  take production offline.
- **S3**: public access blocked at every level, SSE-AES256 enforced, versioning
  on, and a bucket policy that denies non-TLS requests.
- **ECR**: immutable tags, scan-on-push.
- **No secrets in this configuration.** AWS credentials come from your
  credential chain; application secrets (`ANTHROPIC_API_KEY`, etc.) are supplied
  at runtime.

## Environments

Apply the same configuration per environment with a different `-var
environment=`, ideally with separate state keys and separate AWS accounts:

```bash
terraform workspace new staging
terraform apply -var environment=staging
```

## Verification status

The HCL in this directory is syntax-checked in CI (`terraform fmt -check` and
`terraform validate`). It has **not** been applied against a live AWS account as
part of this repository's automated tests, because that would create billable
resources. Treat a first `apply` in your own account as the real validation, and
read the plan output before confirming.
