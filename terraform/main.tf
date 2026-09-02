// FMOps AWS infrastructure -- root module.
//
// What this provisions:
//   * S3 bucket for datasets, model artifacts and reports (versioned, encrypted,
//     public access fully blocked)
//   * ECR repositories for the api / training / inference images
//   * IAM roles for SageMaker, the inference task, and GitHub Actions via OIDC
//   * CloudWatch log groups, alarms, an SNS topic and a dashboard
//   * A SageMaker model-package group
//
// What this deliberately does NOT provision:
//   * SageMaker endpoints -- they bill per instance-hour and are created by the
//     deployment pipeline against an approved model version, not by a
//     `terraform apply`. Infrastructure and model rollout are separate
//     lifecycles; conflating them means every model promotion needs a terraform
//     run, and every terraform run risks touching live traffic.
//   * VPC/networking -- deliberately out of scope so this can be applied into
//     an existing account without fighting an existing network design.
//
// Usage:
//   terraform init
//   terraform plan  -var environment=dev
//   terraform apply -var environment=dev
//   terraform destroy -var environment=dev

terraform {
  required_version = ">= 1.6.0"

  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    random = {
      source  = "hashicorp/random"
      version = "~> 3.6"
    }
  }

  // Remote state is strongly recommended for anything shared. Copy
  // backend.tf.example to backend.tf and fill in your bucket/table.
}

provider "aws" {
  region = var.aws_region

  default_tags {
    tags = local.common_tags
  }
}

data "aws_caller_identity" "current" {}
data "aws_region" "current" {}
data "aws_partition" "current" {}

locals {
  name_prefix = "${var.project}-${var.environment}"
  account_id  = data.aws_caller_identity.current.account_id
  partition   = data.aws_partition.current.partition

  // Bucket names are globally unique, so fall back to including the account id
  // rather than requiring the operator to invent one.
  artifact_bucket = (
    var.artifact_bucket_name != ""
    ? var.artifact_bucket_name
    : "${local.name_prefix}-artifacts-${local.account_id}"
  )

  common_tags = merge(
    {
      Project     = var.project
      Environment = var.environment
      ManagedBy   = "terraform"
      Component   = "fmops-platform"
      CostCenter  = var.cost_center
    },
    var.additional_tags
  )
}

// --------------------------------------------------------------------------- //
// Modules
// --------------------------------------------------------------------------- //
module "storage" {
  source = "./modules/storage"

  name_prefix         = local.name_prefix
  bucket_name         = local.artifact_bucket
  enable_versioning   = var.enable_bucket_versioning
  retention_days      = var.artifact_retention_days
  force_destroy       = var.force_destroy_bucket
  ecr_repositories    = var.ecr_repositories
  ecr_image_retention = var.ecr_image_retention_count
  ecr_scan_on_push    = var.ecr_scan_on_push
  tags                = local.common_tags
}

module "observability" {
  source = "./modules/observability"

  name_prefix             = local.name_prefix
  environment             = var.environment
  aws_region              = var.aws_region
  log_retention_days      = var.log_retention_days
  alert_email             = var.alert_email
  latency_threshold_ms    = var.latency_alarm_threshold_ms
  error_rate_threshold    = var.error_rate_alarm_threshold
  drift_threshold         = var.drift_alarm_threshold
  sagemaker_endpoint_name = "${local.name_prefix}-endpoint"
  tags                    = local.common_tags
}

module "compute" {
  source = "./modules/compute"

  name_prefix         = local.name_prefix
  artifact_bucket     = module.storage.artifact_bucket_name
  artifact_bucket_arn = module.storage.artifact_bucket_arn
  ecr_repository_arns = module.storage.ecr_repository_arns
  enable_sagemaker    = var.enable_sagemaker
  alerts_topic_arn    = module.observability.alerts_topic_arn
  log_group_arns      = module.observability.log_group_arns
  aws_region          = var.aws_region
  account_id          = local.account_id
  partition           = local.partition
  tags                = local.common_tags
}

// --------------------------------------------------------------------------- //
// ECS Fargate service (optional; this is what serves the public URL)
// --------------------------------------------------------------------------- //
// FMOPS_ENV is "production" so the strict approval gates, JSON logging and
// audit log all apply. The overrides below point the pluggable backends at
// what is actually deployed: there is no SageMaker endpoint, no MLflow
// tracking server and no Bedrock access in this stack, and claiming otherwise
// by leaving the profile defaults in place would give a service that fails to
// start rather than one that is honest about its shape.
module "ecs" {
  count  = var.enable_ecs_service ? 1 : 0
  source = "./modules/ecs"

  name_prefix        = local.name_prefix
  aws_region         = var.aws_region
  container_image    = var.container_image
  task_cpu           = var.ecs_task_cpu
  task_memory        = var.ecs_task_memory
  desired_count      = var.ecs_desired_count
  ingress_cidrs      = var.ecs_ingress_cidrs
  log_retention_days = var.log_retention_days
  execution_role_arn = module.compute.task_execution_role_arn
  task_role_arn      = module.compute.inference_task_role_arn

  container_environment = merge(
    {
      FMOPS_ENV        = "production"
      FMOPS_LOG_FORMAT = "json"

      # No SageMaker endpoint in this stack: route in process.
      FMOPS_DEPLOYMENT__PROVIDER = "local"

      # No MLflow tracking server in this stack: use the embedded SQLite store.
      FMOPS_TRACKING__BACKEND          = "local"
      FMOPS_TRACKING__REGISTRY_BACKEND = "local"

      # Deterministic offline provider. Not a language model, and no Bedrock
      # model access has been requested for this account.
      FMOPS_LLM__PROVIDER = "mock"
      FMOPS_LLM__MODEL    = "mock-small"

      # One worker: the task has half a vCPU, and four would thrash.
      FMOPS_SERVER__WORKERS      = "1"
      FMOPS_SERVER__DOCS_ENABLED = "true"

      # Custom CloudWatch metrics bill per metric per month. Container logs
      # still ship via the awslogs driver.
      FMOPS_MONITORING__CLOUDWATCH_ENABLED = "false"

      FMOPS_AWS__ENABLED   = "true"
      FMOPS_AWS__REGION    = var.aws_region
      FMOPS_AWS__S3_BUCKET = module.storage.artifact_bucket_name
    },
    var.ecs_extra_environment
  )

  tags = local.common_tags
}

// --------------------------------------------------------------------------- //
// GitHub Actions OIDC role
// --------------------------------------------------------------------------- //
// Short-lived credentials only. No IAM user, no access key, nothing to leak.
data "aws_iam_policy_document" "github_assume_role" {
  count = var.github_repository != "" ? 1 : 0

  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRoleWithWebIdentity"]

    principals {
      type = "Federated"
      identifiers = [
        var.github_oidc_provider_arn != ""
        ? var.github_oidc_provider_arn
        : aws_iam_openid_connect_provider.github[0].arn
      ]
    }

    condition {
      test     = "StringEquals"
      variable = "token.actions.githubusercontent.com:aud"
      values   = ["sts.amazonaws.com"]
    }

    // Scope to this repository only. Without this condition ANY GitHub repo
    // could assume the role.
    condition {
      test     = "StringLike"
      variable = "token.actions.githubusercontent.com:sub"
      values   = ["repo:${var.github_repository}:*"]
    }
  }
}

resource "aws_iam_openid_connect_provider" "github" {
  count = var.github_repository != "" && var.github_oidc_provider_arn == "" ? 1 : 0

  url            = "https://token.actions.githubusercontent.com"
  client_id_list = ["sts.amazonaws.com"]
  // GitHub's OIDC endpoint uses a well-known root CA; AWS validates the chain.
  thumbprint_list = ["6938fd4d98bab03faadb97b34396831e3780aea1"]

  tags = merge(local.common_tags, { Name = "${local.name_prefix}-github-oidc" })
}

resource "aws_iam_role" "github_actions" {
  count = var.github_repository != "" ? 1 : 0

  name                 = "${local.name_prefix}-github-actions"
  description          = "Assumed by GitHub Actions via OIDC to build, push and deploy."
  assume_role_policy   = data.aws_iam_policy_document.github_assume_role[0].json
  max_session_duration = 3600

  tags = local.common_tags
}

data "aws_iam_policy_document" "github_actions" {
  count = var.github_repository != "" ? 1 : 0

  // Push container images.
  statement {
    sid       = "ECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"]
  }

  statement {
    sid    = "ECRPush"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:CompleteLayerUpload",
      "ecr:InitiateLayerUpload",
      "ecr:PutImage",
      "ecr:UploadLayerPart",
      "ecr:BatchGetImage",
      "ecr:GetDownloadUrlForLayer",
      "ecr:DescribeImages",
    ]
    resources = module.storage.ecr_repository_arns
  }

  // Read/write model artifacts and datasets.
  statement {
    sid    = "ArtifactAccess"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:PutObject",
      "s3:DeleteObject",
      "s3:ListBucket",
      "s3:GetBucketLocation",
    ]
    resources = [
      module.storage.artifact_bucket_arn,
      "${module.storage.artifact_bucket_arn}/*",
    ]
  }

  // Drive SageMaker training, tuning and endpoint updates.
  statement {
    sid    = "SageMakerDeploy"
    effect = "Allow"
    actions = [
      "sagemaker:CreateModel",
      "sagemaker:CreateEndpointConfig",
      "sagemaker:CreateEndpoint",
      "sagemaker:UpdateEndpoint",
      "sagemaker:UpdateEndpointWeightsAndCapacities",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeEndpointConfig",
      "sagemaker:DescribeModel",
      "sagemaker:DeleteEndpointConfig",
      "sagemaker:CreateTrainingJob",
      "sagemaker:DescribeTrainingJob",
      "sagemaker:CreateHyperParameterTuningJob",
      "sagemaker:DescribeHyperParameterTuningJob",
      "sagemaker:ListTrainingJobsForHyperParameterTuningJob",
      "sagemaker:AddTags",
      "sagemaker:ListTags",
      "sagemaker:CreateModelPackage",
      "sagemaker:UpdateModelPackage",
      "sagemaker:DescribeModelPackage",
      "sagemaker:ListModelPackages",
    ]
    resources = ["arn:${local.partition}:sagemaker:${var.aws_region}:${local.account_id}:*"]
  }

  // Deleting an endpoint is destructive; keep it out of the CI role so a
  // runaway workflow cannot take production offline.

  // Pass the SageMaker execution role, restricted to SageMaker itself.
  dynamic "statement" {
    for_each = var.enable_sagemaker ? [1] : []
    content {
      sid       = "PassSageMakerRole"
      effect    = "Allow"
      actions   = ["iam:PassRole"]
      resources = [module.compute.sagemaker_role_arn]

      condition {
        test     = "StringEquals"
        variable = "iam:PassedToService"
        values   = ["sagemaker.amazonaws.com"]
      }
    }
  }

  statement {
    sid    = "Observability"
    effect = "Allow"
    actions = [
      "cloudwatch:PutMetricData",
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
    ]
    resources = ["*"]
  }
}

resource "aws_iam_role_policy" "github_actions" {
  count = var.github_repository != "" ? 1 : 0

  name   = "${local.name_prefix}-github-actions"
  role   = aws_iam_role.github_actions[0].id
  policy = data.aws_iam_policy_document.github_actions[0].json
}

// --------------------------------------------------------------------------- //
// SageMaker model package group
// --------------------------------------------------------------------------- //
// The AWS-native counterpart to the platform's model registry. Kept so model
// versions promoted by FMOps are also visible in SageMaker Studio.
resource "aws_sagemaker_model_package_group" "main" {
  count = var.enable_sagemaker ? 1 : 0

  model_package_group_name        = "${local.name_prefix}-models"
  model_package_group_description = "Registered FMOps model versions for ${var.environment}."

  tags = local.common_tags
}
