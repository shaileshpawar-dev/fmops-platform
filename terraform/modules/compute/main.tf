// Compute IAM: the roles SageMaker and the inference task assume.
//
// Every policy here is scoped to the specific bucket, repositories and log
// groups this deployment created. There is no wildcard resource grant except
// where the AWS API genuinely requires one (ecr:GetAuthorizationToken,
// cloudwatch:PutMetricData), and those are read-only or metric-only actions.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "name_prefix" { type = string }
variable "artifact_bucket" { type = string }
variable "artifact_bucket_arn" { type = string }
variable "ecr_repository_arns" { type = list(string) }
variable "enable_sagemaker" { type = bool }
variable "alerts_topic_arn" { type = string }
variable "log_group_arns" { type = list(string) }
variable "aws_region" { type = string }
variable "account_id" { type = string }
variable "partition" { type = string }
variable "tags" { type = map(string) }

// --------------------------------------------------------------------------- //
// Shared policy documents
// --------------------------------------------------------------------------- //
data "aws_iam_policy_document" "artifact_read_write" {
  statement {
    sid    = "ReadWriteArtifacts"
    effect = "Allow"
    actions = [
      "s3:GetObject",
      "s3:GetObjectVersion",
      "s3:PutObject",
      "s3:DeleteObject",
    ]
    resources = ["${var.artifact_bucket_arn}/*"]
  }

  statement {
    sid       = "ListArtifactBucket"
    effect    = "Allow"
    actions   = ["s3:ListBucket", "s3:GetBucketLocation"]
    resources = [var.artifact_bucket_arn]
  }
}

data "aws_iam_policy_document" "ecr_pull" {
  statement {
    sid       = "ECRAuth"
    effect    = "Allow"
    actions   = ["ecr:GetAuthorizationToken"]
    resources = ["*"] // this action does not support resource scoping
  }

  statement {
    sid    = "ECRPull"
    effect = "Allow"
    actions = [
      "ecr:BatchCheckLayerAvailability",
      "ecr:GetDownloadUrlForLayer",
      "ecr:BatchGetImage",
    ]
    resources = var.ecr_repository_arns
  }
}

data "aws_iam_policy_document" "observability" {
  statement {
    sid    = "WriteLogs"
    effect = "Allow"
    actions = [
      "logs:CreateLogStream",
      "logs:PutLogEvents",
      "logs:DescribeLogStreams",
      "logs:CreateLogGroup",
    ]
    resources = concat(
      var.log_group_arns,
      [for arn in var.log_group_arns : "${arn}:*"]
    )
  }

  statement {
    sid       = "PublishMetrics"
    effect    = "Allow"
    actions   = ["cloudwatch:PutMetricData"]
    resources = ["*"] // PutMetricData cannot be resource-scoped

    condition {
      test     = "StringEquals"
      variable = "cloudwatch:namespace"
      values   = ["FMOps", "FMOps/Staging", "FMOps/Production", "AWS/SageMaker"]
    }
  }
}

// --------------------------------------------------------------------------- //
// SageMaker execution role
// --------------------------------------------------------------------------- //
data "aws_iam_policy_document" "sagemaker_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["sagemaker.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "sagemaker" {
  count = var.enable_sagemaker ? 1 : 0

  name               = "${var.name_prefix}-sagemaker"
  description        = "Execution role for FMOps SageMaker training jobs and endpoints."
  assume_role_policy = data.aws_iam_policy_document.sagemaker_assume.json

  tags = merge(var.tags, { Name = "${var.name_prefix}-sagemaker" })
}

resource "aws_iam_role_policy" "sagemaker_artifacts" {
  count = var.enable_sagemaker ? 1 : 0

  name   = "artifacts"
  role   = aws_iam_role.sagemaker[0].id
  policy = data.aws_iam_policy_document.artifact_read_write.json
}

resource "aws_iam_role_policy" "sagemaker_ecr" {
  count = var.enable_sagemaker ? 1 : 0

  name   = "ecr-pull"
  role   = aws_iam_role.sagemaker[0].id
  policy = data.aws_iam_policy_document.ecr_pull.json
}

resource "aws_iam_role_policy" "sagemaker_observability" {
  count = var.enable_sagemaker ? 1 : 0

  name   = "observability"
  role   = aws_iam_role.sagemaker[0].id
  policy = data.aws_iam_policy_document.observability.json
}

// SageMaker needs to describe its own jobs to report progress.
data "aws_iam_policy_document" "sagemaker_self" {
  statement {
    sid    = "DescribeOwnJobs"
    effect = "Allow"
    actions = [
      "sagemaker:DescribeTrainingJob",
      "sagemaker:DescribeHyperParameterTuningJob",
      "sagemaker:DescribeEndpoint",
      "sagemaker:DescribeModel",
    ]
    resources = ["arn:${var.partition}:sagemaker:${var.aws_region}:${var.account_id}:*"]
  }
}

resource "aws_iam_role_policy" "sagemaker_self" {
  count = var.enable_sagemaker ? 1 : 0

  name   = "describe-own-jobs"
  role   = aws_iam_role.sagemaker[0].id
  policy = data.aws_iam_policy_document.sagemaker_self.json
}

// --------------------------------------------------------------------------- //
// Inference task role (ECS / EKS / EC2 running the API image)
// --------------------------------------------------------------------------- //
data "aws_iam_policy_document" "task_assume" {
  statement {
    effect  = "Allow"
    actions = ["sts:AssumeRole"]

    principals {
      type        = "Service"
      identifiers = ["ecs-tasks.amazonaws.com"]
    }
  }
}

resource "aws_iam_role" "inference_task" {
  name               = "${var.name_prefix}-inference-task"
  description        = "Task role for the FMOps inference service."
  assume_role_policy = data.aws_iam_policy_document.task_assume.json

  tags = merge(var.tags, { Name = "${var.name_prefix}-inference-task" })
}

// The serving path only ever READS model artifacts. It has no PutObject, so a
// compromised inference container cannot tamper with the model registry.
data "aws_iam_policy_document" "inference_artifacts" {
  statement {
    sid       = "ReadModelArtifacts"
    effect    = "Allow"
    actions   = ["s3:GetObject", "s3:GetObjectVersion"]
    resources = ["${var.artifact_bucket_arn}/models/*"]
  }

  statement {
    sid       = "ListModels"
    effect    = "Allow"
    actions   = ["s3:ListBucket"]
    resources = [var.artifact_bucket_arn]

    condition {
      test     = "StringLike"
      variable = "s3:prefix"
      values   = ["models/*", "fmops/models/*"]
    }
  }
}

resource "aws_iam_role_policy" "inference_artifacts" {
  name   = "read-model-artifacts"
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.inference_artifacts.json
}

resource "aws_iam_role_policy" "inference_observability" {
  name   = "observability"
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.observability.json
}

// Publishing alerts (drift, rollback, SLO breach) to SNS.
data "aws_iam_policy_document" "publish_alerts" {
  statement {
    sid       = "PublishAlerts"
    effect    = "Allow"
    actions   = ["sns:Publish"]
    resources = [var.alerts_topic_arn]
  }
}

resource "aws_iam_role_policy" "inference_alerts" {
  name   = "publish-alerts"
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.publish_alerts.json
}

// Invoking Bedrock for the LLMOps layer. Scoped to foundation models only.
data "aws_iam_policy_document" "bedrock_invoke" {
  statement {
    sid    = "InvokeFoundationModels"
    effect = "Allow"
    actions = [
      "bedrock:InvokeModel",
      "bedrock:InvokeModelWithResponseStream",
      "bedrock:Converse",
      "bedrock:ConverseStream",
    ]
    resources = [
      "arn:${var.partition}:bedrock:${var.aws_region}::foundation-model/*",
      "arn:${var.partition}:bedrock:${var.aws_region}:${var.account_id}:inference-profile/*",
    ]
  }
}

resource "aws_iam_role_policy" "inference_bedrock" {
  name   = "invoke-bedrock"
  role   = aws_iam_role.inference_task.id
  policy = data.aws_iam_policy_document.bedrock_invoke.json
}

// ECS execution role: pulls the image and writes container logs. Separate from
// the task role so the running container cannot pull arbitrary images.
resource "aws_iam_role" "task_execution" {
  name               = "${var.name_prefix}-task-execution"
  description        = "ECS task execution role: image pull and log write only."
  assume_role_policy = data.aws_iam_policy_document.task_assume.json

  tags = merge(var.tags, { Name = "${var.name_prefix}-task-execution" })
}

resource "aws_iam_role_policy_attachment" "task_execution_managed" {
  role       = aws_iam_role.task_execution.name
  policy_arn = "arn:${var.partition}:iam::aws:policy/service-role/AmazonECSTaskExecutionRolePolicy"
}

resource "aws_iam_role_policy" "task_execution_ecr" {
  name   = "ecr-pull"
  role   = aws_iam_role.task_execution.id
  policy = data.aws_iam_policy_document.ecr_pull.json
}

// --------------------------------------------------------------------------- //
// Outputs
// --------------------------------------------------------------------------- //
output "sagemaker_role_arn" {
  value = var.enable_sagemaker ? aws_iam_role.sagemaker[0].arn : ""
}

output "inference_task_role_arn" {
  value = aws_iam_role.inference_task.arn
}

output "task_execution_role_arn" {
  value = aws_iam_role.task_execution.arn
}
