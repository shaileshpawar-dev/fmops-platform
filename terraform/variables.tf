// Input variables.
//
// Nothing here has a hard-coded account id, region, ARN or secret. Every
// environment-specific value is supplied via terraform.tfvars (git-ignored) or
// TF_VAR_* environment variables, so the same configuration provisions dev,
// staging and production.

variable "project" {
  description = "Project name; prefixes every resource so accounts stay legible."
  type        = string
  default     = "fmops"

  validation {
    condition     = can(regex("^[a-z][a-z0-9-]{1,20}$", var.project))
    error_message = "project must be lowercase alphanumeric with hyphens, 2-21 characters."
  }
}

variable "environment" {
  description = "Deployment environment."
  type        = string

  validation {
    condition     = contains(["dev", "staging", "prod"], var.environment)
    error_message = "environment must be one of: dev, staging, prod."
  }
}

variable "aws_region" {
  description = "AWS region for every regional resource."
  type        = string
  default     = "us-east-1"
}

// ---------------------------------------------------------------- storage ---
variable "artifact_bucket_name" {
  description = <<-EOT
    S3 bucket for datasets, model artifacts and reports. Leave empty to derive
    a name that includes the account id, which keeps it globally unique without
    the account id ever appearing in version control.
  EOT
  type        = string
  default     = ""
}

variable "enable_bucket_versioning" {
  description = "Keep every object version. Strongly recommended: this is the last line of defence for model artifacts."
  type        = bool
  default     = true
}

variable "artifact_retention_days" {
  description = "Days before noncurrent artifact versions transition to cheaper storage. 0 disables the lifecycle rule."
  type        = number
  default     = 90
}

variable "force_destroy_bucket" {
  description = "Allow `terraform destroy` to delete a non-empty bucket. Never enable in production."
  type        = bool
  default     = false
}

// ------------------------------------------------------------------- ecr ---
variable "ecr_repositories" {
  description = "Container repositories to create."
  type        = list(string)
  default     = ["api", "training", "inference"]
}

variable "ecr_image_retention_count" {
  description = "Number of tagged images to retain per repository."
  type        = number
  default     = 20
}

variable "ecr_scan_on_push" {
  description = "Scan images for CVEs on push."
  type        = bool
  default     = true
}

// ------------------------------------------------------------ sagemaker ---
variable "enable_sagemaker" {
  description = <<-EOT
    Create the SageMaker IAM role and model-package group. This does NOT create
    an endpoint: endpoints bill per instance-hour and are created by the
    deployment pipeline, not by terraform apply.
  EOT
  type        = bool
  default     = true
}

variable "sagemaker_instance_type" {
  description = "Default instance type for SageMaker training and endpoints."
  type        = string
  default     = "ml.m5.large"
}

// ----------------------------------------------------------- monitoring ---
variable "log_retention_days" {
  description = "CloudWatch log retention. Shorter retention is cheaper; 30 days suits most teams."
  type        = number
  default     = 30

  validation {
    condition = contains(
      [1, 3, 5, 7, 14, 30, 60, 90, 120, 150, 180, 365, 400, 545, 731, 1827, 3653],
      var.log_retention_days
    )
    error_message = "log_retention_days must be a value CloudWatch Logs accepts."
  }
}

variable "alert_email" {
  description = "Email subscribed to the alerts SNS topic. Empty means no subscription is created."
  type        = string
  default     = ""
}

variable "latency_alarm_threshold_ms" {
  description = "p95 inference latency that triggers the CloudWatch alarm."
  type        = number
  default     = 250
}

variable "error_rate_alarm_threshold" {
  description = "Error rate (0-1) that triggers the CloudWatch alarm."
  type        = number
  default     = 0.02
}

variable "drift_alarm_threshold" {
  description = "Drift score that triggers the CloudWatch alarm."
  type        = number
  default     = 0.2
}

// -------------------------------------------------------------- ci / oidc ---
variable "github_repository" {
  description = <<-EOT
    GitHub repository allowed to assume the CI role via OIDC, as "owner/repo".
    Empty disables the OIDC role entirely -- no long-lived CI keys are ever
    created by this configuration.
  EOT
  type        = string
  default     = ""
}

variable "github_oidc_provider_arn" {
  description = "ARN of an existing GitHub OIDC provider. Empty creates one."
  type        = string
  default     = ""
}

// -------------------------------------------------------------- tagging ---
variable "additional_tags" {
  description = "Extra tags merged into every resource."
  type        = map(string)
  default     = {}
}

variable "cost_center" {
  description = "Cost allocation tag."
  type        = string
  default     = "ml-platform"
}

// --------------------------------------------------------------------------- //
// ECS Fargate service
// --------------------------------------------------------------------------- //
variable "enable_ecs_service" {
  description = <<-EOT
    Create the VPC, ALB and Fargate service that serve the public URL.
    This is the only part of the stack with a meaningful hourly cost: an ALB
    is roughly USD 16-18/month and a 0.5 vCPU / 1 GB task about USD 15/month
    in ap-south-1. Set to false to keep only ECR, S3 and IAM.
  EOT
  type        = bool
  default     = false
}

variable "container_image" {
  description = <<-EOT
    Image the ECS task runs, as a commit-SHA tag or a @sha256 digest.

    Must not be a moving tag. The ECR repositories are IMMUTABLE, so a tag
    that has been pushed cannot be repointed -- but a moving tag would still
    make the running revision ambiguous and a rollback unrepeatable, which is
    the actual reason to refuse it here.
  EOT
  type        = string
  default     = ""

  validation {
    condition = (
      var.container_image == "" ||
      !can(regex(":(latest|stable|main|master|prod|production)$", var.container_image))
    )
    error_message = "container_image must be a commit-SHA tag or a @sha256 digest, not a moving tag such as :latest."
  }
}

variable "ecs_task_cpu" {
  description = "Fargate CPU units for the API task. 512 = 0.5 vCPU."
  type        = string
  default     = "512"
}

variable "ecs_task_memory" {
  description = "Fargate memory (MiB) for the API task."
  type        = string
  default     = "1024"
}

variable "ecs_desired_count" {
  description = "Number of API tasks. One for a portfolio deployment."
  type        = number
  default     = 1
}

variable "ecs_ingress_cidrs" {
  description = "CIDRs allowed to reach the load balancer on port 80."
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "ecs_extra_environment" {
  description = "Additional container environment variables, merged last."
  type        = map(string)
  default     = {}
}
