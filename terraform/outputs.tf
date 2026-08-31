// Outputs.
//
// These map onto the FMOPS_* environment variables the application reads, so
// wiring the platform to the infrastructure is a copy/paste rather than a
// scavenger hunt. See `terraform output fmops_environment`.

output "artifact_bucket" {
  description = "S3 bucket for datasets, model artifacts and reports."
  value       = module.storage.artifact_bucket_name
}

output "artifact_bucket_arn" {
  value = module.storage.artifact_bucket_arn
}

output "ecr_repository_urls" {
  description = "Push targets for the container images."
  value       = module.storage.ecr_repository_urls
}

output "sagemaker_role_arn" {
  description = "Execution role for SageMaker training jobs and endpoints."
  value       = module.compute.sagemaker_role_arn
}

output "inference_task_role_arn" {
  value = module.compute.inference_task_role_arn
}

output "task_execution_role_arn" {
  value = module.compute.task_execution_role_arn
}

output "alerts_topic_arn" {
  description = "SNS topic the platform publishes drift, rollback and SLO alerts to."
  value       = module.observability.alerts_topic_arn
}

output "log_groups" {
  value = module.observability.log_group_names
}

output "cloudwatch_dashboard_url" {
  value = format(
    "https://%s.console.aws.amazon.com/cloudwatch/home?region=%s#dashboards:name=%s",
    var.aws_region,
    var.aws_region,
    module.observability.dashboard_name
  )
}

output "github_actions_role_arn" {
  description = "Role for GitHub Actions to assume via OIDC. Set as the AWS_ROLE_ARN repository variable."
  value       = var.github_repository != "" ? aws_iam_role.github_actions[0].arn : ""
}

output "model_package_group" {
  value = var.enable_sagemaker ? aws_sagemaker_model_package_group.main[0].model_package_group_name : ""
}

// Everything the application needs, ready to paste into a task definition,
// a .env file, or `eval $(terraform output -raw fmops_env_exports)`.
output "fmops_environment" {
  description = "FMOPS_* settings that point the platform at this infrastructure."
  value = {
    FMOPS_ENV                          = var.environment == "prod" ? "production" : (var.environment == "staging" ? "staging" : "development")
    FMOPS_AWS__ENABLED                 = "true"
    FMOPS_AWS__REGION                  = var.aws_region
    FMOPS_AWS__S3_BUCKET               = module.storage.artifact_bucket_name
    FMOPS_AWS__S3_PREFIX               = "fmops/${var.environment}"
    FMOPS_AWS__SAGEMAKER_ROLE_ARN      = module.compute.sagemaker_role_arn
    FMOPS_AWS__SAGEMAKER_ENDPOINT_NAME = "${local.name_prefix}-endpoint"
    FMOPS_AWS__BEDROCK_REGION          = var.aws_region
    FMOPS_ALERTS__SNS_TOPIC_ARN        = module.observability.alerts_topic_arn
    FMOPS_MONITORING__CLOUDWATCH_ENABLED   = "true"
    FMOPS_MONITORING__CLOUDWATCH_NAMESPACE = module.observability.metric_namespace
    FMOPS_DEPLOYMENT__ENDPOINT_NAME    = "${local.name_prefix}-endpoint"
  }
}

output "fmops_env_exports" {
  description = "Shell-ready export statements. Usage: eval \"$(terraform output -raw fmops_env_exports)\""
  value = join("\n", [
    for k, v in {
      FMOPS_ENV                          = var.environment == "prod" ? "production" : (var.environment == "staging" ? "staging" : "development")
      FMOPS_AWS__ENABLED                 = "true"
      FMOPS_AWS__REGION                  = var.aws_region
      FMOPS_AWS__S3_BUCKET               = module.storage.artifact_bucket_name
      FMOPS_AWS__S3_PREFIX               = "fmops/${var.environment}"
      FMOPS_AWS__SAGEMAKER_ROLE_ARN      = module.compute.sagemaker_role_arn
      FMOPS_AWS__SAGEMAKER_ENDPOINT_NAME = "${local.name_prefix}-endpoint"
      FMOPS_ALERTS__SNS_TOPIC_ARN        = module.observability.alerts_topic_arn
      FMOPS_MONITORING__CLOUDWATCH_ENABLED = "true"
      FMOPS_DEPLOYMENT__ENDPOINT_NAME    = "${local.name_prefix}-endpoint"
    } : "export ${k}=${v}"
  ])
}
