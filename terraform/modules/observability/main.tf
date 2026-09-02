// Observability: log groups, the alerts topic, CloudWatch alarms and a
// dashboard.
//
// The alarms here mirror the SLOs the platform enforces internally
// (app/monitoring), so a breach is visible whether you are looking at Grafana,
// the FMOps dashboard, or the AWS console.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "name_prefix" { type = string }
variable "environment" { type = string }
variable "aws_region" { type = string }
variable "log_retention_days" { type = number }
variable "alert_email" { type = string }
variable "latency_threshold_ms" { type = number }
variable "error_rate_threshold" { type = number }
variable "drift_threshold" { type = number }
variable "sagemaker_endpoint_name" { type = string }
variable "tags" { type = map(string) }

locals {
  // Matches monitoring.cloudwatch_namespace in configs/<env>.yaml.
  metric_namespace = var.environment == "prod" ? "FMOps/Production" : (
    var.environment == "staging" ? "FMOps/Staging" : "FMOps"
  )
}

// --------------------------------------------------------------------------- //
// Log groups
// --------------------------------------------------------------------------- //
resource "aws_cloudwatch_log_group" "api" {
  name              = "/fmops/${var.environment}/api"
  retention_in_days = var.log_retention_days
  tags              = merge(var.tags, { Component = "api" })
}

resource "aws_cloudwatch_log_group" "training" {
  name              = "/fmops/${var.environment}/training"
  retention_in_days = var.log_retention_days
  tags              = merge(var.tags, { Component = "training" })
}

resource "aws_cloudwatch_log_group" "pipeline" {
  name              = "/fmops/${var.environment}/pipeline"
  retention_in_days = var.log_retention_days
  tags              = merge(var.tags, { Component = "pipeline" })
}

// --------------------------------------------------------------------------- //
// Alerts topic
// --------------------------------------------------------------------------- //
resource "aws_sns_topic" "alerts" {
  name              = "${var.name_prefix}-alerts"
  display_name      = "FMOps ${var.environment} alerts"
  kms_master_key_id = "alias/aws/sns"

  tags = merge(var.tags, { Name = "${var.name_prefix}-alerts" })
}

resource "aws_sns_topic_subscription" "email" {
  count = var.alert_email != "" ? 1 : 0

  topic_arn = aws_sns_topic.alerts.arn
  protocol  = "email"
  endpoint  = var.alert_email
}

// --------------------------------------------------------------------------- //
// Metric filters -- turn structured log lines into CloudWatch metrics
// --------------------------------------------------------------------------- //
// The platform logs JSON, so drift and rollback events can be extracted from
// the log stream without any additional instrumentation.
resource "aws_cloudwatch_log_metric_filter" "drift_detected" {
  name           = "${var.name_prefix}-drift-detected"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "{ $.message = \"drift.scan_completed\" && $.detected IS TRUE }"

  metric_transformation {
    name          = "DriftDetected"
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "rollback" {
  name           = "${var.name_prefix}-rollback"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "{ $.message = \"deployment.rolled_back\" }"

  metric_transformation {
    name          = "Rollbacks"
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "errors" {
  name           = "${var.name_prefix}-errors"
  log_group_name = aws_cloudwatch_log_group.api.name
  pattern        = "{ $.level = \"ERROR\" }"

  metric_transformation {
    name          = "ApplicationErrors"
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}

resource "aws_cloudwatch_log_metric_filter" "candidate_rejected" {
  name           = "${var.name_prefix}-candidate-rejected"
  log_group_name = aws_cloudwatch_log_group.pipeline.name
  pattern        = "{ $.message = \"retraining.candidate_rejected\" }"

  metric_transformation {
    name          = "RetrainingCandidatesRejected"
    namespace     = local.metric_namespace
    value         = "1"
    default_value = "0"
  }
}

// --------------------------------------------------------------------------- //
// Alarms
// --------------------------------------------------------------------------- //
resource "aws_cloudwatch_metric_alarm" "drift" {
  alarm_name        = "${var.name_prefix}-drift-detected"
  alarm_description = "A drift scan exceeded the configured threshold. Retraining will be triggered."

  namespace           = local.metric_namespace
  metric_name         = "DriftDetected"
  statistic           = "Maximum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "rollback" {
  alarm_name        = "${var.name_prefix}-rollback-occurred"
  alarm_description = "A model deployment was rolled back. Investigate before redeploying."

  namespace           = local.metric_namespace
  metric_name         = "Rollbacks"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 1
  comparison_operator = "GreaterThanOrEqualToThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "error_spike" {
  alarm_name        = "${var.name_prefix}-error-spike"
  alarm_description = "Sustained application errors in the FMOps API."

  namespace           = local.metric_namespace
  metric_name         = "ApplicationErrors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 2
  threshold           = 10
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  alarm_actions = [aws_sns_topic.alerts.arn]
  tags          = var.tags
}

// SageMaker publishes endpoint latency itself. The alarm is created regardless
// of whether the endpoint exists yet; it sits in INSUFFICIENT_DATA until the
// deployment pipeline creates the endpoint.
resource "aws_cloudwatch_metric_alarm" "endpoint_latency" {
  alarm_name        = "${var.name_prefix}-endpoint-latency"
  alarm_description = "p95 model latency exceeded the SLO."

  namespace           = "AWS/SageMaker"
  metric_name         = "ModelLatency"
  extended_statistic  = "p95"
  period              = 300
  evaluation_periods  = 2
  threshold           = var.latency_threshold_ms * 1000 // SageMaker reports microseconds
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = var.sagemaker_endpoint_name
    VariantName  = "AllTraffic"
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  tags          = var.tags
}

resource "aws_cloudwatch_metric_alarm" "endpoint_5xx" {
  alarm_name        = "${var.name_prefix}-endpoint-5xx"
  alarm_description = "The SageMaker endpoint is returning server errors."

  namespace           = "AWS/SageMaker"
  metric_name         = "Invocation5XXErrors"
  statistic           = "Sum"
  period              = 300
  evaluation_periods  = 1
  threshold           = 5
  comparison_operator = "GreaterThanThreshold"
  treat_missing_data  = "notBreaching"

  dimensions = {
    EndpointName = var.sagemaker_endpoint_name
    VariantName  = "AllTraffic"
  }

  alarm_actions = [aws_sns_topic.alerts.arn]
  tags          = var.tags
}

// --------------------------------------------------------------------------- //
// Dashboard
// --------------------------------------------------------------------------- //
resource "aws_cloudwatch_dashboard" "main" {
  dashboard_name = "${var.name_prefix}-overview"

  dashboard_body = jsonencode({
    widgets = [
      {
        type   = "metric"
        x      = 0
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Endpoint invocations and errors"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Sum"
          period = 300
          metrics = [
            ["AWS/SageMaker", "Invocations", "EndpointName", var.sagemaker_endpoint_name, "VariantName", "AllTraffic"],
            [".", "Invocation4XXErrors", ".", ".", ".", "."],
            [".", "Invocation5XXErrors", ".", ".", ".", "."],
          ]
        }
      },
      {
        type   = "metric"
        x      = 12
        y      = 0
        width  = 12
        height = 6
        properties = {
          title  = "Model latency"
          region = var.aws_region
          view   = "timeSeries"
          period = 300
          metrics = [
            ["AWS/SageMaker", "ModelLatency", "EndpointName", var.sagemaker_endpoint_name, "VariantName", "AllTraffic", { stat = "p50" }],
            ["...", { stat = "p95" }],
            ["...", { stat = "p99" }],
          ]
          annotations = {
            horizontal = [{
              label = "SLO"
              value = var.latency_threshold_ms * 1000
            }]
          }
        }
      },
      {
        type   = "metric"
        x      = 0
        y      = 6
        width  = 8
        height = 6
        properties = {
          title   = "Drift detections"
          region  = var.aws_region
          view    = "timeSeries"
          stat    = "Maximum"
          period  = 300
          metrics = [[local.metric_namespace, "DriftDetected"]]
        }
      },
      {
        type   = "metric"
        x      = 8
        y      = 6
        width  = 8
        height = 6
        properties = {
          title  = "Rollbacks and rejected candidates"
          region = var.aws_region
          view   = "timeSeries"
          stat   = "Sum"
          period = 3600
          metrics = [
            [local.metric_namespace, "Rollbacks"],
            [".", "RetrainingCandidatesRejected"],
          ]
        }
      },
      {
        type   = "metric"
        x      = 16
        y      = 6
        width  = 8
        height = 6
        properties = {
          title   = "Application errors"
          region  = var.aws_region
          view    = "timeSeries"
          stat    = "Sum"
          period  = 300
          metrics = [[local.metric_namespace, "ApplicationErrors"]]
        }
      },
      {
        type   = "log"
        x      = 0
        y      = 12
        width  = 24
        height = 6
        properties = {
          title  = "Recent errors"
          region = var.aws_region
          query  = <<-EOT
            SOURCE '${aws_cloudwatch_log_group.api.name}'
            | fields @timestamp, message, error_code, request_id, error.message
            | filter level = "ERROR"
            | sort @timestamp desc
            | limit 50
          EOT
        }
      },
    ]
  })
}

// --------------------------------------------------------------------------- //
// Outputs
// --------------------------------------------------------------------------- //
output "alerts_topic_arn" {
  value = aws_sns_topic.alerts.arn
}

output "log_group_names" {
  value = [
    aws_cloudwatch_log_group.api.name,
    aws_cloudwatch_log_group.training.name,
    aws_cloudwatch_log_group.pipeline.name,
  ]
}

output "log_group_arns" {
  value = [
    aws_cloudwatch_log_group.api.arn,
    aws_cloudwatch_log_group.training.arn,
    aws_cloudwatch_log_group.pipeline.arn,
  ]
}

output "dashboard_name" {
  value = aws_cloudwatch_dashboard.main.dashboard_name
}

output "metric_namespace" {
  value = local.metric_namespace
}
