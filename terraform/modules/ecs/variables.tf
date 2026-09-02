variable "name_prefix" {
  description = "Prefix applied to every resource name."
  type        = string
}

variable "aws_region" {
  description = "Region the log group and task run in."
  type        = string
}

variable "vpc_cidr" {
  description = "CIDR for the deployment VPC."
  type        = string
  default     = "10.20.0.0/16"
}

variable "ingress_cidrs" {
  description = <<-EOT
    Who may reach the load balancer on port 80. Defaults to the whole internet
    because the point of this deployment is a publicly reachable demo URL.
    Narrow it to your own address if you would rather not have it open.
  EOT
  type        = list(string)
  default     = ["0.0.0.0/0"]
}

variable "container_image" {
  description = "Fully qualified image reference, including tag or digest."
  type        = string
}

variable "container_port" {
  description = "Port the application listens on inside the container."
  type        = number
  default     = 8000
}

variable "container_environment" {
  description = "Environment variables passed to the container."
  type        = map(string)
  default     = {}
}

variable "task_cpu" {
  description = "Fargate CPU units. 512 = 0.5 vCPU."
  type        = string
  default     = "512"

  validation {
    condition     = contains(["256", "512", "1024", "2048", "4096"], var.task_cpu)
    error_message = "task_cpu must be a valid Fargate CPU value."
  }
}

variable "task_memory" {
  description = <<-EOT
    Fargate memory in MiB. The image carries scikit-learn, pandas and MLflow;
    below 1024 MiB the interpreter has been observed to be killed during import.
  EOT
  type        = string
  default     = "1024"
}

variable "desired_count" {
  description = "Number of tasks. One, for a portfolio deployment."
  type        = number
  default     = 1

  validation {
    condition     = var.desired_count >= 0 && var.desired_count <= 4
    error_message = "desired_count is capped at 4 to avoid an accidental cost surprise."
  }
}

variable "health_check_path" {
  description = <<-EOT
    Path the ALB polls. Must be a liveness probe, not a readiness probe:
    /health/ready returns 503 until a model is registered, which is correct
    application behaviour but would prevent the service from ever stabilising.
  EOT
  type        = string
  default     = "/health/live"
}

variable "health_check_grace_period" {
  description = "Seconds before the ALB health check starts counting against a new task."
  type        = number
  default     = 120
}

variable "log_retention_days" {
  description = "CloudWatch log retention. Kept short; logs are billed by volume."
  type        = number
  default     = 14
}

variable "execution_role_arn" {
  description = "Role ECS itself assumes to pull the image and write logs."
  type        = string
}

variable "task_role_arn" {
  description = "Role the application assumes at runtime (S3, CloudWatch, SNS)."
  type        = string
}

variable "tags" {
  description = "Tags applied to every resource."
  type        = map(string)
  default     = {}
}
