output "alb_dns_name" {
  description = "Public DNS name of the load balancer."
  value       = aws_lb.main.dns_name
}

output "alb_url" {
  description = "Public base URL. HTTP only -- no certificate is provisioned."
  value       = "http://${aws_lb.main.dns_name}"
}

output "alb_arn" {
  description = "ARN of the load balancer."
  value       = aws_lb.main.arn
}

output "target_group_arn" {
  description = "ARN of the target group, for checking target health."
  value       = aws_lb_target_group.app.arn
}

output "cluster_name" {
  description = "ECS cluster name."
  value       = aws_ecs_cluster.main.name
}

output "cluster_arn" {
  description = "ECS cluster ARN."
  value       = aws_ecs_cluster.main.arn
}

output "service_name" {
  description = "ECS service name."
  value       = aws_ecs_service.app.name
}

output "task_definition_arn" {
  description = "ARN of the active task definition revision."
  value       = aws_ecs_task_definition.app.arn
}

output "log_group_name" {
  description = "CloudWatch log group carrying container output."
  value       = aws_cloudwatch_log_group.app.name
}

output "vpc_id" {
  description = "ID of the deployment VPC."
  value       = aws_vpc.main.id
}

output "public_subnet_ids" {
  description = "Public subnet IDs the ALB and tasks run in."
  value       = aws_subnet.public[*].id
}
