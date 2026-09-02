// ECS Fargate service behind an Application Load Balancer.
//
// Deliberately the cheapest shape that still gives a stable public URL:
//
//   * Public subnets only. Fargate tasks get a public IP and reach ECR,
//     CloudWatch and S3 through the internet gateway. A NAT Gateway would add
//     roughly USD 32/month plus data processing for no benefit here -- it only
//     matters if the tasks must sit in private subnets, which is a security
//     posture this portfolio deployment does not claim to have.
//   * One task. No autoscaling, no second replica.
//   * The ECS cluster itself is free; the ALB and the running task are what
//     actually cost money.
//
// The ALB health check targets /health/live, not /health/ready. Liveness
// returns 200 whenever the process is up; readiness correctly reports
// not_ready until a model is registered. Checking readiness would make a
// correctly behaving task fail to stabilise and roll the deployment back.

terraform {
  required_providers {
    aws = {
      source  = "hashicorp/aws"
      version = ">= 5.0"
    }
  }
}

data "aws_availability_zones" "available" {
  state = "available"
}

locals {
  // An ALB requires subnets in at least two availability zones.
  azs = slice(data.aws_availability_zones.available.names, 0, 2)
}

// --------------------------------------------------------------------------- //
// Networking -- public only, no NAT gateway
// --------------------------------------------------------------------------- //
resource "aws_vpc" "main" {
  cidr_block           = var.vpc_cidr
  enable_dns_support   = true
  enable_dns_hostnames = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-vpc" })
}

resource "aws_internet_gateway" "main" {
  vpc_id = aws_vpc.main.id
  tags   = merge(var.tags, { Name = "${var.name_prefix}-igw" })
}

resource "aws_subnet" "public" {
  count = length(local.azs)

  vpc_id                  = aws_vpc.main.id
  cidr_block              = cidrsubnet(var.vpc_cidr, 8, count.index)
  availability_zone       = local.azs[count.index]
  map_public_ip_on_launch = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-public-${local.azs[count.index]}" })
}

resource "aws_route_table" "public" {
  vpc_id = aws_vpc.main.id

  route {
    cidr_block = "0.0.0.0/0"
    gateway_id = aws_internet_gateway.main.id
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-public-rt" })
}

resource "aws_route_table_association" "public" {
  count = length(aws_subnet.public)

  subnet_id      = aws_subnet.public[count.index].id
  route_table_id = aws_route_table.public.id
}

// --------------------------------------------------------------------------- //
// Security groups
// --------------------------------------------------------------------------- //
resource "aws_security_group" "alb" {
  name        = "${var.name_prefix}-alb-sg"
  description = "Public HTTP ingress to the FMOps load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description = "HTTP from the internet"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = var.ingress_cidrs
  }

  egress {
    description = "Forward to the ECS tasks"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-alb-sg" })
}

// The task accepts traffic only from the load balancer, never directly from
// the internet, even though it sits in a public subnet with a routable address.
resource "aws_security_group" "task" {
  name        = "${var.name_prefix}-task-sg"
  description = "FMOps API task; ingress restricted to the load balancer"
  vpc_id      = aws_vpc.main.id

  ingress {
    description     = "Application traffic from the ALB only"
    from_port       = var.container_port
    to_port         = var.container_port
    protocol        = "tcp"
    security_groups = [aws_security_group.alb.id]
  }

  egress {
    description = "Pull images, ship logs, reach S3"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-task-sg" })
}

// --------------------------------------------------------------------------- //
// Load balancer
// --------------------------------------------------------------------------- //
resource "aws_lb" "main" {
  name               = substr("${var.name_prefix}-alb", 0, 32)
  internal           = false
  load_balancer_type = "application"
  security_groups    = [aws_security_group.alb.id]
  subnets            = aws_subnet.public[*].id

  drop_invalid_header_fields = true

  tags = merge(var.tags, { Name = "${var.name_prefix}-alb" })
}

resource "aws_lb_target_group" "app" {
  name        = substr("${var.name_prefix}-tg", 0, 32)
  port        = var.container_port
  protocol    = "HTTP"
  vpc_id      = aws_vpc.main.id
  target_type = "ip"

  health_check {
    enabled             = true
    path                = var.health_check_path
    protocol            = "HTTP"
    matcher             = "200"
    interval            = 30
    timeout             = 10
    healthy_threshold   = 2
    unhealthy_threshold = 5
  }

  deregistration_delay = 30

  tags = merge(var.tags, { Name = "${var.name_prefix}-tg" })
}

resource "aws_lb_listener" "http" {
  load_balancer_arn = aws_lb.main.arn
  port              = 80
  protocol          = "HTTP"

  default_action {
    type             = "forward"
    target_group_arn = aws_lb_target_group.app.arn
  }
}

// --------------------------------------------------------------------------- //
// ECS
// --------------------------------------------------------------------------- //
resource "aws_cloudwatch_log_group" "app" {
  name              = "/ecs/${var.name_prefix}-api"
  retention_in_days = var.log_retention_days

  tags = var.tags
}

resource "aws_ecs_cluster" "main" {
  name = "${var.name_prefix}-cluster"

  setting {
    name = "containerInsights"
    // Container Insights bills per metric. Off for a portfolio deployment; the
    // application already exports its own Prometheus metrics on /metrics.
    value = "disabled"
  }

  tags = var.tags
}

resource "aws_ecs_task_definition" "app" {
  family                   = "${var.name_prefix}-api"
  network_mode             = "awsvpc"
  requires_compatibilities = ["FARGATE"]
  cpu                      = var.task_cpu
  memory                   = var.task_memory
  execution_role_arn       = var.execution_role_arn
  task_role_arn            = var.task_role_arn

  runtime_platform {
    operating_system_family = "LINUX"
    cpu_architecture        = "X86_64"
  }

  container_definitions = jsonencode([
    {
      name      = "api"
      image     = var.container_image
      essential = true

      portMappings = [
        {
          containerPort = var.container_port
          protocol      = "tcp"
        }
      ]

      environment = [
        for k, v in var.container_environment : { name = k, value = v }
      ]

      logConfiguration = {
        logDriver = "awslogs"
        options = {
          "awslogs-group"         = aws_cloudwatch_log_group.app.name
          "awslogs-region"        = var.aws_region
          "awslogs-stream-prefix" = "api"
        }
      }
    }
  ])

  tags = var.tags
}

resource "aws_ecs_service" "app" {
  name            = "${var.name_prefix}-api"
  cluster         = aws_ecs_cluster.main.id
  task_definition = aws_ecs_task_definition.app.arn
  desired_count   = var.desired_count
  launch_type     = "FARGATE"

  network_configuration {
    subnets = aws_subnet.public[*].id
    // Required: without a public IP a task in a public subnet cannot reach
    // ECR, and the only alternatives are a NAT gateway or interface endpoints,
    // both of which cost more than this whole deployment.
    assign_public_ip = true
    security_groups  = [aws_security_group.task.id]
  }

  load_balancer {
    target_group_arn = aws_lb_target_group.app.arn
    container_name   = "api"
    container_port   = var.container_port
  }

  // The image is large and the sklearn import is not instant; do not let the
  // health check kill the task before it has had a chance to start serving.
  health_check_grace_period_seconds = var.health_check_grace_period

  deployment_minimum_healthy_percent = 0
  deployment_maximum_percent         = 200

  depends_on = [aws_lb_listener.http]

  tags = var.tags
}
