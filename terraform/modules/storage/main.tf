// Storage: the S3 artifact bucket and the ECR repositories.

terraform {
  required_providers {
    aws = { source = "hashicorp/aws", version = "~> 5.0" }
  }
}

variable "name_prefix" { type = string }
variable "bucket_name" { type = string }
variable "enable_versioning" { type = bool }
variable "retention_days" { type = number }
variable "force_destroy" { type = bool }
variable "ecr_repositories" { type = list(string) }
variable "ecr_image_retention" { type = number }
variable "ecr_scan_on_push" { type = bool }
variable "tags" { type = map(string) }

// --------------------------------------------------------------------------- //
// S3
// --------------------------------------------------------------------------- //
resource "aws_s3_bucket" "artifacts" {
  bucket        = var.bucket_name
  force_destroy = var.force_destroy

  tags = merge(var.tags, { Name = var.bucket_name })
}

// Versioning is the last line of defence for a model artifact: an overwritten
// or deleted object stays recoverable, so a bad deploy cannot destroy the only
// copy of the model currently serving production.
resource "aws_s3_bucket_versioning" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  versioning_configuration {
    status = var.enable_versioning ? "Enabled" : "Suspended"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
    bucket_key_enabled = true
  }
}

// Model artifacts and training data are never public.
resource "aws_s3_bucket_public_access_block" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_ownership_controls" "artifacts" {
  bucket = aws_s3_bucket.artifacts.id

  rule {
    object_ownership = "BucketOwnerEnforced"
  }
}

// Reject any request that is not TLS.
data "aws_iam_policy_document" "bucket_policy" {
  statement {
    sid    = "DenyInsecureTransport"
    effect = "Deny"

    principals {
      type        = "*"
      identifiers = ["*"]
    }

    actions = ["s3:*"]
    resources = [
      aws_s3_bucket.artifacts.arn,
      "${aws_s3_bucket.artifacts.arn}/*",
    ]

    condition {
      test     = "Bool"
      variable = "aws:SecureTransport"
      values   = ["false"]
    }
  }
}

resource "aws_s3_bucket_policy" "artifacts" {
  bucket     = aws_s3_bucket.artifacts.id
  policy     = data.aws_iam_policy_document.bucket_policy.json
  depends_on = [aws_s3_bucket_public_access_block.artifacts]
}

// Superseded model versions are rarely read but must stay retrievable for audit
// and rollback, so they move to cheaper storage rather than being deleted.
// Incomplete multipart uploads are cleaned up because they are invisible in the
// console and bill silently.
resource "aws_s3_bucket_lifecycle_configuration" "artifacts" {
  count  = var.retention_days > 0 ? 1 : 0
  bucket = aws_s3_bucket.artifacts.id

  rule {
    id     = "transition-noncurrent-artifacts"
    status = "Enabled"

    filter {
      prefix = ""
    }

    noncurrent_version_transition {
      noncurrent_days = var.retention_days
      storage_class   = "STANDARD_IA"
    }

    noncurrent_version_transition {
      noncurrent_days = var.retention_days * 3
      storage_class   = "GLACIER_IR"
    }
  }

  rule {
    id     = "abort-incomplete-multipart-uploads"
    status = "Enabled"

    filter {
      prefix = ""
    }

    abort_incomplete_multipart_upload {
      days_after_initiation = 7
    }
  }

  depends_on = [aws_s3_bucket_versioning.artifacts]
}

// --------------------------------------------------------------------------- //
// ECR
// --------------------------------------------------------------------------- //
resource "aws_ecr_repository" "repos" {
  for_each = toset(var.ecr_repositories)

  name                 = "${var.name_prefix}/${each.value}"
  image_tag_mutability = "IMMUTABLE" // a tag always means the same bytes

  image_scanning_configuration {
    scan_on_push = var.ecr_scan_on_push
  }

  encryption_configuration {
    encryption_type = "AES256"
  }

  tags = merge(var.tags, { Name = "${var.name_prefix}-${each.value}" })
}

// Untagged layers are pure cost; tagged images are capped so the registry does
// not grow without bound, while keeping enough history to roll back.
resource "aws_ecr_lifecycle_policy" "repos" {
  for_each   = aws_ecr_repository.repos
  repository = each.value.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Expire untagged images after 7 days"
        selection = {
          tagStatus   = "untagged"
          countType   = "sinceImagePushed"
          countUnit   = "days"
          countNumber = 7
        }
        action = { type = "expire" }
      },
      {
        rulePriority = 2
        description  = "Retain only the most recent tagged images"
        selection = {
          tagStatus     = "tagged"
          tagPrefixList = ["v", "sha-", "main", "latest"]
          countType     = "imageCountMoreThan"
          countNumber   = var.ecr_image_retention
        }
        action = { type = "expire" }
      },
    ]
  })
}

// --------------------------------------------------------------------------- //
// Outputs
// --------------------------------------------------------------------------- //
output "artifact_bucket_name" {
  value = aws_s3_bucket.artifacts.id
}

output "artifact_bucket_arn" {
  value = aws_s3_bucket.artifacts.arn
}

output "ecr_repository_urls" {
  value = { for k, v in aws_ecr_repository.repos : k => v.repository_url }
}

output "ecr_repository_arns" {
  value = [for v in aws_ecr_repository.repos : v.arn]
}
