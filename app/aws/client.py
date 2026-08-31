"""AWS session handling.

One place that constructs boto3 clients, so credential resolution and the
"AWS is not configured" error message are consistent everywhere.

Credentials are resolved by boto3's normal chain (environment, shared config,
instance/task role, SSO). The platform never reads an access key from its own
config files -- the ``aws_access_key_id`` settings exist only so a developer can
export them as environment variables, and they are redacted from every dump.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from app.core.config import Settings, get_settings
from app.core.exceptions import DependencyMissingError, ProviderUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


def require_boto3():
    try:
        import boto3

        return boto3
    except ImportError as exc:
        raise DependencyMissingError(
            "boto3 is not installed. Install the AWS extra:\n" "    pip install -e '.[aws]'",
            backend="aws",
        ) from exc


def require_aws_enabled(settings: Settings | None = None) -> Settings:
    settings = settings or get_settings()
    if not settings.aws.enabled:
        raise ProviderUnavailableError(
            "AWS integration is disabled. Set FMOPS_AWS__ENABLED=true and "
            "configure the region, bucket and role ARN before using AWS backends.",
            hint="see docs/deployment.md and terraform/README.md",
        )
    return settings


@lru_cache(maxsize=8)
def get_client(service: str, region: str | None = None):
    """Cached boto3 client for a service."""
    boto3 = require_boto3()
    settings = get_settings()
    return boto3.client(service, region_name=region or settings.aws.region)


def credentials_available() -> tuple[bool, str]:
    """Whether AWS credentials can be resolved right now."""
    try:
        import botocore.session
    except ImportError:
        return False, "botocore is not installed"
    try:
        credentials = botocore.session.get_session().get_credentials()
    except Exception as exc:
        return False, f"credential resolution failed: {exc}"
    if credentials is None:
        return False, "no AWS credentials found in the environment or instance role"
    return True, "credentials resolved"


def aws_status(settings: Settings | None = None) -> dict[str, Any]:
    """Diagnostic summary used by the CLI and the API."""
    settings = settings or get_settings()
    try:
        require_boto3()
        boto3_installed = True
    except DependencyMissingError:
        boto3_installed = False

    available, detail = (False, "boto3 not installed")
    if boto3_installed:
        available, detail = credentials_available()

    account = None
    if available and settings.aws.enabled:
        try:
            account = get_client("sts").get_caller_identity().get("Account")
        except Exception as exc:
            detail = f"credentials present but STS call failed: {exc}"
            available = False

    return {
        "enabled": settings.aws.enabled,
        "boto3_installed": boto3_installed,
        "credentials_available": available,
        "detail": detail,
        "region": settings.aws.region,
        "account_id": account,
        "s3_bucket": settings.aws.s3_bucket,
        "sagemaker_role_arn": bool(settings.aws.sagemaker_role_arn),
        "ecr_repository": settings.aws.ecr_repository,
    }
