"""Artifact storage abstraction.

The platform writes datasets, model binaries, evaluation reports and drift
reports through :class:`ArtifactStore`.  Two implementations ship:

* :class:`LocalArtifactStore` -- the filesystem. The default everywhere.
* :class:`S3ArtifactStore`    -- Amazon S3 via boto3. Used when
  ``FMOPS_AWS__ENABLED=true`` and a bucket is configured.

Both speak URIs, so the rest of the codebase records an ``artifact_uri`` and
never cares which backend produced it:

    file:///abs/path/models/loan_default/3/model.joblib
    s3://my-bucket/fmops/models/loan_default/3/model.joblib
"""

from __future__ import annotations

import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO
from urllib.parse import urlparse
from urllib.request import url2pathname

from app.core.config import Settings, get_settings
from app.core.exceptions import DependencyMissingError, ProviderUnavailableError
from app.core.logging import get_logger

logger = get_logger(__name__)


class ArtifactStore(ABC):
    """Content-addressable-ish blob store keyed by a relative logical path."""

    scheme: str = "file"

    @abstractmethod
    def put_bytes(self, key: str, payload: bytes) -> str:
        """Store raw bytes and return the artifact URI."""

    @abstractmethod
    def get_bytes(self, key: str) -> bytes:
        """Read raw bytes for a key."""

    @abstractmethod
    def put_file(self, key: str, source: Path | str) -> str:
        """Upload a local file and return its URI."""

    @abstractmethod
    def get_file(self, key: str, destination: Path | str) -> Path:
        """Download a key to a local path and return that path."""

    @abstractmethod
    def exists(self, key: str) -> bool: ...

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]: ...

    @abstractmethod
    def delete(self, key: str) -> None: ...

    @abstractmethod
    def uri(self, key: str) -> str:
        """The fully-qualified URI a key resolves to."""

    def put_text(self, key: str, text: str) -> str:
        return self.put_bytes(key, text.encode("utf-8"))

    def get_text(self, key: str) -> str:
        return self.get_bytes(key).decode("utf-8")

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<{type(self).__name__} scheme={self.scheme}>"


class LocalArtifactStore(ArtifactStore):
    """Filesystem-backed store rooted at a directory."""

    scheme = "file"

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        # Reject traversal: the resolved path must stay under the root.
        candidate = (self.root / key.lstrip("/")).resolve()
        if not str(candidate).startswith(str(self.root)):
            raise ValueError(f"artifact key escapes store root: {key!r}")
        return candidate

    def put_bytes(self, key: str, payload: bytes) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(payload)
        tmp.replace(path)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        path = self._path(key)
        if not path.is_file():
            raise FileNotFoundError(f"artifact not found: {self.uri(key)}")
        return path.read_bytes()

    def put_file(self, key: str, source: Path | str) -> str:
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # The producer often writes straight into the store root (models are
        # dumped under artifacts/models/... which is also the store's key
        # space). Copying a file onto itself is a no-op, not an error.
        if Path(source).resolve() != path:
            shutil.copyfile(source, path)
        return self.uri(key)

    def get_file(self, key: str, destination: Path | str) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        src = self._path(key)
        if not src.is_file():
            raise FileNotFoundError(f"artifact not found: {self.uri(key)}")
        if src != dest.resolve():
            shutil.copyfile(src, dest)
        return dest

    def exists(self, key: str) -> bool:
        return self._path(key).exists()

    def list(self, prefix: str = "") -> list[str]:
        base = self._path(prefix) if prefix else self.root
        if base.is_file():
            return [base.relative_to(self.root).as_posix()]
        if not base.is_dir():
            return []
        return sorted(
            p.relative_to(self.root).as_posix() for p in base.rglob("*") if p.is_file()
        )

    def delete(self, key: str) -> None:
        path = self._path(key)
        if path.is_dir():
            shutil.rmtree(path)
        elif path.exists():
            path.unlink()

    def uri(self, key: str) -> str:
        return self._path(key).as_uri()

    def local_path(self, key: str) -> Path:
        """Escape hatch for callers that genuinely need a filesystem path."""
        return self._path(key)


class S3ArtifactStore(ArtifactStore):
    """Amazon S3 backed store. Requires the ``[aws]`` extra (boto3)."""

    scheme = "s3"

    def __init__(
        self,
        bucket: str,
        prefix: str = "",
        region: str | None = None,
        client: object | None = None,
    ) -> None:
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        self.region = region
        self._client = client

    @property
    def client(self):
        if self._client is None:
            try:
                import boto3
            except ImportError as exc:
                raise DependencyMissingError(
                    "boto3 is required for S3 artifact storage; "
                    "install the [aws] extra: pip install -e '.[aws]'",
                    backend="s3",
                ) from exc
            self._client = boto3.client("s3", region_name=self.region)
        return self._client

    def _key(self, key: str) -> str:
        key = key.lstrip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def _call(self, operation: str, **kwargs):
        try:
            return getattr(self.client, operation)(**kwargs)
        except Exception as exc:  # botocore raises many concrete types
            name = type(exc).__name__
            if name in {"NoSuchKey", "404", "ClientError"} and operation in {
                "get_object",
                "head_object",
            }:
                raise
            raise ProviderUnavailableError(
                f"S3 {operation} failed: {exc}",
                bucket=self.bucket,
                operation=operation,
            ) from exc

    def put_bytes(self, key: str, payload: bytes) -> str:
        self._call("put_object", Bucket=self.bucket, Key=self._key(key), Body=payload)
        return self.uri(key)

    def get_bytes(self, key: str) -> bytes:
        try:
            response = self._call("get_object", Bucket=self.bucket, Key=self._key(key))
        except Exception as exc:
            raise FileNotFoundError(f"artifact not found: {self.uri(key)}") from exc
        body: BinaryIO = response["Body"]
        return body.read()

    def put_file(self, key: str, source: Path | str) -> str:
        try:
            self.client.upload_file(str(source), self.bucket, self._key(key))
        except Exception as exc:
            raise ProviderUnavailableError(
                f"S3 upload failed: {exc}", bucket=self.bucket, key=key
            ) from exc
        return self.uri(key)

    def get_file(self, key: str, destination: Path | str) -> Path:
        dest = Path(destination)
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download_file(self.bucket, self._key(key), str(dest))
        except Exception as exc:
            raise FileNotFoundError(f"artifact not found: {self.uri(key)}") from exc
        return dest

    def exists(self, key: str) -> bool:
        try:
            self.client.head_object(Bucket=self.bucket, Key=self._key(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = self._key(prefix) if prefix else self.prefix
        keys: list[str] = []
        token: str | None = None
        strip = f"{self.prefix}/" if self.prefix else ""
        while True:
            kwargs = {"Bucket": self.bucket, "Prefix": full_prefix}
            if token:
                kwargs["ContinuationToken"] = token
            response = self._call("list_objects_v2", **kwargs)
            for item in response.get("Contents", []):
                name = item["Key"]
                keys.append(name[len(strip) :] if strip and name.startswith(strip) else name)
            if not response.get("IsTruncated"):
                break
            token = response.get("NextContinuationToken")
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._call("delete_object", Bucket=self.bucket, Key=self._key(key))

    def uri(self, key: str) -> str:
        return f"s3://{self.bucket}/{self._key(key)}"


# --------------------------------------------------------------------------- #
# Factory
# --------------------------------------------------------------------------- #
def build_artifact_store(settings: Settings | None = None) -> ArtifactStore:
    """Return the artifact store implied by configuration.

    S3 is used only when AWS is explicitly enabled *and* a bucket is set;
    otherwise the local store is returned.  We never silently pretend a local
    directory is S3.
    """
    settings = settings or get_settings()
    if settings.aws.enabled and settings.aws.s3_bucket:
        logger.info(
            "artifact_store.s3",
            extra={"bucket": settings.aws.s3_bucket, "prefix": settings.aws.s3_prefix},
        )
        return S3ArtifactStore(
            bucket=settings.aws.s3_bucket,
            prefix=settings.aws.s3_prefix,
            region=settings.aws.region,
        )
    logger.debug("artifact_store.local", extra={"root": str(settings.paths.artifacts_dir)})
    return LocalArtifactStore(settings.paths.artifacts_dir)


def resolve_uri_to_local(uri: str, destination: Path | str | None = None) -> Path:
    """Materialise any artifact URI as a local file path.

    ``file://`` URIs and plain paths resolve directly; ``s3://`` URIs are
    downloaded (requires boto3).
    """
    parsed = urlparse(uri)
    if parsed.scheme in ("", "file"):
        if parsed.scheme == "file":
            # url2pathname handles both the percent-decoding (paths with spaces
            # or parentheses round-trip through file:// encoded) and the
            # platform-specific shape (/C:/x -> C:\x on Windows).
            path = Path(url2pathname(parsed.path))
            if parsed.netloc:  # UNC share: file://server/share/x
                path = Path(f"//{parsed.netloc}") / path.relative_to(path.anchor or "/")
            return path
        return Path(uri)
    if parsed.scheme == "s3":
        bucket = parsed.netloc
        key = parsed.path.lstrip("/")
        store = S3ArtifactStore(bucket=bucket, prefix="")
        dest = Path(destination) if destination else Path(key).name
        return store.get_file(key, dest)
    raise ValueError(f"unsupported artifact URI scheme: {uri!r}")
