"""Prompt versioning.

Prompts are treated as code, not configuration: they live in the repo under
``app/llmops/prompts/library/<name>.yaml``, carry explicit semantic versions,
and are content-hashed. Every LLM trace records the prompt name, version and
hash, so any output can be traced back to the exact text that produced it -- and
a prompt edited without a version bump is detectable, because the stored hash
will not match.

File format::

    name: support_summarizer
    description: Summarise a support ticket into a structured triage record.
    versions:
      - version: "1.0.0"
        description: initial version
        system: You are a support triage assistant.
        template: |
          Summarise the following ticket.
          Ticket: {ticket_text}
        variables: [ticket_text]
        model: null
        temperature: 0.0
        tags: [baseline]
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

from app.core.config import Settings, get_settings
from app.core.exceptions import PromptNotFoundError
from app.core.logging import get_logger
from app.core.utils import git_commit, hash_text
from app.schemas.llm import PromptRenderResult, PromptVersion

logger = get_logger(__name__)

_VARIABLE_PATTERN = re.compile(r"\{([a-zA-Z_][a-zA-Z0-9_]*)\}")


def _version_key(version: str) -> tuple:
    """Sort key for semantic-ish versions; unparseable parts sort last."""
    parts: list[tuple[int, int | str]] = []
    for chunk in str(version).split("."):
        try:
            parts.append((0, int(chunk)))
        except ValueError:
            parts.append((1, chunk))
    return tuple(parts)


class PromptRegistry:
    """Loads, validates and renders versioned prompts."""

    def __init__(self, directory: Path | None = None, settings: Settings | None = None):
        self.settings = settings or get_settings()
        self.directory = directory or self.settings.resolved_prompt_dir()
        self._cache: dict[str, list[PromptVersion]] = {}

    # -- loading ------------------------------------------------------------- #
    def _load_file(self, name: str) -> list[PromptVersion]:
        if name in self._cache:
            return self._cache[name]

        path = self.directory / f"{name}.yaml"
        if not path.is_file():
            available = self.list_names()
            raise PromptNotFoundError(
                f"prompt {name!r} not found in {self.directory}",
                prompt=name,
                available=available,
            )

        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        raw_versions = payload.get("versions") or []
        if not raw_versions:
            raise PromptNotFoundError(f"prompt file {path} declares no versions", prompt=name)

        commit = git_commit()
        versions: list[PromptVersion] = []
        for entry in raw_versions:
            template = entry.get("template", "")
            system = entry.get("system")
            declared = entry.get("variables")
            detected = sorted(set(_VARIABLE_PATTERN.findall(template)))
            variables = list(declared) if declared else detected

            if declared is not None:
                missing = sorted(set(detected) - set(declared))
                if missing:
                    # A placeholder nobody declared is almost always a typo.
                    logger.warning(
                        "prompt.undeclared_variables",
                        extra={
                            "prompt": name,
                            "version": entry.get("version"),
                            "undeclared": missing,
                        },
                    )
                    variables = sorted(set(variables) | set(missing))

            versions.append(
                PromptVersion(
                    name=payload.get("name", name),
                    version=str(entry["version"]),
                    template=template,
                    system=system,
                    description=entry.get("description", payload.get("description", "")),
                    variables=variables,
                    model=entry.get("model"),
                    temperature=entry.get("temperature"),
                    max_tokens=entry.get("max_tokens"),
                    tags=list(entry.get("tags") or []),
                    content_hash=hash_text(f"{system or ''}\n{template}"),
                    git_commit=commit,
                )
            )

        versions.sort(key=lambda v: _version_key(v.version))
        self._cache[name] = versions
        return versions

    # -- queries ------------------------------------------------------------- #
    def list_names(self) -> list[str]:
        if not self.directory.is_dir():
            return []
        return sorted(p.stem for p in self.directory.glob("*.yaml"))

    def list_versions(self, name: str) -> list[PromptVersion]:
        return list(self._load_file(name))

    def get(self, name: str, version: str | None = None) -> PromptVersion:
        """Fetch a prompt version; ``None`` means the highest version."""
        versions = self._load_file(name)
        if version is None:
            return versions[-1]
        for candidate in versions:
            if candidate.version == version:
                return candidate
        raise PromptNotFoundError(
            f"prompt {name!r} has no version {version!r}",
            prompt=name,
            version=version,
            available=[v.version for v in versions],
        )

    def latest_version(self, name: str) -> str:
        return self._load_file(name)[-1].version

    def all_prompts(self) -> dict[str, list[PromptVersion]]:
        return {name: self.list_versions(name) for name in self.list_names()}

    # -- rendering ----------------------------------------------------------- #
    def render(
        self,
        name: str,
        variables: dict[str, Any] | None = None,
        version: str | None = None,
    ) -> PromptRenderResult:
        """Substitute variables into a prompt version.

        Missing variables raise rather than rendering a template with a literal
        ``{placeholder}`` in it, which would silently degrade output quality.
        """
        prompt = self.get(name, version)
        values = variables or {}
        missing = [v for v in prompt.variables if v not in values]
        if missing:
            raise PromptNotFoundError(
                f"prompt {prompt.key} requires variables that were not supplied: "
                f"{', '.join(missing)}",
                prompt=name,
                version=prompt.version,
                missing_variables=missing,
                required=prompt.variables,
            )

        try:
            rendered = prompt.template.format(**values)
            system = prompt.system.format(**values) if prompt.system else None
        except (KeyError, IndexError, ValueError) as exc:
            raise PromptNotFoundError(
                f"prompt {prompt.key} could not be rendered: {exc}",
                prompt=name,
                version=prompt.version,
            ) from exc

        return PromptRenderResult(
            prompt=prompt,
            rendered=rendered,
            system=system,
            variables_used={k: values[k] for k in prompt.variables if k in values},
        )

    def reload(self) -> None:
        self._cache.clear()

    def diff(self, name: str, version_a: str, version_b: str) -> dict[str, Any]:
        """Unified diff between two prompt versions, for review."""
        import difflib

        a = self.get(name, version_a)
        b = self.get(name, version_b)
        diff = list(
            difflib.unified_diff(
                (a.system or "").splitlines() + a.template.splitlines(),
                (b.system or "").splitlines() + b.template.splitlines(),
                fromfile=a.key,
                tofile=b.key,
                lineterm="",
            )
        )
        return {
            "prompt": name,
            "from_version": version_a,
            "to_version": version_b,
            "from_hash": a.content_hash[:12],
            "to_hash": b.content_hash[:12],
            "identical": a.content_hash == b.content_hash,
            "diff": diff,
        }


_REGISTRY: PromptRegistry | None = None


def get_prompt_registry() -> PromptRegistry:
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = PromptRegistry()
    return _REGISTRY


def set_prompt_registry(registry: PromptRegistry | None) -> None:
    global _REGISTRY
    _REGISTRY = registry
