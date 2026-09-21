"""Per-model context: everything that used to be one global setting.

The platform used to serve exactly one model, named in configuration, with one
endpoint, one feature list and one drift reference. Every model now carries its
own, and this module is the single place that answers, for a given model:

* which serving endpoint it has,
* which data contract (``DataConfig``) its versions were trained against,
* whether a new version is compatible with its existing lineage.

The configured model keeps its configured endpoint name, so an existing
deployment of the reference model is unaffected.
"""

from __future__ import annotations

import re

from app.core.config import DataConfig, Settings, get_settings
from app.core.exceptions import FMOpsError
from app.core.signature import ModelSignature
from app.registry.base import ModelRegistry
from app.schemas.model import ModelVersion


class UnservableModelError(FMOpsError):
    """A version whose input contract was never recorded."""

    code = "model_signature_missing"
    http_status = 409


class LineageMismatchError(FMOpsError):
    """A new version would change what the model predicts."""

    code = "lineage_mismatch"
    http_status = 409


def default_model_name(settings: Settings | None = None) -> str:
    settings = settings or get_settings()
    return settings.tracking.registered_model_name


def is_reference_model(model_name: str, settings: Settings | None = None) -> bool:
    return model_name == default_model_name(settings)


def endpoint_for(model_name: str, settings: Settings | None = None) -> str:
    """The serving endpoint that belongs to one model."""
    settings = settings or get_settings()
    if is_reference_model(model_name, settings):
        return settings.deployment.endpoint_name
    return f"fmops-{model_name.replace('_', '-')}"


def suggest_model_name(target: str, reference: bool, settings: Settings | None = None) -> str:
    """A name for a model nobody named: ``<target>_classifier``, slugged.

    The reference dataset trained on its declared target keeps the reference
    model, so its lineage and its gate comparison continue.
    """
    settings = settings or get_settings()
    if reference and target == settings.data.target_column:
        return default_model_name(settings)
    slug = re.sub(r"[^a-z0-9]+", "_", target.lower()).strip("_") or "model"
    if not slug[0].isalpha():
        slug = f"m_{slug}"
    return f"{slug}_classifier"[:64]


def signature_of(version: ModelVersion) -> ModelSignature | None:
    if not version.signature:
        return None
    return ModelSignature.model_validate(version.signature)


def data_config_for(version: ModelVersion, settings: Settings | None = None) -> DataConfig:
    """The data contract a registered version was trained against.

    Versions registered before signatures existed are all versions of the
    reference model, trained on the reference contract -- which is exactly the
    configured ``DataConfig``. Any other model without a signature cannot be
    served correctly, and saying so beats guessing a feature list.
    """
    settings = settings or get_settings()
    signature = signature_of(version)
    if signature is not None:
        return signature.to_data_config(settings.data)
    if is_reference_model(version.name, settings):
        return settings.data
    raise UnservableModelError(
        f"model {version.name} v{version.version} has no recorded input signature, so its "
        "input contract is unknown; retrain it to register a servable version",
        model=version.name,
        version=version.version,
    )


def check_lineage_compatible(
    registry: ModelRegistry,
    model_name: str,
    target: str,
    class_labels: list[str],
    task: str = "binary_classification",
) -> None:
    """Refuse a version that predicts something different from its predecessors.

    Features may change between versions -- that is ordinary model evolution.
    The target and its classes may not: the approval gate compares a candidate
    with the incumbent on the same metric, and that comparison is meaningless
    across two different prediction problems.
    """
    latest = registry.get_latest(model_name)
    if latest is None:
        return
    previous = signature_of(latest)
    if previous is None:
        return
    if previous.target != target or previous.task != task:
        raise LineageMismatchError(
            f"model {model_name} predicts {previous.target!r}; this run predicts "
            f"{target!r}. Use a new model name for a different target.",
            model=model_name,
            existing_target=previous.target,
            new_target=target,
        )
    if sorted(previous.class_labels) != sorted(class_labels):
        raise LineageMismatchError(
            f"model {model_name} has classes {previous.class_labels}; this dataset's target "
            f"has {class_labels}. Use a new model name.",
            model=model_name,
        )
