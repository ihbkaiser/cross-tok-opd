"""Canonical W&B metadata for comparable KDFlow experiments."""

from __future__ import annotations

import re
from collections.abc import Iterable


SCHEMA_VERSION = "experiment-v1"
IMPLEMENTATION_VALIDATION_JOB_TYPE = "implementation-validation"

_SLUG_RE = re.compile(r"^[a-z0-9][a-z0-9._-]*$")


def _validate_slug(field: str, value: str) -> str:
    if not isinstance(value, str) or not _SLUG_RE.fullmatch(value):
        raise ValueError(
            f"{field} must be a lowercase W&B tag slug matching "
            "[a-z0-9][a-z0-9._-]*"
        )
    return value


def build_wandb_tags(
    *,
    method: str,
    regime: str,
    objective: str,
    platform: str,
    accelerator: str,
    budget: str,
    stage: str,
    student: str,
    teacher: str,
    variant: str | None = None,
    extras: Iterable[tuple[str, str]] = (),
) -> str:
    """Return deterministic, namespaced, comma-separated W&B tags.

    Run names remain human-readable identities. Tags are orthogonal metadata
    dimensions and therefore always use ``key:value`` form.
    """

    dimensions: list[tuple[str, str]] = [
        ("schema", SCHEMA_VERSION),
        ("framework", "kdflow"),
        ("method", method),
        ("regime", regime),
        ("objective", objective),
    ]
    if variant is not None:
        dimensions.append(("variant", variant))
    dimensions.extend(
        [
            ("platform", platform),
            ("accelerator", accelerator),
            ("budget", budget),
            ("stage", stage),
            ("student", student),
            ("teacher", teacher),
        ]
    )
    dimensions.extend(extras)

    seen: set[str] = set()
    tags: list[str] = []
    for key, value in dimensions:
        key = _validate_slug("tag key", key)
        value = _validate_slug(f"tag value for {key}", value)
        if key in seen:
            raise ValueError(f"duplicate W&B tag dimension: {key}")
        seen.add(key)
        tags.append(f"{key}:{value}")
    return ",".join(tags)
