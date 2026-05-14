"""Compatibility helpers for realistic synthetic measurement evaluations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import evaluation as _base
from .measurement_config import MeasurementConfigLike, MeasurementRealismConfig

MeasurementKwargs = dict[str, Any]


@contextmanager
def realistic_measurement_context(measurement_kwargs: MeasurementConfigLike) -> Iterator[None]:
    """Compatibility no-op context for older callers.

    Realistic measurement options are now passed through first-class
    ``measurement_config`` arguments in :mod:`pose_filter.evaluation`. This
    context remains so external scripts importing it do not break.
    """

    _ = MeasurementRealismConfig.coerce(measurement_kwargs)
    yield


def evaluate_filter_with_artifacts(*args: Any, measurement_kwargs: MeasurementConfigLike = None, **kwargs: Any) -> Any:
    kwargs.setdefault("measurement_config", measurement_kwargs)
    return _base.evaluate_filter_with_artifacts(*args, **kwargs)


def ablation_rows(*args: Any, measurement_kwargs: MeasurementConfigLike = None, **kwargs: Any) -> Any:
    kwargs.setdefault("measurement_config", measurement_kwargs)
    return _base.ablation_rows(*args, **kwargs)


def robustness_rows(*args: Any, measurement_kwargs: MeasurementConfigLike = None, **kwargs: Any) -> Any:
    kwargs.setdefault("measurement_config", measurement_kwargs)
    return _base.robustness_rows(*args, **kwargs)


def trajectory_preview_rows(*args: Any, measurement_kwargs: MeasurementConfigLike = None, **kwargs: Any) -> Any:
    kwargs.setdefault("measurement_config", measurement_kwargs)
    return _base.trajectory_preview_rows(*args, **kwargs)
