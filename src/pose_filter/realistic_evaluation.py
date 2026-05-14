"""Compatibility exports for realistic synthetic-measurement evaluation.

The realistic measurement path is now first-class in :mod:`pose_filter.evaluation`.
This module remains as a small import bridge for code that used the previous
``pose_filter.realistic_evaluation`` wrappers.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from .evaluation import (
    ablation_rows,
    evaluate_filter,
    evaluate_filter_sequence,
    evaluate_filter_sequence_artifacts,
    evaluate_filter_with_artifacts,
    robustness_rows,
    trajectory_preview_rows,
)
from .measurement_config import MeasurementConfigLike, MeasurementRealismConfig


@contextmanager
def realistic_measurement_context(measurement_config: MeasurementConfigLike = None) -> Iterator[MeasurementRealismConfig]:
    """Yield a typed measurement config without monkey-patching global functions.

    This context manager is kept only for backward compatibility with the first
    integration layer. New code should pass ``measurement_config`` directly to the
    evaluation functions.
    """

    yield MeasurementRealismConfig.coerce(measurement_config)


__all__ = [
    "MeasurementConfigLike",
    "MeasurementRealismConfig",
    "ablation_rows",
    "evaluate_filter",
    "evaluate_filter_sequence",
    "evaluate_filter_sequence_artifacts",
    "evaluate_filter_with_artifacts",
    "realistic_measurement_context",
    "robustness_rows",
    "trajectory_preview_rows",
]
