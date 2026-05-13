"""Evaluation wrappers for configurable realistic synthetic measurements."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from . import evaluation as _base

MeasurementKwargs = dict[str, Any]


@contextmanager
def realistic_measurement_context(measurement_kwargs: MeasurementKwargs | None) -> Iterator[None]:
    """Temporarily pass realistic measurement options into base evaluations.

    The base evaluation functions generate synthetic measurements internally. This
    context injects additional measurement-generator keyword arguments and also
    forwards confidence-calibrated per-joint noise into ``run_filter`` so the
    likelihood matches the generated measurement model.
    """

    kwargs = dict(measurement_kwargs or {})
    if not kwargs:
        yield
        return

    original_make_synthetic_measurements = _base.make_synthetic_measurements
    original_run_filter = _base.run_filter
    measurements_by_observation_id: dict[int, Any] = {}

    def make_synthetic_measurements(*args: Any, **call_kwargs: Any) -> Any:
        call_kwargs = {**call_kwargs, **kwargs}
        measurements = original_make_synthetic_measurements(*args, **call_kwargs)
        measurements_by_observation_id[id(measurements.observations)] = measurements
        return measurements

    def run_filter(*args: Any, **call_kwargs: Any) -> Any:
        if args:
            measurements = measurements_by_observation_id.get(id(args[0]))
            if measurements is not None and call_kwargs.get("joint_noise_sigma_rad") is None:
                call_kwargs["joint_noise_sigma_rad"] = measurements.joint_noise_sigma_rad
        return original_run_filter(*args, **call_kwargs)

    _base.make_synthetic_measurements = make_synthetic_measurements
    _base.run_filter = run_filter
    try:
        yield
    finally:
        _base.make_synthetic_measurements = original_make_synthetic_measurements
        _base.run_filter = original_run_filter


def evaluate_filter_with_artifacts(*args: Any, measurement_kwargs: MeasurementKwargs | None = None, **kwargs: Any) -> Any:
    with realistic_measurement_context(measurement_kwargs):
        return _base.evaluate_filter_with_artifacts(*args, **kwargs)


def ablation_rows(*args: Any, measurement_kwargs: MeasurementKwargs | None = None, **kwargs: Any) -> Any:
    with realistic_measurement_context(measurement_kwargs):
        return _base.ablation_rows(*args, **kwargs)


def robustness_rows(*args: Any, measurement_kwargs: MeasurementKwargs | None = None, **kwargs: Any) -> Any:
    with realistic_measurement_context(measurement_kwargs):
        return _base.robustness_rows(*args, **kwargs)


def trajectory_preview_rows(*args: Any, measurement_kwargs: MeasurementKwargs | None = None, **kwargs: Any) -> Any:
    with realistic_measurement_context(measurement_kwargs):
        return _base.trajectory_preview_rows(*args, **kwargs)
