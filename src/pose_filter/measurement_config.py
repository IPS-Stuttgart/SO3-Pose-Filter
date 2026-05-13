"""Config helpers for synthetic measurement realism settings."""

from __future__ import annotations

from typing import Any

_REALISTIC_MEASUREMENT_FLOAT_KEYS = (
    "occlusion_entry_prob",
    "occlusion_recovery_prob",
    "outlier_noise_deg",
    "confidence_noise_min_deg",
    "confidence_noise_max_deg",
)


def _optional_float(config: dict[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    return float(value)


def measurement_realism_kwargs(config: dict[str, Any]) -> dict[str, Any]:
    """Return synthetic measurement-generator options from an experiment config.

    Omitting all keys reproduces the previous clean synthetic benchmark: IID
    occlusion, no explicit outlier observations, and scalar measurement noise.
    Providing these keys activates the more realistic generator implemented in
    :mod:`pose_filter.measurements`.
    """

    kwargs: dict[str, Any] = {}

    occlusion_model = str(config.get("occlusion_model", "iid")).strip().lower()
    if occlusion_model and occlusion_model != "iid":
        kwargs["occlusion_model"] = occlusion_model

    for key in _REALISTIC_MEASUREMENT_FLOAT_KEYS:
        value = _optional_float(config, key)
        if value is not None:
            kwargs[key] = value

    outlier_prob = _optional_float(config, "outlier_prob")
    if outlier_prob is not None and outlier_prob != 0.0:
        kwargs["outlier_prob"] = outlier_prob

    outlier_mode = str(config.get("outlier_mode", "uniform")).strip().lower()
    if outlier_mode and outlier_mode != "uniform":
        kwargs["outlier_mode"] = outlier_mode

    if bool(config.get("confidence_calibrated_noise", False)):
        kwargs["confidence_calibrated_noise"] = True

    confidence_gamma = _optional_float(config, "confidence_noise_gamma")
    if confidence_gamma is not None and confidence_gamma != 1.0:
        kwargs["confidence_noise_gamma"] = confidence_gamma

    return kwargs


def measurement_realism_summary(config: dict[str, Any]) -> dict[str, Any]:
    """Return a JSON-friendly summary of synthetic measurement settings."""

    kwargs = measurement_realism_kwargs(config)
    return {
        "occlusion_model": str(kwargs.get("occlusion_model", "iid")),
        "occlusion_entry_prob": kwargs.get("occlusion_entry_prob"),
        "occlusion_recovery_prob": kwargs.get("occlusion_recovery_prob"),
        "outlier_prob": float(kwargs.get("outlier_prob", 0.0)),
        "outlier_noise_deg": kwargs.get("outlier_noise_deg"),
        "outlier_mode": str(kwargs.get("outlier_mode", "uniform")),
        "confidence_calibrated_noise": bool(kwargs.get("confidence_calibrated_noise", False)),
        "confidence_noise_min_deg": kwargs.get("confidence_noise_min_deg"),
        "confidence_noise_max_deg": kwargs.get("confidence_noise_max_deg"),
        "confidence_noise_gamma": float(kwargs.get("confidence_noise_gamma", 1.0)),
    }
