"""Typed config helpers for synthetic measurement realism settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

_PROBABILITY_KEYS = {
    "occlusion_entry_prob",
    "occlusion_recovery_prob",
    "outlier_prob",
}
_OPTIONAL_POSITIVE_KEYS = {
    "outlier_noise_deg",
    "confidence_noise_min_deg",
    "confidence_noise_max_deg",
}
_ALLOWED_OCCLUSION_MODELS = {"iid", "independent", "bernoulli", "markov", "bursty", "temporal"}
_ALLOWED_OUTLIER_MODES = {"uniform", "large_noise", "heavy_noise"}


def _optional_float(config: Mapping[str, Any], key: str) -> float | None:
    value = config.get(key)
    if value is None:
        return None
    return float(value)


def _validate_probability(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def _validate_positive(name: str, value: float | None) -> float | None:
    if value is None:
        return None
    value = float(value)
    if value <= 0.0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


@dataclass(frozen=True)
class MeasurementRealismConfig:
    """Configuration for realistic synthetic SO(3)^K measurements.

    The all-default instance reproduces the original clean synthetic benchmark:
    IID occlusion, no explicit outliers, and scalar measurement noise. Non-default
    fields are forwarded directly to :func:`pose_filter.measurements.make_synthetic_measurements`.
    """

    occlusion_model: str = "iid"
    occlusion_entry_prob: float | None = None
    occlusion_recovery_prob: float | None = None
    outlier_prob: float = 0.0
    outlier_mode: str = "uniform"
    outlier_noise_deg: float | None = None
    confidence_calibrated_noise: bool = False
    confidence_noise_min_deg: float | None = None
    confidence_noise_max_deg: float | None = None
    confidence_noise_gamma: float = 1.0

    def __post_init__(self) -> None:
        occlusion_model = self.occlusion_model.strip().lower().replace("-", "_")
        if occlusion_model not in _ALLOWED_OCCLUSION_MODELS:
            allowed = ", ".join(sorted(_ALLOWED_OCCLUSION_MODELS))
            raise ValueError(f"occlusion_model must be one of {allowed}, got {self.occlusion_model!r}")
        object.__setattr__(self, "occlusion_model", occlusion_model)

        outlier_mode = self.outlier_mode.strip().lower().replace("-", "_")
        if outlier_mode not in _ALLOWED_OUTLIER_MODES:
            allowed = ", ".join(sorted(_ALLOWED_OUTLIER_MODES))
            raise ValueError(f"outlier_mode must be one of {allowed}, got {self.outlier_mode!r}")
        object.__setattr__(self, "outlier_mode", outlier_mode)

        for key in _PROBABILITY_KEYS:
            object.__setattr__(self, key, _validate_probability(key, getattr(self, key)))
        for key in _OPTIONAL_POSITIVE_KEYS:
            object.__setattr__(self, key, _validate_positive(key, getattr(self, key)))

        gamma = float(self.confidence_noise_gamma)
        if gamma <= 0.0:
            raise ValueError(f"confidence_noise_gamma must be positive, got {gamma}")
        object.__setattr__(self, "confidence_noise_gamma", gamma)

    @classmethod
    def from_mapping(cls, config: Mapping[str, Any] | None) -> "MeasurementRealismConfig":
        if config is None:
            return cls()
        return cls(
            occlusion_model=str(config.get("occlusion_model", "iid")),
            occlusion_entry_prob=_optional_float(config, "occlusion_entry_prob"),
            occlusion_recovery_prob=_optional_float(config, "occlusion_recovery_prob"),
            outlier_prob=float(config.get("outlier_prob", 0.0)),
            outlier_mode=str(config.get("outlier_mode", "uniform")),
            outlier_noise_deg=_optional_float(config, "outlier_noise_deg"),
            confidence_calibrated_noise=bool(config.get("confidence_calibrated_noise", False)),
            confidence_noise_min_deg=_optional_float(config, "confidence_noise_min_deg"),
            confidence_noise_max_deg=_optional_float(config, "confidence_noise_max_deg"),
            confidence_noise_gamma=float(config.get("confidence_noise_gamma", 1.0)),
        )

    @classmethod
    def coerce(
        cls,
        value: "MeasurementRealismConfig | Mapping[str, Any] | None",
    ) -> "MeasurementRealismConfig":
        if isinstance(value, MeasurementRealismConfig):
            return value
        return cls.from_mapping(value)

    @property
    def enabled(self) -> bool:
        return self.to_kwargs() != {}

    def to_kwargs(self) -> dict[str, Any]:
        """Return non-default kwargs for ``make_synthetic_measurements``."""
        kwargs: dict[str, Any] = {}
        if self.occlusion_model != "iid":
            kwargs["occlusion_model"] = self.occlusion_model
        if self.occlusion_entry_prob is not None:
            kwargs["occlusion_entry_prob"] = self.occlusion_entry_prob
        if self.occlusion_recovery_prob is not None:
            kwargs["occlusion_recovery_prob"] = self.occlusion_recovery_prob
        if self.outlier_prob != 0.0:
            kwargs["outlier_prob"] = self.outlier_prob
        if self.outlier_mode != "uniform":
            kwargs["outlier_mode"] = self.outlier_mode
        if self.outlier_noise_deg is not None:
            kwargs["outlier_noise_deg"] = self.outlier_noise_deg
        if self.confidence_calibrated_noise:
            kwargs["confidence_calibrated_noise"] = True
        if self.confidence_noise_min_deg is not None:
            kwargs["confidence_noise_min_deg"] = self.confidence_noise_min_deg
        if self.confidence_noise_max_deg is not None:
            kwargs["confidence_noise_max_deg"] = self.confidence_noise_max_deg
        if self.confidence_noise_gamma != 1.0:
            kwargs["confidence_noise_gamma"] = self.confidence_noise_gamma
        return kwargs

    def to_summary(
        self,
        *,
        noise_deg: float,
        occlusion_prob: float,
        confidence_noise_std: float = 0.0,
        min_confidence: float = 0.2,
    ) -> dict[str, Any]:
        return {
            "noise_deg": float(noise_deg),
            "occlusion_prob": float(occlusion_prob),
            "confidence_noise_std": float(confidence_noise_std),
            "min_confidence": float(min_confidence),
            **asdict(self),
        }


MeasurementConfigLike = MeasurementRealismConfig | Mapping[str, Any] | None


def measurement_realism_kwargs(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible helper returning non-default measurement kwargs."""
    return MeasurementRealismConfig.from_mapping(config).to_kwargs()


def measurement_realism_summary(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Backward-compatible summary helper without base noise/confidence fields."""
    cfg = MeasurementRealismConfig.from_mapping(config)
    return asdict(cfg)
