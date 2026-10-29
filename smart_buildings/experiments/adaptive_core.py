"""Pure control helpers. Kelvin units; decisions every five minutes.

These functions contain no simulator imports and never use hidden future state.
The trend estimate is deliberately bounded, not a calibrated thermal model.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite


@dataclass(frozen=True)
class AdaptiveConfig:
    horizon_steps: int = 3
    max_observed_delta_k: float = 0.08
    max_forecast_delta_k: float = 0.24
    forecast_weight: float = 0.30
    outlier_activation_k: float = 0.60
    max_outlier_weight: float = 0.65
    outlier_ramp_k: float = 3.0
    damper_full_k: float = 2.0
    reheat_full_k: float = 2.0

    def __post_init__(self) -> None:
        if not isinstance(self.horizon_steps, int) or self.horizon_steps < 0:
            raise ValueError("horizon_steps must be a nonnegative integer")
        for key in ("max_observed_delta_k", "max_forecast_delta_k",
                    "outlier_activation_k", "outlier_ramp_k",
                    "damper_full_k", "reheat_full_k"):
            v = getattr(self, key)
            if not isfinite(v) or v <= 0:
                raise ValueError(f"{key} must be positive and finite")
        for key in ("forecast_weight", "max_outlier_weight"):
            v = getattr(self, key)
            if not isfinite(v) or not 0 <= v <= 1:
                raise ValueError(f"{key} must be in [0,1]")


def deadband_error(temp: float, lower: float, upper: float) -> float:
    """Signed demand: positive = too cold; negative = too hot."""
    if not all(isfinite(v) for v in (temp, lower, upper)) or lower >= upper:
        raise ValueError("Invalid temperature or comfort band")
    return max(0.0, lower - temp) - max(0.0, temp - upper)


def projected_temperature(current: float, previous: float | None,
                          config: AdaptiveConfig) -> float:
    if not isfinite(current):
        raise ValueError("Nonfinite zone temperature")
    if previous is None or not isfinite(previous):
        return float(current)
    delta = max(-config.max_observed_delta_k,
                min(config.max_observed_delta_k, current - previous))
    projection = max(-config.max_forecast_delta_k,
                     min(config.max_forecast_delta_k,
                         config.horizon_steps * delta))
    return float(current + projection)


def zone_request(temp: float, previous: float | None, lower: float,
                 upper: float, config: AdaptiveConfig) -> float:
    now = deadband_error(temp, lower, upper)
    future = deadband_error(projected_temperature(temp, previous, config),
                            lower, upper)
    return (1 - config.forecast_weight) * now + config.forecast_weight * future


def aggregate_demand(requests: list[float], config: AdaptiveConfig) -> float:
    """Blend signed mean demand with the largest excursion.

    This can be worse than the reference controller when zones disagree.
    Quantifying that risk is a required experimental ablation.
    """
    if not requests:
        return 0.0
    if not all(isfinite(v) for v in requests):
        raise ValueError("Nonfinite control demand")
    mean = sum(requests) / len(requests)
    worst = max(requests, key=abs)
    weight = min(config.max_outlier_weight,
                 config.max_outlier_weight *
                 max(0.0, abs(worst) - config.outlier_activation_k)
                 / config.outlier_ramp_k)
    return (1 - weight) * mean + weight * worst


def zone_commands(demand: float, config: AdaptiveConfig) -> tuple[float, float]:
    """Normalized damper, reheat valve commands within [-1,1]."""
    if not isfinite(demand):
        raise ValueError("Nonfinite zone demand")
    damper = min(1.0, abs(demand) / config.damper_full_k)
    reheat = min(1.0, max(0.0, demand) / config.reheat_full_k)
    return 2 * damper - 1, 2 * reheat - 1
