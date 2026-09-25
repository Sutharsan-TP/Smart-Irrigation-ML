"""
Threshold baseline: the reference controller every ML model is compared to.

    soil moisture only, fixed team-chosen threshold, deterministic:
        WATER  if calibrated moisture_pct < moisture_threshold_pct
        NO_WATER otherwise

It is written with the scikit-learn predict() interface so the evaluation
code can treat it like any other model. It is never fitted to data.
"""

from __future__ import annotations

import numpy as np

from .config import Config


class ThresholdBaseline:
    name = "threshold_baseline"

    def __init__(self, threshold_pct: float, soil_feature: str = "moisture_pct",
                 soil_raw_dry: float | None = None, soil_raw_wet: float | None = None):
        self.threshold_pct = float(threshold_pct)
        self.soil_feature = soil_feature
        self.soil_raw_dry = soil_raw_dry
        self.soil_raw_wet = soil_raw_wet

    @classmethod
    def from_config(cls, cfg: Config) -> "ThresholdBaseline":
        dry = wet = None
        if cfg.soil_feature == "soil_raw":
            dry, wet = cfg.require_calibration()
        return cls(cfg.require_threshold(), cfg.soil_feature, dry, wet)

    def fit(self, X, y=None):  # nothing to learn; kept for interface compatibility
        return self

    def _moisture(self, X) -> np.ndarray:
        soil = np.asarray(X, dtype=np.float32)[:, 0]
        if self.soil_feature == "moisture_pct":
            return soil
        from .preprocessing import raw_to_moisture_pct
        return raw_to_moisture_pct(soil, self.soil_raw_dry, self.soil_raw_wet)

    def predict(self, X) -> np.ndarray:
        return (self._moisture(X) < np.float32(self.threshold_pct)).astype(int)
