"""
Configuration loading.

All team-specific numbers (sensor calibration, labeling threshold) live in
config/settings.json, never in code. The demo configuration in
config/demo_settings.json contains invented placeholder values used only for
software testing.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"
REAL_CONFIG_PATH = CONFIG_DIR / "settings.json"
DEMO_CONFIG_PATH = CONFIG_DIR / "demo_settings.json"

LABEL_WATER = "WATER"
LABEL_NO_WATER = "NO_WATER"
LABELS = (LABEL_NO_WATER, LABEL_WATER)  # index == encoded class (WATER = 1 = positive)

# CSV columns written by the serial logger. Only REQUIRED_COLUMNS must exist
# for training; the rest are metadata and are never used as model features.
CSV_COLUMNS = [
    "timestamp",
    "soil_raw",
    "temperature_c",
    "humidity_pct",
    "plant_id",
    "soil_condition",
    "irrigation_label",
    "session_id",
    "label_source",
]
REQUIRED_COLUMNS = ["timestamp", "soil_raw", "temperature_c", "humidity_pct"]

SOIL_FEATURES = ("moisture_pct", "soil_raw")


class ConfigError(ValueError):
    """Raised when the configuration is missing a value the current step needs."""


@dataclass
class Config:
    soil_raw_dry: Optional[float]
    soil_raw_wet: Optional[float]
    moisture_threshold_pct: Optional[float]
    soil_feature: str = "moisture_pct"
    adc_max: int = 1023
    valid_ranges: dict = field(default_factory=dict)
    random_seed: int = 42
    split: dict = field(default_factory=dict)
    cv_folds: int = 5
    selection_metric: str = "f2"
    models: dict = field(default_factory=dict)
    paths: dict = field(default_factory=dict)
    is_demo: bool = False
    source_path: Optional[str] = None

    # ---- derived helpers -------------------------------------------------

    @property
    def feature_names(self) -> list[str]:
        """Model input order. The exported C code uses exactly this order."""
        return [self.soil_feature, "temperature_c", "humidity_pct"]

    def path(self, key: str) -> Path:
        return PROJECT_ROOT / self.paths[key]

    def require_calibration(self) -> tuple[float, float]:
        if self.soil_raw_dry is None or self.soil_raw_wet is None:
            raise ConfigError(
                "Soil calibration is not set. Measure the sensor's raw value in "
                "your dry soil and your saturated soil, then fill 'soil_raw_dry' "
                f"and 'soil_raw_wet' in {self.source_path}."
            )
        if self.soil_raw_dry == self.soil_raw_wet:
            raise ConfigError("soil_raw_dry and soil_raw_wet must differ.")
        return float(self.soil_raw_dry), float(self.soil_raw_wet)

    def require_threshold(self) -> float:
        if self.moisture_threshold_pct is None:
            raise ConfigError(
                "moisture_threshold_pct is not set. The team must choose it after "
                f"calibration and write it into {self.source_path}."
            )
        return float(self.moisture_threshold_pct)

    @property
    def needs_calibration(self) -> bool:
        return self.soil_feature == "moisture_pct"

    def to_dict(self) -> dict:
        return {
            "soil_raw_dry": self.soil_raw_dry,
            "soil_raw_wet": self.soil_raw_wet,
            "moisture_threshold_pct": self.moisture_threshold_pct,
            "soil_feature": self.soil_feature,
            "adc_max": self.adc_max,
            "valid_ranges": self.valid_ranges,
            "random_seed": self.random_seed,
            "split": self.split,
            "cv_folds": self.cv_folds,
            "selection_metric": self.selection_metric,
            "models": self.models,
            "is_demo": self.is_demo,
            "source_path": self.source_path,
        }


def load_config(path: Optional[Path | str] = None, demo: bool = False) -> Config:
    """Load settings.json (or demo_settings.json when demo=True)."""
    if path is None:
        path = DEMO_CONFIG_PATH if demo else REAL_CONFIG_PATH
    path = Path(path)
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    raw = {k: v for k, v in raw.items() if not k.startswith("_")}

    cfg = Config(**raw, source_path=str(path))

    if cfg.soil_feature not in SOIL_FEATURES:
        raise ConfigError(f"soil_feature must be one of {SOIL_FEATURES}, got {cfg.soil_feature!r}")
    if cfg.selection_metric not in ("f1", "f2", "recall", "accuracy"):
        raise ConfigError(f"Unknown selection_metric {cfg.selection_metric!r}")
    return cfg


def add_config_args(parser) -> None:
    """Shared CLI flags: --demo and --config."""
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Use the SYNTHETIC demo dataset/config (software testing only).",
    )
    parser.add_argument("--config", type=Path, default=None, help="Path to a settings JSON file.")


def config_from_args(args) -> Config:
    return load_config(args.config, demo=args.demo)
