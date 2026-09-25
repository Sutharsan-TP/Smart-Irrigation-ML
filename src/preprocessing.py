"""
Loading, cleaning, calibration and feature construction.

The moisture calibration is a straight line through two points measured by the
team:  soil_raw_dry -> 0 %,  soil_raw_wet -> 100 %.  Nothing assumes that a
higher raw value means wetter (or drier) soil: the direction comes from which
calibration value is larger.

The same arithmetic is emitted into the Arduino header by export_embedded.py,
and it is done in 32-bit float here because an Arduino Uno `float` (and even
`double`) is 32-bit. That keeps Python and the microcontroller bit-compatible.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd

from .config import (LABEL_NO_WATER, LABEL_WATER, LABELS, REQUIRED_COLUMNS, Config,
                     ConfigError)


class DataError(ValueError):
    """Raised when the dataset cannot be used for the requested step."""


@dataclass
class CleaningReport:
    rows_loaded: int = 0
    rows_kept: int = 0
    dropped: dict = field(default_factory=dict)
    files: list = field(default_factory=list)

    def drop(self, reason: str, n: int) -> None:
        if n:
            self.dropped[reason] = self.dropped.get(reason, 0) + int(n)

    def as_dict(self) -> dict:
        return {"rows_loaded": self.rows_loaded, "rows_kept": self.rows_kept,
                "dropped": dict(self.dropped), "files": list(self.files)}


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------

def raw_to_moisture_pct(soil_raw, soil_raw_dry: float, soil_raw_wet: float) -> np.ndarray:
    """
    Linear calibration, clipped to [0, 100], computed in float32 in the same
    operation order as the generated C code:
        pct = (dry - raw) / (dry - wet) * 100
    """
    raw = np.asarray(soil_raw, dtype=np.float32)
    dry = np.float32(soil_raw_dry)
    wet = np.float32(soil_raw_wet)
    pct = (dry - raw) / (dry - wet) * np.float32(100.0)
    return np.clip(pct, np.float32(0.0), np.float32(100.0)).astype(np.float32)


# ---------------------------------------------------------------------------
# Loading and cleaning
# ---------------------------------------------------------------------------

def normalise_label(value) -> Optional[str]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    text = str(value).strip().upper().replace(" ", "_").replace("-", "_")
    aliases = {"WATER": LABEL_WATER, "1": LABEL_WATER, "YES": LABEL_WATER, "ON": LABEL_WATER,
               "NO_WATER": LABEL_NO_WATER, "NOWATER": LABEL_NO_WATER, "0": LABEL_NO_WATER,
               "NO": LABEL_NO_WATER, "OFF": LABEL_NO_WATER}
    return aliases.get(text)


def list_csv_files(data_dir: Path) -> list[Path]:
    return sorted(p for p in Path(data_dir).glob("*.csv") if p.is_file())


def load_raw(paths: Iterable[Path]) -> pd.DataFrame:
    """Read and concatenate CSVs. session_id defaults to the file name."""
    frames = []
    for path in paths:
        df = pd.read_csv(path)
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise DataError(f"{path.name} is missing required columns {missing}")
        if "session_id" not in df.columns:
            df["session_id"] = path.stem
        df["session_id"] = df["session_id"].fillna(path.stem).astype(str)
        df["source_file"] = path.name
        frames.append(df)
    if not frames:
        raise DataError("No CSV files found.")
    return pd.concat(frames, ignore_index=True)


def clean(df: pd.DataFrame, cfg: Config, report: Optional[CleaningReport] = None,
          require_labels: bool = True) -> pd.DataFrame:
    """
    Validate and clean raw rows. Every dropped row is counted in `report`
    under a reason, so the dataset summary can show what happened.
    """
    report = report or CleaningReport()
    report.rows_loaded += len(df)
    df = df.copy()

    for col in ("plant_id", "soil_condition", "label_source"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    if not cfg.is_demo:
        demo_rows = df["plant_id"].str.upper().str.startswith("DEMO") | \
            df["label_source"].str.contains("synthetic", case=False)
        if demo_rows.any():
            raise DataError(
                f"{int(demo_rows.sum())} rows look like SYNTHETIC demo data "
                "(plant_id DEMO_* / label_source synthetic). Demo data must not be "
                "mixed with real data. Use --demo for demo runs.")

    for col in ("soil_raw", "temperature_c", "humidity_pct"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")

    bad = df[["timestamp", "soil_raw", "temperature_c", "humidity_pct"]].isna().any(axis=1)
    report.drop("missing or non-numeric sensor value / timestamp", bad.sum())
    df = df[~bad]

    soil_bad = (df["soil_raw"] < 0) | (df["soil_raw"] > cfg.adc_max)
    report.drop(f"soil_raw outside 0..{cfg.adc_max} (impossible for a 10-bit Uno ADC)", soil_bad.sum())
    df = df[~soil_bad]

    for col in ("temperature_c", "humidity_pct"):
        lo, hi = cfg.valid_ranges.get(col, (-np.inf, np.inf))
        out = (df[col] < lo) | (df[col] > hi)
        report.drop(f"{col} outside [{lo}, {hi}]", out.sum())
        df = df[~out]

    dup = df.duplicated(subset=["timestamp", "session_id", "soil_raw", "temperature_c", "humidity_pct"])
    report.drop("exact duplicate row", dup.sum())
    df = df[~dup]

    if "irrigation_label" in df.columns:
        df["irrigation_label"] = df["irrigation_label"].map(normalise_label)
    else:
        df["irrigation_label"] = None
    if require_labels:
        unlabeled = df["irrigation_label"].isna()
        report.drop("missing / unrecognised irrigation_label", unlabeled.sum())
        df = df[~unlabeled]

    df = df.sort_values(["timestamp", "session_id"]).reset_index(drop=True)
    report.rows_kept = len(df)
    return df


def add_features(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    df = df.copy()
    if cfg.needs_calibration:
        dry, wet = cfg.require_calibration()
        df["moisture_pct"] = raw_to_moisture_pct(df["soil_raw"].to_numpy(), dry, wet)
    elif cfg.soil_raw_dry is not None and cfg.soil_raw_wet is not None:
        df["moisture_pct"] = raw_to_moisture_pct(df["soil_raw"].to_numpy(), cfg.soil_raw_dry, cfg.soil_raw_wet)
    return df


def feature_matrix(df: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Features in the fixed order cfg.feature_names, as float32."""
    return df[cfg.feature_names].to_numpy(dtype=np.float32)


def encode_labels(labels) -> np.ndarray:
    return np.array([LABELS.index(l) for l in labels], dtype=int)


def load_dataset(cfg: Config, paths: Optional[list[Path]] = None) -> tuple[pd.DataFrame, CleaningReport]:
    """Load every CSV in the configured data dir, clean it and add features."""
    paths = paths or list_csv_files(cfg.path("data_dir"))
    if not paths:
        where = cfg.path("data_dir")
        raise DataError(
            f"No CSV files in {where}. Collect real data with `python -m src.data_collection` "
            "first, or use --demo to test the software on synthetic data.")
    report = CleaningReport(files=[p.name for p in paths])
    df = clean(load_raw(paths), cfg, report)
    if df.empty:
        raise DataError(f"No usable rows after cleaning: {report.dropped}")
    return add_features(df, cfg), report


def single_input_features(soil_raw: float, temperature_c: float, humidity_pct: float,
                          cfg: Config) -> np.ndarray:
    """Preprocess one reading exactly as during training (used by predict)."""
    row = pd.DataFrame({"soil_raw": [soil_raw], "temperature_c": [temperature_c],
                        "humidity_pct": [humidity_pct]})
    return feature_matrix(add_features(row, cfg), cfg)


# ---------------------------------------------------------------------------
# Train / test split (leakage-aware)
# ---------------------------------------------------------------------------

@dataclass
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    notes: list


def split_dataset(df: pd.DataFrame, cfg: Config) -> SplitResult:
    """
    Readings within one collection session are strongly correlated (one
    reading per minute from the same pot). A random row-wise split would put
    near-identical rows on both sides and inflate the scores, so:

      * group: whole sessions go to either train or test
               (used when there are enough sessions);
      * time:  the chronologically last `test_size` fraction is the test set;
      * random: stratified row-wise split. Only used if explicitly requested.
    """
    from sklearn.model_selection import StratifiedGroupKFold, train_test_split

    split_cfg = cfg.split
    strategy = split_cfg.get("strategy", "auto")
    test_size = float(split_cfg.get("test_size", 0.25))
    min_sessions = int(split_cfg.get("min_sessions_for_group_split", 5))
    y = encode_labels(df["irrigation_label"])
    groups = df["session_id"].to_numpy()
    n_sessions = len(np.unique(groups))
    notes: list[str] = []

    if strategy == "auto":
        strategy = "group" if n_sessions >= min_sessions else "time"
        if strategy == "time":
            notes.append(f"Only {n_sessions} session(s) (< {min_sessions}); used a time-based split "
                         "instead of a session-based split.")

    idx = np.arange(len(df))
    if strategy == "group":
        n_splits = max(2, int(round(1 / test_size)))
        sgkf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=cfg.random_seed)
        train_idx, test_idx = next(sgkf.split(idx, y, groups))
    elif strategy == "time":
        order = np.argsort(df["timestamp"].to_numpy(), kind="stable")
        cut = int(round(len(df) * (1 - test_size)))
        train_idx, test_idx = np.sort(order[:cut]), np.sort(order[cut:])
    elif strategy == "random":
        stratify = y if np.bincount(y, minlength=2).min() >= 2 else None
        train_idx, test_idx = train_test_split(idx, test_size=test_size, random_state=cfg.random_seed,
                                               stratify=stratify)
        notes.append("Random row-wise split: correlated readings from the same session can appear "
                     "in both train and test, so test scores are likely optimistic.")
    else:
        raise ConfigError(f"Unknown split strategy {strategy!r}")

    if len(np.unique(y[train_idx])) < 2:
        raise DataError("The training split contains only one class; collect data for both "
                        "WATER and NO_WATER conditions (or more sessions) before training.")
    if len(np.unique(y[test_idx])) < 2:
        notes.append("The test split contains only one class; precision/recall for the missing "
                     "class are undefined and the test is weak.")
    return SplitResult(np.asarray(train_idx), np.asarray(test_idx), strategy, notes)
