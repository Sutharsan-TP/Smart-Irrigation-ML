"""
Labeling utilities.

Labels must come from a rule the TEAM decides and documents, never from the
model. Two ways to get labels into the CSVs:

1. manual:          pass --label to the logger for a session, or type labels
                    into the CSV (label_source = "manual").
2. threshold_rule:  fill EMPTY labels with the calibrated-moisture threshold
                    rule below (label_source = "threshold_rule").

      if calibrated_moisture_pct < moisture_threshold_pct: WATER
      else:                                                NO_WATER

The threshold is read from config/settings.json and must be chosen by the
team after calibration; this code never invents one.

Caution: if every label comes from the threshold rule, the threshold baseline
is correct by construction and no ML model can "beat" it. Labels that use
independent evidence (visual/finger check, plant condition, the team's
watering judgement) are needed for ML to learn anything beyond the threshold.

Usage:
    python -m src.labeling data/raw/session_x.csv            # fill empty labels in place
    python -m src.labeling data/raw/session_x.csv --overwrite  # relabel every row
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .config import (LABEL_NO_WATER, LABEL_WATER, Config, ConfigError, add_config_args,
                     config_from_args)
from .preprocessing import normalise_label, raw_to_moisture_pct


def threshold_label(moisture_pct, threshold_pct: float) -> np.ndarray:
    moisture_pct = np.asarray(moisture_pct, dtype=np.float32)
    return np.where(moisture_pct < np.float32(threshold_pct), LABEL_WATER, LABEL_NO_WATER)


def apply_threshold_labels(df: pd.DataFrame, cfg: Config, overwrite: bool = False) -> tuple[pd.DataFrame, int]:
    """Fill (or overwrite) irrigation_label using the configured threshold rule."""
    dry, wet = cfg.require_calibration()
    threshold = cfg.require_threshold()
    df = df.copy()
    for col in ("irrigation_label", "label_source"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str)

    has_label = df["irrigation_label"].map(normalise_label).notna()
    target = np.ones(len(df), bool) if overwrite else ~has_label.to_numpy()
    soil = pd.to_numeric(df["soil_raw"], errors="coerce")
    target &= soil.notna().to_numpy()

    pct = raw_to_moisture_pct(soil[target].to_numpy(), dry, wet)
    df.loc[target, "irrigation_label"] = threshold_label(pct, threshold)
    df.loc[target, "label_source"] = "threshold_rule"
    return df, int(target.sum())


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Fill irrigation labels with the team's threshold rule.")
    p.add_argument("csv", type=Path, nargs="+")
    p.add_argument("--overwrite", action="store_true", help="Relabel rows that already have a label")
    p.add_argument("--output", type=Path, default=None, help="Write here instead of in place (single file)")
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = config_from_args(args)

    if args.output and len(args.csv) > 1:
        print("--output only works with a single input file", file=sys.stderr)
        return 2
    for path in args.csv:
        try:
            df, n = apply_threshold_labels(pd.read_csv(path), cfg, args.overwrite)
        except ConfigError as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            return 1
        out = args.output or path
        df.to_csv(out, index=False)
        print(f"{path.name}: labeled {n} row(s) with threshold rule "
              f"(moisture < {cfg.moisture_threshold_pct}% -> WATER) -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
