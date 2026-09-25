"""
Predict WATER / NO_WATER for one sensor reading using the saved model.

The soil value is the RAW sensor reading; calibration is applied here the same
way as in training.

Usage:
    python -m src.predict --soil 800 --temperature 29 --humidity 60
    python -m src.predict --demo --soil 400 --temperature 29 --humidity 60
    python -m src.predict --soil 400 --temperature 29 --humidity 60 --model decision_tree
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional

import joblib

from .baseline import ThresholdBaseline
from .config import LABELS, Config, ConfigError, add_config_args, config_from_args
from .preprocessing import single_input_features


def load_model(cfg: Config, name: Optional[str] = None):
    model_dir = cfg.path("model_dir")
    path = model_dir / ("model.joblib" if name is None else f"candidates/{name}.joblib")
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python -m src.train"
                                f"{' --demo' if cfg.is_demo else ''}` first.")
    return joblib.load(path)


def predict_label(cfg: Config, soil_raw: float, temperature_c: float, humidity_pct: float,
                  model=None, model_name: Optional[str] = None) -> str:
    model = model if model is not None else load_model(cfg, model_name)
    X = single_input_features(soil_raw, temperature_c, humidity_pct, cfg)
    return LABELS[int(model.predict(X)[0])]


def baseline_label(cfg: Config, soil_raw: float, temperature_c: float, humidity_pct: float) -> str:
    X = single_input_features(soil_raw, temperature_c, humidity_pct, cfg)
    return LABELS[int(ThresholdBaseline.from_config(cfg).predict(X)[0])]


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Predict an irrigation decision for one reading.")
    p.add_argument("--soil", type=float, required=True, help="Raw soil sensor reading (ADC 0-1023)")
    p.add_argument("--temperature", type=float, required=True, help="Temperature in C")
    p.add_argument("--humidity", type=float, required=True, help="Relative humidity in %%")
    p.add_argument("--model", default=None, help="Candidate name (default: the selected model)")
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = config_from_args(args)

    try:
        model = load_model(cfg, args.model)
        X = single_input_features(args.soil, args.temperature, args.humidity, cfg)
        decision = LABELS[int(model.predict(X)[0])]
        meta_path = cfg.path("model_dir") / "metadata.json"
        meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}
        name = args.model or meta.get("selection", {}).get("chosen", "selected model")
        try:
            base = baseline_label(cfg, args.soil, args.temperature, args.humidity)
        except ConfigError:
            base = "n/a (no threshold configured)"
    except (FileNotFoundError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    if cfg.is_demo:
        print("*** SYNTHETIC DEMO model - not a real irrigation decision ***")
    print(f"Inputs:   soil_raw={args.soil:g}  temperature={args.temperature:g} C  humidity={args.humidity:g} %")
    print(f"Features: " + ", ".join(f"{n}={v:.2f}" for n, v in zip(cfg.feature_names, X[0])))
    print(f"Model ({name}): {decision}")
    print(f"Threshold baseline:  {base}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
