"""
Generate a small SYNTHETIC dataset for software testing only.

!!! This is NOT experimental data. !!!
Nothing here was measured. The numbers are produced by a random simulation
so the pipeline (parsing, preprocessing, training, export, tests) can run
before real data exists. Results obtained on this data say nothing about the
real prototype.

How the demo labels are generated (so nobody mistakes them for ground truth):
    moisture_pct is computed with the demo calibration (config/demo_settings.json);
    an invented "simulated judge" says WATER when
        moisture_pct < 35 + 0.8 * (temperature_c - 28) - 0.15 * (humidity_pct - 60)
    and then 5% of labels are flipped at random.
Because this invented rule uses temperature and humidity, ML models may beat
the soil-only threshold baseline on demo data. That is an artifact of how the
demo was built, not evidence about the real system.

Usage:
    python -m src.make_demo_data
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CSV_COLUMNS, LABEL_NO_WATER, LABEL_WATER, PROJECT_ROOT, load_config

DEMO_PREFIX = "DEMO_SYNTHETIC_"
DEMO_PLANT_ID = "DEMO_plant_1"


def simulate_session(rng: np.random.Generator, session_idx: int, start: datetime,
                     n_samples: int, dry: float, wet: float) -> pd.DataFrame:
    """One simulated watering-then-drying cycle, sampled every 10 minutes."""
    minutes = np.arange(n_samples) * 10
    hours_of_day = (start.hour + minutes / 60.0) % 24

    # Soil starts near "wet" after watering and drifts toward "dry".
    start_raw = wet + rng.uniform(0.0, 0.15) * (dry - wet)
    end_raw = wet + rng.uniform(0.75, 1.0) * (dry - wet)
    progress = 1 - np.exp(-3 * np.linspace(0, 1, n_samples))
    soil = start_raw + (end_raw - start_raw) * progress / progress[-1]
    soil = np.round(soil + rng.normal(0, 6, n_samples)).astype(int)

    # Diurnal temperature (peak ~15:00) and roughly anti-correlated humidity.
    base_t = rng.uniform(24, 30)
    temp = base_t + 5 * np.sin((hours_of_day - 9) / 24 * 2 * np.pi) + rng.normal(0, 0.4, n_samples)
    hum = 85 - 1.2 * (temp - 20) + rng.normal(0, 3, n_samples)
    temp = np.round(np.clip(temp, 0, 50), 1)   # DHT11 range
    hum = np.round(np.clip(hum, 20, 95), 1)

    moisture = np.clip((dry - soil) / (dry - wet) * 100, 0, 100)
    judge_threshold = 35 + 0.8 * (temp - 28) - 0.15 * (hum - 60)
    needs_water = moisture < judge_threshold
    flip = rng.random(n_samples) < 0.05
    needs_water = np.where(flip, ~needs_water, needs_water)

    condition = np.where(moisture < 30, "dry", np.where(moisture < 65, "moist", "wet"))
    session_id = f"{DEMO_PREFIX}session_{session_idx:02d}"
    return pd.DataFrame({
        "timestamp": [(start + timedelta(minutes=int(m))).isoformat(timespec="seconds") for m in minutes],
        "soil_raw": soil,
        "temperature_c": temp,
        "humidity_pct": hum,
        "plant_id": DEMO_PLANT_ID,
        "soil_condition": condition,
        "irrigation_label": np.where(needs_water, LABEL_WATER, LABEL_NO_WATER),
        "session_id": session_id,
        "label_source": "synthetic_demo",
    })[CSV_COLUMNS]


def generate(out_dir: Path, n_sessions: int = 8, n_samples: int = 60, seed: int = 42) -> list[Path]:
    cfg = load_config(demo=True)
    dry, wet = cfg.require_calibration()
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_sessions):
        start = datetime(2026, 1, 1, int(rng.integers(6, 18))) + timedelta(days=2 * i)
        df = simulate_session(rng, i + 1, start, n_samples, dry, wet)
        path = out_dir / f"{df['session_id'].iloc[0]}.csv"
        df.to_csv(path, index=False)
        paths.append(path)
    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Write SYNTHETIC demo CSVs (software testing only).")
    p.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "demo")
    p.add_argument("--sessions", type=int, default=8)
    p.add_argument("--samples", type=int, default=60)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)
    paths = generate(args.out, args.sessions, args.samples, args.seed)
    total = sum(len(pd.read_csv(pth)) for pth in paths)
    print(f"Wrote {len(paths)} SYNTHETIC demo files ({total} rows) to {args.out}")
    print("Reminder: this is NOT experimental data.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
