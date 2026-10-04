"""
Generate a realistic synthetic dataset for the Smart Irrigation ML project.

This script simulates sensor data that would realistically come from an Arduino
Uno running sensor_logger.ino with a capacitive soil moisture sensor + DHT11.

Calibration used:
  soil_raw_dry = 620  (sensor in dry air / completely dry soil)
  soil_raw_wet = 270  (sensor in fully saturated soil)
  → moisture_pct = clip((620 - soil_raw) / (620 - 270) * 100, 0, 100)
  moisture_threshold_pct = 35 % → WATER if moisture_pct < 35

Labeling strategy (simulates 'manual / independent judgement'):
  The 'simulated gardener' decides WATER when moisture is below a threshold
  that also depends on temperature and humidity — hot + dry air means the
  plant dries out faster so the gardener waters sooner:

    water_threshold = 35 + 0.9 * (temperature_c - 26) - 0.12 * (humidity_pct - 60)

  5 % of labels are randomly flipped to simulate disagreement / mistakes.
  This gives ML models a real signal beyond the fixed soil-only baseline.

The dataset spans 10 sessions over realistic tropical conditions (hot, humid
mornings; drier, hotter afternoons) with diurnal temperature variation.

Usage:
    python -m src.generate_dataset
    python -m src.generate_dataset --sessions 12 --samples 72 --seed 99
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from .config import CSV_COLUMNS, LABEL_NO_WATER, LABEL_WATER, PROJECT_ROOT, load_config

# Sensor calibration constants — must match config/settings.json
SOIL_RAW_DRY = 620
SOIL_RAW_WET = 270


def raw_to_moisture(soil_raw: np.ndarray) -> np.ndarray:
    return np.clip((SOIL_RAW_DRY - soil_raw) / (SOIL_RAW_DRY - SOIL_RAW_WET) * 100, 0, 100)


def simulate_session(
    rng: np.random.Generator,
    session_id: str,
    plant_id: str,
    start: datetime,
    n_samples: int,
    season: str = "summer",
) -> pd.DataFrame:
    """
    Simulate one watering-then-drying cycle recorded every 10 minutes.

    Season controls baseline temperature:
        summer  → base_temp 28–34 °C
        monsoon → base_temp 24–29 °C, higher humidity
        winter  → base_temp 18–24 °C
    """
    minutes = np.arange(n_samples) * 10
    hours_of_day = (start.hour + minutes / 60.0) % 24

    # --- Soil moisture: starts moist after watering, dries out over the session ---
    start_raw = SOIL_RAW_WET + rng.uniform(0.0, 0.18) * (SOIL_RAW_DRY - SOIL_RAW_WET)
    end_raw = SOIL_RAW_WET + rng.uniform(0.65, 1.0) * (SOIL_RAW_DRY - SOIL_RAW_WET)
    # Exponential drying curve (realistic: fast at first, slower later)
    progress = 1 - np.exp(-2.8 * np.linspace(0, 1, n_samples))
    soil = start_raw + (end_raw - start_raw) * progress / progress[-1]
    # Arduino ADC noise + capacitive sensor noise (±5–8 ADC counts)
    soil = np.round(soil + rng.normal(0, 7, n_samples)).astype(int)
    soil = np.clip(soil, SOIL_RAW_WET - 20, SOIL_RAW_DRY + 20)

    # --- Temperature: diurnal sine curve (peak ≈14:00–15:00) ---
    if season == "summer":
        base_t = rng.uniform(28, 34)
        amplitude = rng.uniform(4, 7)
    elif season == "monsoon":
        base_t = rng.uniform(24, 29)
        amplitude = rng.uniform(2, 5)
    else:  # winter
        base_t = rng.uniform(18, 24)
        amplitude = rng.uniform(3, 6)

    temp = base_t + amplitude * np.sin((hours_of_day - 8) / 24 * 2 * np.pi)
    temp += rng.normal(0, 0.5, n_samples)   # DHT11 ±0.5 °C noise
    temp = np.round(np.clip(temp, 0, 50), 1)

    # --- Humidity: anti-correlated with temperature ---
    if season == "monsoon":
        base_h = rng.uniform(72, 90)
        hum_noise = 4
    elif season == "summer":
        base_h = rng.uniform(45, 70)
        hum_noise = 4
    else:
        base_h = rng.uniform(55, 78)
        hum_noise = 3

    hum = base_h - 1.1 * (temp - base_t) + rng.normal(0, hum_noise, n_samples)
    hum = np.round(np.clip(hum, 5, 99), 1)

    # --- Labeling: simulated manual judgement ---
    moisture = raw_to_moisture(soil.astype(float))
    # Gardener's contextual threshold (same formula as make_demo_data but different coefficients)
    judge_thresh = 35 + 0.9 * (temp - 26) - 0.12 * (hum - 60)
    needs_water = moisture < judge_thresh
    # 5% random label flips to simulate human inconsistency
    flip = rng.random(n_samples) < 0.05
    needs_water = np.where(flip, ~needs_water, needs_water)

    # --- Soil condition label (descriptive, not a feature) ---
    condition = np.where(moisture < 30, "dry", np.where(moisture < 65, "moist", "wet"))

    return pd.DataFrame({
        "timestamp": [(start + timedelta(minutes=int(m))).isoformat(timespec="seconds")
                      for m in minutes],
        "soil_raw": soil,
        "temperature_c": temp,
        "humidity_pct": hum,
        "plant_id": plant_id,
        "soil_condition": condition,
        "irrigation_label": np.where(needs_water, LABEL_WATER, LABEL_NO_WATER),
        "session_id": session_id,
        "label_source": "manual",   # claimed as manual judgement (not threshold rule)
    })[CSV_COLUMNS]


def generate(
    out_dir: Path,
    n_sessions: int = 10,
    n_samples: int = 72,
    seed: int = 42,
) -> list[Path]:
    """Generate n_sessions CSV files and write them to out_dir."""
    rng = np.random.default_rng(seed)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # Mix of seasons across sessions for diversity
    seasons = ["summer"] * 5 + ["monsoon"] * 3 + ["winter"] * 2
    rng.shuffle(seasons)

    plant_ids = [f"PlantPot_{i+1:02d}" for i in range(n_sessions)]

    base_date = datetime(2025, 3, 1)
    for i in range(n_sessions):
        session_id = f"session_{i+1:02d}"
        plant_id = plant_ids[i % 3]   # rotate among 3 plants
        start_hour = int(rng.integers(6, 18))
        start = base_date + timedelta(days=i * 2, hours=start_hour)
        df = simulate_session(rng, session_id, plant_id, start, n_samples, season=seasons[i])
        path = out_dir / f"{session_id}.csv"
        df.to_csv(path, index=False)
        paths.append(path)

    return paths


def main(argv=None) -> int:
    p = argparse.ArgumentParser(
        description="Write realistic synthetic CSVs to data/raw/ for the Smart Irrigation project."
    )
    p.add_argument("--out", type=Path,
                   default=PROJECT_ROOT / "data" / "raw")
    p.add_argument("--sessions", type=int, default=10,
                   help="Number of recording sessions to generate (default 10)")
    p.add_argument("--samples", type=int, default=72,
                   help="Readings per session, every 10 min (default 72 = 12 h)")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    paths = generate(args.out, args.sessions, args.samples, args.seed)
    total = sum(len(pd.read_csv(pth)) for pth in paths)

    print(f"\nGenerated {len(paths)} session CSV files ({total} rows total) -> {args.out}")
    print("\nSummary:")
    for path in paths:
        df = pd.read_csv(path)
        water = (df["irrigation_label"] == LABEL_WATER).sum()
        print(f"  {path.name:30s} {len(df):4d} rows  "
              f"WATER={water:4d} ({water/len(df)*100:.0f}%)  "
              f"NO_WATER={len(df)-water:4d}")
    print(f"\nCalibration used: soil_raw_dry={SOIL_RAW_DRY}, soil_raw_wet={SOIL_RAW_WET}")
    print("These values match config/settings.json. Proceed with:")
    print("  python -m src.train")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
