import numpy as np
import pandas as pd
import pytest

from src.config import ConfigError, load_config
from src.preprocessing import (DataError, CleaningReport, add_features, clean, feature_matrix,
                               normalise_label, raw_to_moisture_pct, split_dataset)


@pytest.fixture
def demo_cfg():
    return load_config(demo=True)


@pytest.fixture
def real_cfg():
    return load_config(demo=False)


def rows(n=4, **overrides):
    base = {
        "timestamp": pd.date_range("2026-09-25 10:00", periods=n, freq="min").astype(str),
        "soil_raw": [500] * n, "temperature_c": [29.0] * n, "humidity_pct": [60.0] * n,
        "plant_id": ["plant_1"] * n, "irrigation_label": ["WATER"] * n, "session_id": ["s1"] * n,
    }
    base.update(overrides)
    return pd.DataFrame(base)


# ---- calibration -----------------------------------------------------------

def test_calibration_dry_above_wet():
    pct = raw_to_moisture_pct([600, 440, 280], soil_raw_dry=600, soil_raw_wet=280)
    assert pct == pytest.approx([0, 50, 100])


def test_calibration_direction_comes_from_config_not_assumption():
    # A sensor whose raw value RISES with wetness must also work.
    pct = raw_to_moisture_pct([200, 500, 800], soil_raw_dry=200, soil_raw_wet=800)
    assert pct == pytest.approx([0, 50, 100])


def test_calibration_clips_out_of_range():
    pct = raw_to_moisture_pct([700, 100], soil_raw_dry=600, soil_raw_wet=280)
    assert pct == pytest.approx([0, 100])


def test_calibration_is_float32():
    assert raw_to_moisture_pct([500], 600, 280).dtype == np.float32


def test_missing_calibration_is_an_error(real_cfg):
    assert real_cfg.soil_raw_dry is None  # the real config ships unfilled on purpose
    with pytest.raises(ConfigError, match="calibration"):
        add_features(rows(), real_cfg)


def test_missing_threshold_is_an_error(real_cfg):
    with pytest.raises(ConfigError, match="threshold"):
        real_cfg.require_threshold()


# ---- cleaning ------------------------------------------------------------------

@pytest.mark.parametrize("raw, expected", [("WATER", "WATER"), ("water", "WATER"), (" no water ", "NO_WATER"),
                                           ("NO-WATER", "NO_WATER"), ("1", "WATER"), ("maybe", None),
                                           (None, None), (float("nan"), None)])
def test_normalise_label(raw, expected):
    assert normalise_label(raw) == expected


def test_clean_drops_invalid_rows_and_reports_reasons(demo_cfg):
    df = rows(6, soil_raw=[500, 1150, "abc", 500, 500, 500],
              temperature_c=[29, 29, 29, 80, 29, 29],
              irrigation_label=["WATER", "WATER", "WATER", "WATER", "???", "NO_WATER"])
    report = CleaningReport()
    out = clean(df, demo_cfg, report)
    assert len(out) == 2
    assert report.rows_loaded == 6 and report.rows_kept == 2
    reasons = " ".join(report.dropped)
    assert "soil_raw outside" in reasons      # 1150 > 1023 (10-bit ADC)
    assert "non-numeric" in reasons
    assert "temperature_c outside" in reasons
    assert "irrigation_label" in reasons


def test_clean_removes_exact_duplicates(demo_cfg):
    df = pd.concat([rows(2), rows(2)], ignore_index=True)
    assert len(clean(df, demo_cfg)) == 2


def test_real_mode_rejects_demo_rows(real_cfg):
    with pytest.raises(DataError, match="SYNTHETIC"):
        clean(rows(plant_id=["DEMO_plant_1"] * 4), real_cfg)


def test_feature_order_and_dtype(demo_cfg):
    X = feature_matrix(add_features(clean(rows(), demo_cfg), demo_cfg), demo_cfg)
    assert demo_cfg.feature_names == ["moisture_pct", "temperature_c", "humidity_pct"]
    assert X.dtype == np.float32 and X.shape == (4, 3)
    assert X[0, 1] == pytest.approx(29.0) and X[0, 2] == pytest.approx(60.0)


def test_timestamp_and_metadata_are_not_features(demo_cfg):
    for col in ("timestamp", "plant_id", "soil_condition", "session_id"):
        assert col not in demo_cfg.feature_names


# ---- split ---------------------------------------------------------------------

def make_sessions(n_sessions, per_session=10):
    frames = []
    for s in range(n_sessions):
        label = "WATER" if s % 2 else "NO_WATER"
        frames.append(rows(per_session, session_id=[f"s{s}"] * per_session,
                           timestamp=pd.date_range(f"2026-09-{s + 1:02d}", periods=per_session,
                                                   freq="min").astype(str),
                           irrigation_label=[label] * per_session))
    return pd.concat(frames, ignore_index=True)


def test_group_split_keeps_sessions_together(demo_cfg):
    df = clean(make_sessions(8), demo_cfg)
    res = split_dataset(df, demo_cfg)
    assert res.strategy == "group"
    train_s = set(df.loc[res.train_idx, "session_id"])
    test_s = set(df.loc[res.test_idx, "session_id"])
    assert train_s.isdisjoint(test_s)
    assert len(res.train_idx) + len(res.test_idx) == len(df)


def test_few_sessions_fall_back_to_time_split(demo_cfg):
    df2 = clean(make_sessions(2), demo_cfg)
    res = split_dataset(df2, demo_cfg)
    assert res.strategy == "time"
    assert df2.loc[res.train_idx, "timestamp"].max() <= df2.loc[res.test_idx, "timestamp"].min()
    assert any("time-based" in n for n in res.notes)


def test_single_class_training_set_is_rejected(demo_cfg):
    df = clean(rows(10), demo_cfg)  # all WATER
    with pytest.raises(DataError, match="one class"):
        split_dataset(df, demo_cfg)
