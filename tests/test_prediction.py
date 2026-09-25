import numpy as np
import pytest

from src.baseline import ThresholdBaseline
from src.config import load_config
from src.make_demo_data import generate
from src.predict import baseline_label, predict_label
from src.train import train


@pytest.fixture(scope="module")
def trained(tmp_path_factory):
    """Train on freshly generated SYNTHETIC data inside a temp dir."""
    root = tmp_path_factory.mktemp("demo_run")
    cfg = load_config(demo=True)
    cfg.paths = {"data_dir": str(root / "data"), "processed_dir": str(root / "processed"),
                 "model_dir": str(root / "models"), "report_dir": str(root / "reports"),
                 "firmware_model_header": str(root / "models" / "model.h")}
    generate(root / "data")
    meta = train(cfg)
    return cfg, meta


def test_baseline_threshold_semantics():
    b = ThresholdBaseline(35.0)
    X = np.array([[10.0, 25, 60], [34.99, 25, 60], [35.0, 25, 60], [90.0, 25, 60]], dtype=np.float32)
    assert b.predict(X).tolist() == [1, 1, 0, 0]          # WATER strictly below the threshold


def test_baseline_ignores_temperature_and_humidity():
    b = ThresholdBaseline(35.0)
    lo = np.array([[20.0, 5, 10]], dtype=np.float32)
    hi = np.array([[20.0, 45, 95]], dtype=np.float32)
    assert b.predict(lo)[0] == b.predict(hi)[0] == 1


def test_baseline_can_work_from_raw_soil_feature():
    b = ThresholdBaseline(50.0, soil_feature="soil_raw", soil_raw_dry=600, soil_raw_wet=280)
    assert b.predict(np.array([[580.0, 0, 0], [300.0, 0, 0]], dtype=np.float32)).tolist() == [1, 0]


def test_training_is_reproducible(trained):
    cfg, meta = trained
    again = train(cfg)
    assert again["split"]["test_sessions"] == meta["split"]["test_sessions"]
    assert again["cross_validation"]["results"] == meta["cross_validation"]["results"]


def test_metadata_records_what_reproduction_needs(trained):
    _, meta = trained
    assert meta["data_kind"] == "SYNTHETIC_DEMO"
    assert meta["feature_names"] == ["moisture_pct", "temperature_c", "humidity_pct"]
    assert meta["config"]["random_seed"] == 42
    assert meta["split"]["train_rows"] + meta["split"]["test_rows"] == meta["dataset"]["rows"]
    assert set(meta["split"]["train_sessions"]).isdisjoint(meta["split"]["test_sessions"])
    assert meta["data_files"] and all(len(h) == 16 for h in meta["data_files"].values())


def test_predict_dry_and_wet_extremes(trained):
    cfg, _ = trained
    assert predict_label(cfg, 590, 27, 75, model_name="decision_tree") == "WATER"      # near dry calibration
    assert predict_label(cfg, 290, 27, 75, model_name="decision_tree") == "NO_WATER"   # near wet calibration
    assert baseline_label(cfg, 590, 27, 75) == "WATER"


def test_predict_returns_valid_label_for_every_candidate(trained):
    cfg, _ = trained
    for name in ("soil_threshold_learned", "logistic_regression", "decision_tree", "random_forest"):
        assert predict_label(cfg, 450, 28, 70, model_name=name) in ("WATER", "NO_WATER")


def test_selected_model_saved_and_loadable(trained):
    cfg, meta = trained
    assert (cfg.path("model_dir") / "model.joblib").exists()
    assert predict_label(cfg, 590, 27, 75) in ("WATER", "NO_WATER")
    assert meta["selection"]["chosen"] in meta["cross_validation"]["results"]
