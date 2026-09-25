"""Embedded-export consistency: the generated C must make the same decisions as Python."""
import numpy as np
import pytest
from sklearn.tree import DecisionTreeClassifier

from src.export_embedded import (ExportError, c_float, export, find_compiler, float32_floor,
                                 generate_header, tree_to_dict, unwrap_tree)
from test_prediction import trained  # noqa: F401  (fixture reuse)

needs_compiler = pytest.mark.skipif(find_compiler() is None, reason="no g++/clang++ on PATH")


def test_c_float_roundtrips_exactly():
    rng = np.random.default_rng(1)
    for v in rng.uniform(-1e3, 1e3, 200).astype(np.float32):
        assert np.float32(float(c_float(v)[:-1])) == v


def test_float32_floor_preserves_comparisons():
    rng = np.random.default_rng(2)
    for thr in rng.uniform(0, 100, 200):
        t32 = float32_floor(float(thr))
        assert float(t32) <= thr
        for x in np.nextafter(np.float32(thr), [np.float32(-1e9), np.float32(1e9)]).tolist() + [float(np.float32(thr))]:
            x32 = np.float32(x)
            assert (x32 <= t32) == (float(x32) <= float(thr))


def test_redundant_branches_are_collapsed():
    X = np.array([[0.0], [1.0], [2.0], [3.0]] * 5, dtype=np.float32)
    y = np.ones(20, dtype=int)
    tree = DecisionTreeClassifier(max_depth=3, random_state=0).fit(np.c_[X, X, X], y)
    d = tree_to_dict(tree, ["a", "b", "c"])
    assert d["leaf"] and d["water"] is True


@needs_compiler
def test_all_exportable_models_match_python(trained, tmp_path):  # noqa: F811
    cfg, _ = trained
    for name in ("soil_threshold_learned", "decision_tree", "logistic_regression"):
        result = export(cfg, model_name=name, out=tmp_path / f"{name}.h")
        v = result["verification"]
        assert v["n"] > 3000, name
        assert v["mismatch"] == 0, f"{name}: Python and C disagree: {v.get('examples')}"
        assert (tmp_path / f"{name}.h").exists()


@needs_compiler
def test_exported_c_agrees_on_every_adc_value(trained, tmp_path):  # noqa: F811
    """All 1024 possible raw ADC values are in the vector set, so calibration math is covered."""
    cfg, _ = trained
    result = export(cfg, model_name="decision_tree", out=tmp_path / "m.h")
    assert result["verification"]["n"] >= 1024 * 3


def test_header_is_marked_demo_and_has_expected_api(trained):  # noqa: F811
    import joblib
    import json
    cfg, meta = trained
    model = joblib.load(cfg.path("model_dir") / "candidates" / "decision_tree.joblib")
    text, summary = generate_header(cfg, meta, "decision_tree", model)
    assert "SYNTHETIC DEMO DATA" in text and "#define MODEL_IS_DEMO 1" in text
    assert "bool needsWater(" in text and "needsWaterFromRaw" in text and "soilRawToMoisturePct" in text
    assert "// check " in text                         # comments name the feature being tested
    assert f"{cfg.soil_raw_dry:g}" in text             # calibration preserved
    assert summary["type"] == "decision_tree"


def test_random_forest_is_not_exportable(trained):  # noqa: F811
    cfg, meta = trained
    with pytest.raises(ExportError, match="cannot be exported"):
        generate_header(cfg, meta, "random_forest", object())


@needs_compiler
def test_disagreement_is_caught(trained, tmp_path, monkeypatch):  # noqa: F811
    """If the generated code were wrong, export must refuse to write the header."""
    import src.export_embedded as ee
    cfg, _ = trained
    real = ee.generate_header

    def corrupt(*a, **kw):
        text, summary = real(*a, **kw)
        return text.replace("return true;", "return false;@@").replace("false;@@", "false;"), summary

    monkeypatch.setattr(ee, "generate_header", corrupt)
    with pytest.raises(ExportError, match="disagree"):
        ee.export(cfg, model_name="decision_tree", out=tmp_path / "bad.h")
    assert not (tmp_path / "bad.h").exists()
