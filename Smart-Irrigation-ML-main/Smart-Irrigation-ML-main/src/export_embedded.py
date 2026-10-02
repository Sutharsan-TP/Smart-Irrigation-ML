"""
Export a trained model as plain Arduino-compatible C/C++.

Supported models
  decision_tree / soil_threshold_learned -> nested `if` statements
  logistic_regression                   -> z = b + w1*x1 + w2*x2 + w3*x3 ; WATER if z > 0
                                            (the StandardScaler is folded into w and b)
  random_forest                         -> not exported (see README)

The generated header contains:
  * the soil calibration constants and soilRawToMoisturePct(), identical to
    src/preprocessing.py (same float32 arithmetic, same operation order);
  * bool needsWater(soil, temperatureC, humidityPct);
  * bool needsWaterFromRaw(int soilRaw, temperatureC, humidityPct).

Arduino Uno note: `float` and `double` are both 32-bit on AVR. scikit-learn
compares float32 features against float64 thresholds, so each threshold is
emitted as the largest float32 <= the float64 threshold. For any float32
input x, (x <= t32) == (x <= t64), so the C code makes exactly the same
decision as Python.

After writing the header, the export compiles it with g++ (if available) and
checks that C and Python agree on every stored train/test row plus random and
near-threshold inputs.

Usage:
    python -m src.export_embedded                # selected model -> firmware/.../model.h
    python -m src.export_embedded --model decision_tree
    python -m src.export_embedded --demo         # writes models/demo/model.h (never into firmware/)
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from .config import Config, add_config_args, config_from_args
from .models import EXPORTABLE, FeatureSubset
from .preprocessing import add_features, feature_matrix

C_PARAM_NAMES = {"moisture_pct": "soilMoisturePct", "soil_raw": "soilRaw",
                 "temperature_c": "temperatureC", "humidity_pct": "humidityPct"}


class ExportError(RuntimeError):
    pass


# ---------------------------------------------------------------------------
# Number formatting
# ---------------------------------------------------------------------------

def float32_floor(value: float) -> np.float32:
    """Largest float32 that is <= value (so `x <= result` == `x <= value` for float32 x)."""
    f = np.float32(value)
    if float(f) > value:
        f = np.nextafter(f, np.float32(-np.inf))
    return f


def c_float(value) -> str:
    """A C float literal that parses back to exactly this float32."""
    text = f"{float(np.float32(value)):.9g}"
    if not any(ch in text for ch in ".eEn"):
        text += ".0"
    return text + "f"


# ---------------------------------------------------------------------------
# Model -> structured description
# ---------------------------------------------------------------------------

def unwrap_tree(model, feature_names: list[str]) -> tuple[DecisionTreeClassifier, list[str]]:
    if isinstance(model, FeatureSubset):
        return model.estimator_, [feature_names[i] for i in model.columns]
    if isinstance(model, DecisionTreeClassifier):
        return model, list(feature_names)
    raise ExportError(f"Not a decision tree: {type(model).__name__}")


def tree_to_dict(tree_model: DecisionTreeClassifier, names: list[str]) -> dict:
    """Recursive dict of the tree; subtrees whose leaves all agree are collapsed."""
    t = tree_model.tree_
    classes = list(tree_model.classes_)

    def node(i: int) -> dict:
        n_samples = int(t.n_node_samples[i])
        if t.children_left[i] == -1:
            cls = int(classes[int(np.argmax(t.value[i][0]))])
            return {"leaf": True, "water": cls == 1, "samples": n_samples}
        left, right = node(t.children_left[i]), node(t.children_right[i])
        if left.get("leaf") and right.get("leaf") and left["water"] == right["water"]:
            return {"leaf": True, "water": left["water"], "samples": n_samples, "collapsed": True}
        thr64 = float(t.threshold[i])
        return {"leaf": False, "feature": names[t.feature[i]], "threshold": thr64,
                "threshold_f32": float(float32_floor(thr64)), "samples": n_samples,
                "left": left, "right": right}

    return node(0)


def tree_stats(d: dict) -> tuple[int, int, int]:
    """(decision nodes, leaves, depth) of the exported tree."""
    if d["leaf"]:
        return 0, 1, 0
    a = tree_stats(d["left"])
    b = tree_stats(d["right"])
    return a[0] + b[0] + 1, a[1] + b[1], 1 + max(a[2], b[2])


def logistic_to_dict(model, names: list[str]) -> dict:
    if not (isinstance(model, Pipeline) and "clf" in model.named_steps):
        raise ExportError("Expected a Pipeline(scaler, LogisticRegression)")
    scaler, clf = model.named_steps["scaler"], model.named_steps["clf"]
    coef = clf.coef_[0].astype(float)
    mean, scale = scaler.mean_.astype(float), scaler.scale_.astype(float)
    weights = coef / scale
    bias = float(clf.intercept_[0] - np.sum(coef * mean / scale))
    return {"weights": dict(zip(names, weights.tolist())), "bias": bias,
            "scaled_coefficients": dict(zip(names, coef.tolist())),
            "scaler_mean": dict(zip(names, mean.tolist())), "scaler_scale": dict(zip(names, scale.tolist()))}


# ---------------------------------------------------------------------------
# C code generation
# ---------------------------------------------------------------------------

def tree_to_c(d: dict, indent: int = 1) -> list[str]:
    pad = "    " * indent
    if d["leaf"]:
        label = "WATER" if d["water"] else "NO_WATER"
        return [f"{pad}return {'true' if d['water'] else 'false'};  // {label} ({d['samples']} training samples)"]
    var = C_PARAM_NAMES[d["feature"]]
    lines = [f"{pad}// check {d['feature']} (split learned from {d['samples']} training samples)",
             f"{pad}if ({var} <= {c_float(d['threshold_f32'])}) {{"]
    lines += tree_to_c(d["left"], indent + 1)
    lines.append(f"{pad}}} else {{")
    lines += tree_to_c(d["right"], indent + 1)
    lines.append(f"{pad}}}")
    return lines


def logistic_to_c(d: dict, names: list[str]) -> list[str]:
    lines = ["    // Logistic regression with the StandardScaler folded into the weights.",
             "    // z > 0  <=>  predicted probability of WATER > 0.5",
             f"    float z = {c_float(d['bias'])};"]
    for n in names:
        lines.append(f"    z += {c_float(d['weights'][n])} * {C_PARAM_NAMES[n]};  // {n}")
    lines.append("    return z > 0.0f;")
    return lines


def generate_header(cfg: Config, meta: dict, model_name: str, model) -> tuple[str, dict]:
    names = cfg.feature_names
    demo = meta["data_kind"] != "REAL"
    soil_param = C_PARAM_NAMES[cfg.soil_feature]

    if model_name in ("decision_tree", "soil_threshold_learned"):
        tree, tree_names = unwrap_tree(model, names)
        structure = tree_to_dict(tree, tree_names)
        body = tree_to_c(structure)
        n_nodes, n_leaves, depth = tree_stats(structure)
        summary = {"type": "decision_tree", "tree": structure, "decision_nodes": n_nodes,
                   "leaves": n_leaves, "depth": depth}
    elif model_name == "logistic_regression":
        structure = logistic_to_dict(model, names)
        body = logistic_to_c(structure, names)
        summary = {"type": "logistic_regression", **structure}
    else:
        raise ExportError(f"Model {model_name!r} cannot be exported. Exportable: {sorted(EXPORTABLE)}")

    has_cal = cfg.soil_raw_dry is not None and cfg.soil_raw_wet is not None
    dry = c_float(cfg.soil_raw_dry) if has_cal else None
    wet = c_float(cfg.soil_raw_wet) if has_cal else None

    H = []
    H.append("// ============================================================================")
    H.append("//  model.h  -  AUTO-GENERATED by src/export_embedded.py. Do not edit by hand;")
    H.append("//  retrain and re-export instead.")
    H.append("// ============================================================================")
    if demo:
        H.append("//  !!! TRAINED ON SYNTHETIC DEMO DATA - FOR SOFTWARE TESTING ONLY !!!")
        H.append("//  !!! The decision logic below is NOT based on real measurements.  !!!")
    H.append(f"//  Model:        {model_name}")
    H.append(f"//  Data kind:    {meta['data_kind']}")
    H.append(f"//  Trained at:   {meta['created_at']}")
    H.append(f"//  Exported at:  {datetime.now().isoformat(timespec='seconds')}")
    H.append(f"//  Training rows: {meta['split']['train_rows']}   Test rows: {meta['split']['test_rows']}")
    H.append(f"//  Features (order): {', '.join(names)}")
    H.append("//  Output: true = WATER, false = NO_WATER")
    H.append("//")
    H.append("//  Pure C/C++ with no Arduino dependencies, so it also compiles on a PC")
    H.append("//  (used by the Python-vs-C consistency check).")
    H.append("// ============================================================================")
    H.append("")
    H.append("#ifndef SMART_IRRIGATION_MODEL_H")
    H.append("#define SMART_IRRIGATION_MODEL_H")
    H.append("")
    H.append(f"#define MODEL_IS_DEMO {1 if demo else 0}")
    H.append(f"#define MODEL_NAME \"{model_name}\"")
    H.append(f"#define MODEL_SOIL_FEATURE_IS_PCT {1 if cfg.soil_feature == 'moisture_pct' else 0}")
    H.append("")
    if has_cal:
        H.append("// ---- Soil calibration (from the config used for training) ----")
        H.append("// raw == SOIL_RAW_DRY -> 0 %,  raw == SOIL_RAW_WET -> 100 %, linear, clipped.")
        H.append(f"static const float SOIL_RAW_DRY = {dry};")
        H.append(f"static const float SOIL_RAW_WET = {wet};")
        H.append("")
        H.append("static inline float soilRawToMoisturePct(float soilRaw) {")
        H.append("    // Same float32 operations, in the same order, as preprocessing.raw_to_moisture_pct")
        H.append("    float pct = (SOIL_RAW_DRY - soilRaw) / (SOIL_RAW_DRY - SOIL_RAW_WET) * 100.0f;")
        H.append("    if (pct < 0.0f) pct = 0.0f;")
        H.append("    if (pct > 100.0f) pct = 100.0f;")
        H.append("    return pct;")
        H.append("}")
        H.append("")
    H.append(f"// {soil_param}: {'calibrated soil moisture in %' if cfg.soil_feature == 'moisture_pct' else 'averaged raw ADC value'}")
    H.append(f"static inline bool needsWater(float {soil_param}, float temperatureC, float humidityPct) {{")
    unused = [C_PARAM_NAMES[n] for n in names if C_PARAM_NAMES[n] not in "\n".join(body)]
    for u in unused:
        H.append(f"    (void){u};  // not used by this model")
    H += body
    H.append("}")
    H.append("")
    H.append("// Convenience wrapper: raw ADC value -> same preprocessing as training -> decision.")
    H.append("static inline bool needsWaterFromRaw(int soilRaw, float temperatureC, float humidityPct) {")
    if cfg.soil_feature == "moisture_pct":
        H.append("    return needsWater(soilRawToMoisturePct((float)soilRaw), temperatureC, humidityPct);")
    else:
        H.append("    return needsWater((float)soilRaw, temperatureC, humidityPct);")
    H.append("}")
    H.append("")
    H.append("#endif  // SMART_IRRIGATION_MODEL_H")
    H.append("")
    return "\n".join(H), summary


# ---------------------------------------------------------------------------
# Human-readable explanation
# ---------------------------------------------------------------------------

def explain_tree(d: dict, depth: int = 0) -> list[str]:
    pad = "  " * depth
    if d["leaf"]:
        return [f"{pad}- → **{'WATER' if d['water'] else 'NO_WATER'}** ({d['samples']} training samples)"]
    thr = d["threshold_f32"]
    return ([f"{pad}- if `{d['feature']} <= {thr:.4g}`:"] + explain_tree(d["left"], depth + 1)
            + [f"{pad}- else (`{d['feature']} > {thr:.4g}`):"] + explain_tree(d["right"], depth + 1))


def write_explanation(path: Path, cfg: Config, meta: dict, model_name: str, summary: dict,
                      header_path: Path, verification: dict | None) -> None:
    demo = meta["data_kind"] != "REAL"
    L = [f"# Exported embedded model: `{model_name}`", ""]
    if demo:
        L += ["> **⚠ Trained on SYNTHETIC demo data. For software testing only; do not flash this "
              "onto the real controller.**", ""]
    L += [f"- Header: `{header_path.as_posix()}`",
          f"- Data kind: {meta['data_kind']}; trained {meta['created_at']}",
          f"- Input order: {', '.join(cfg.feature_names)}",
          f"- Calibration: soil_raw_dry = {cfg.soil_raw_dry} → 0 %, soil_raw_wet = {cfg.soil_raw_wet} → 100 %", ""]
    if summary["type"] == "decision_tree":
        L += [f"Decision tree after merging redundant branches: {summary['decision_nodes']} comparisons, "
              f"{summary['leaves']} leaves, depth {summary['depth']}. At most {summary['depth']} "
              "comparisons run per decision.", "", "## Decision logic", ""]
        L += explain_tree(summary["tree"])
    else:
        L += ["## Decision logic", "", "`z = bias + Σ weight_i × feature_i`, WATER if `z > 0`.", "",
              "| Term | Value |", "|---|---:|", f"| bias | {summary['bias']:.6g} |"]
        L += [f"| weight · {n} | {w:.6g} |" for n, w in summary["weights"].items()]
        L += ["", "A negative weight on moisture means drier soil pushes the decision towards WATER."]
    L += ["", "## Python ↔ C consistency check", ""]
    if verification is None:
        L.append("Not run (use without `--no-verify` and with g++ installed).")
    elif verification.get("skipped"):
        L.append(f"Skipped: {verification['skipped']}")
    else:
        L += [f"- Vectors tested: {verification['n']} (stored train/test rows, random inputs, near-threshold inputs)",
              f"- Agreement: {verification['agree']} / {verification['n']}",
              f"- Mismatches: {verification['mismatch']}"
              + (" (all within float rounding of the decision boundary)" if verification.get("mismatch_boundary_only") else "")]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# Verification: compile the header and compare against Python
# ---------------------------------------------------------------------------

HARNESS = r"""
#include <cstdio>
#include "model.h"
int main() {
    int raw; float t, h;
    while (std::scanf("%d %f %f", &raw, &t, &h) == 3) {
        std::printf("%d\n", needsWaterFromRaw(raw, t, h) ? 1 : 0);
    }
    return 0;
}
"""


def find_compiler() -> str | None:
    for c in ("g++", "clang++"):
        if shutil.which(c):
            return c
    return None


def test_vectors(cfg: Config, summary: dict, n_random: int = 3000, seed: int = 0) -> pd.DataFrame:
    """soil_raw (int), temperature_c and humidity_pct (float32) vectors to compare."""
    rng = np.random.default_rng(seed)
    frames = []
    processed = cfg.path("processed_dir")
    for name in ("train.csv", "test.csv"):
        p = processed / name
        if p.exists():
            frames.append(pd.read_csv(p)[["soil_raw", "temperature_c", "humidity_pct"]])
    frames.append(pd.DataFrame({
        "soil_raw": rng.integers(0, cfg.adc_max + 1, n_random),
        "temperature_c": rng.uniform(0, 50, n_random),
        "humidity_pct": rng.uniform(0, 100, n_random),
    }))
    # Inputs right at / next to every tree threshold.
    if summary["type"] == "decision_tree":
        edges = []

        def walk(d):
            if d["leaf"]:
                return
            edges.append((d["feature"], d["threshold_f32"]))
            walk(d["left"])
            walk(d["right"])
        walk(summary["tree"])
        base = {"soil_raw": 500, "temperature_c": 25.0, "humidity_pct": 60.0}
        rows = []
        for feat, thr in edges:
            t = np.float32(thr)
            for v in (np.nextafter(t, np.float32(-np.inf)), t, np.nextafter(t, np.float32(np.inf))):
                if feat in ("temperature_c", "humidity_pct"):
                    for raw in range(0, cfg.adc_max + 1, 16):
                        rows.append({**base, "soil_raw": raw, feat: float(v)})
        if cfg.soil_raw_dry is not None:
            for raw in range(0, cfg.adc_max + 1):  # every possible ADC value
                for t_, h_ in ((10.0, 30.0), (25.0, 60.0), (40.0, 90.0)):
                    rows.append({"soil_raw": raw, "temperature_c": t_, "humidity_pct": h_})
        if rows:
            frames.append(pd.DataFrame(rows))
    df = pd.concat(frames, ignore_index=True)
    df["soil_raw"] = df["soil_raw"].round().astype(int)
    df["temperature_c"] = df["temperature_c"].astype(np.float32)
    df["humidity_pct"] = df["humidity_pct"].astype(np.float32)
    return df


def verify(header_text: str, cfg: Config, model, summary: dict) -> dict:
    compiler = find_compiler()
    if compiler is None:
        return {"skipped": "no C++ compiler (g++/clang++) found on PATH"}
    vectors = test_vectors(cfg, summary)
    X = feature_matrix(add_features(vectors, cfg), cfg)
    py_pred = model.predict(X).astype(int)

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        (tmp / "model.h").write_text(header_text, encoding="utf-8")
        (tmp / "harness.cpp").write_text(HARNESS, encoding="utf-8")
        exe = tmp / "harness.exe"
        # -ffp-contract=off: no fused multiply-add, matching the AVR's separate float ops.
        cmd = [compiler, "-std=c++11", "-O0", "-ffp-contract=off", "-Wall", "-Werror",
               "-Wno-unused-function", "-o", str(exe), str(tmp / "harness.cpp")]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode != 0:
            raise ExportError(f"Generated header failed to compile:\n{res.stderr}")
        lines = "\n".join(f"{r} {float(t):.9g} {float(h):.9g}" for r, t, h in
                          zip(vectors["soil_raw"], vectors["temperature_c"], vectors["humidity_pct"]))
        run = subprocess.run([str(exe)], input=lines + "\n", capture_output=True, text=True)
        if run.returncode != 0:
            raise ExportError(f"Harness failed: {run.stderr}")
        c_pred = np.array([int(x) for x in run.stdout.split()], dtype=int)

    if len(c_pred) != len(py_pred):
        raise ExportError(f"Harness returned {len(c_pred)} results for {len(py_pred)} inputs")
    mismatch = np.flatnonzero(c_pred != py_pred)
    result = {"n": int(len(py_pred)), "agree": int(len(py_pred) - len(mismatch)),
              "mismatch": int(len(mismatch)), "compiler": compiler}
    if len(mismatch) and summary["type"] == "logistic_regression":
        margin = np.abs(model.decision_function(X[mismatch]))
        result["mismatch_boundary_only"] = bool(np.all(margin < 1e-4))
    if len(mismatch):
        result["examples"] = vectors.iloc[mismatch[:5]].to_dict("records")
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def export(cfg: Config, model_name: str | None = None, out: Path | None = None,
           run_verify: bool = True) -> dict:
    model_dir = cfg.path("model_dir")
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise ExportError(f"{meta_path} not found. Train first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    selected = meta["selection"]["chosen"]
    note = None
    if model_name is None:
        model_name = selected
        if model_name not in EXPORTABLE:
            note = (f"Selected model {selected!r} is not exportable to the Uno; exporting the "
                    "small decision tree instead.")
            model_name = "decision_tree"
    model = joblib.load(model_dir / "candidates" / f"{model_name}.joblib")

    header, summary = generate_header(cfg, meta, model_name, model)
    verification = verify(header, cfg, model, summary) if run_verify else None
    if verification and verification.get("mismatch") and not verification.get("mismatch_boundary_only"):
        raise ExportError(f"C and Python disagree on {verification['mismatch']} inputs, e.g. "
                          f"{verification.get('examples')}. Header NOT written.")

    out = Path(out) if out else cfg.path("firmware_model_header")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(header, encoding="utf-8")
    (model_dir / "embedded_model.json").write_text(json.dumps(
        {"model": model_name, "data_kind": meta["data_kind"], "feature_names": cfg.feature_names,
         "calibration": {"soil_raw_dry": cfg.soil_raw_dry, "soil_raw_wet": cfg.soil_raw_wet},
         "summary": summary, "verification": verification}, indent=2, default=str), encoding="utf-8")
    write_explanation(model_dir / "embedded_model.md", cfg, meta, model_name, summary, out, verification)
    return {"header": out, "model": model_name, "summary": summary, "verification": verification,
            "note": note, "data_kind": meta["data_kind"], "selected": selected}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Export the trained model as Arduino C/C++.")
    add_config_args(p)
    p.add_argument("--model", choices=sorted(EXPORTABLE), default=None,
                   help="Which trained candidate to export (default: the selected model)")
    p.add_argument("--out", type=Path, default=None, help="Output header path")
    p.add_argument("--no-verify", action="store_true", help="Skip the g++ consistency check")
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    try:
        r = export(cfg, args.model, args.out, not args.no_verify)
    except ExportError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    if r["data_kind"] != "REAL":
        print("*** SYNTHETIC DEMO model - for software testing only ***")
    if r["note"]:
        print(r["note"])
    print(f"Exported {r['model']} (selected during training: {r['selected']}) -> {r['header']}")
    s = r["summary"]
    if s["type"] == "decision_tree":
        print(f"  {s['decision_nodes']} comparisons, {s['leaves']} leaves, depth {s['depth']}")
    v = r["verification"]
    if v is None:
        print("  Python/C consistency check: not run")
    elif v.get("skipped"):
        print(f"  Python/C consistency check skipped: {v['skipped']}")
    else:
        print(f"  Python/C consistency check ({v['compiler']}): {v['agree']}/{v['n']} vectors agree")
    return 0


if __name__ == "__main__":
    sys.exit(main())
