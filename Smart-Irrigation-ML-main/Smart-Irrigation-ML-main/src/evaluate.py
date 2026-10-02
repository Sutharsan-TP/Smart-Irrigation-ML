"""
Evaluate the trained candidates on the held-out test set and write the report.

Outputs (in the configured report/model dirs):
  <model_dir>/evaluation.json        all metrics, machine-readable
  <report_dir>/model_report.md       human-readable report (numbers from this run)
  <report_dir>/figures/*.png         confusion matrices, data scatter, tree diagram

Usage:
    python -m src.evaluate
    python -m src.evaluate --demo
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from .baseline import ThresholdBaseline
from .config import Config, ConfigError, add_config_args, config_from_args
from .models import EMBEDDED_SUITABILITY, classification_metrics
from .preprocessing import encode_labels, feature_matrix

# Chart styling: reference palette (categorical blue/orange, blue sequential ramp).
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
SURFACE = "#fcfcfb"
COLOR_NO_WATER = "#2a78d6"
COLOR_WATER = "#eb6834"
SEQ_BLUE = LinearSegmentedColormap.from_list("seq_blue", ["#f0f6fe", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"])

MODEL_ORDER = ["threshold_baseline", "soil_threshold_learned", "logistic_regression",
               "decision_tree", "random_forest"]
PRETTY = {
    "threshold_baseline": "Threshold baseline (team threshold)",
    "soil_threshold_learned": "Learned soil-only threshold",
    "logistic_regression": "Logistic Regression",
    "decision_tree": "Small Decision Tree",
    "random_forest": "Small Random Forest",
}


def load_models(cfg: Config, meta: dict) -> tuple[dict, str | None]:
    models = {}
    note = None
    try:
        models["threshold_baseline"] = ThresholdBaseline.from_config(cfg)
    except ConfigError as exc:
        note = f"Threshold baseline not evaluated: {exc}"
    for name in MODEL_ORDER[1:]:
        path = cfg.path("model_dir") / "candidates" / f"{name}.joblib"
        if path.exists():
            models[name] = joblib.load(path)
    return models, note


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_confusion(metrics: dict, title: str, path) -> None:
    cm = np.array([[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.8), facecolor=SURFACE)
    ax.imshow(cm, cmap=SEQ_BLUE, vmin=0, vmax=max(cm.max(), 1))
    names = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            dark = cm[i, j] > 0.55 * max(cm.max(), 1)
            ax.text(j, i, f"{cm[i, j]}\n{names[i][j]}", ha="center", va="center", fontsize=10,
                    color="#ffffff" if dark else INK)
    ax.set_xticks([0, 1], ["NO_WATER", "WATER"])
    ax.set_yticks([0, 1], ["NO_WATER", "WATER"])
    ax.set_xlabel("Predicted   (FN = needed water, predicted NO_WATER)", color=INK_2, fontsize=8)
    ax.set_ylabel("Actual", color=INK_2)
    ax.set_title(title, color=INK, fontsize=10)
    ax.tick_params(colors=MUTED, labelcolor=INK_2)
    for s in ax.spines.values():
        s.set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_scatter(df: pd.DataFrame, cfg: Config, threshold, path) -> None:
    soil = cfg.soil_feature
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.8), facecolor=SURFACE, sharex=True)
    for ax, other, label in ((axes[0], "temperature_c", "Temperature (°C)"),
                             (axes[1], "humidity_pct", "Humidity (%)")):
        _style(ax)
        for lab, color, marker in (("NO_WATER", COLOR_NO_WATER, "o"), ("WATER", COLOR_WATER, "^")):
            sub = df[df["irrigation_label"] == lab]
            ax.scatter(sub[soil], sub[other], s=16, c=color, marker=marker, alpha=0.75,
                       edgecolors=SURFACE, linewidths=0.5, label=lab)
        if threshold is not None and soil == "moisture_pct":
            ax.axvline(threshold, color=INK_2, linewidth=1, linestyle="--")
            ax.text(threshold, ax.get_ylim()[1], " team threshold", color=INK_2, fontsize=8, va="top")
        ax.set_xlabel("Calibrated soil moisture (%)" if soil == "moisture_pct" else "Soil raw (ADC)",
                      color=INK_2)
        ax.set_ylabel(label, color=INK_2)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, labelcolor=INK_2, fontsize=9, ncol=2,
               loc="upper right", bbox_to_anchor=(0.98, 0.99))
    fig.suptitle("All labeled readings", color=INK, fontsize=11, x=0.02, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_tree_figure(model, cfg: Config, path) -> None:
    from sklearn.tree import plot_tree
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor=SURFACE)
    plot_tree(model, feature_names=cfg.feature_names, class_names=["NO_WATER", "WATER"],
              filled=False, impurity=False, proportion=False, rounded=True, fontsize=8, ax=ax)
    ax.set_title("Small Decision Tree (trained model)", color=INK)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def fmt(x) -> str:
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def verdict(test: dict, chosen: str, meta: dict) -> str:
    if "threshold_baseline" not in test:
        return ("The threshold baseline could not be evaluated (no team threshold configured), so "
                "this run cannot say whether ML adds value.")
    b, m = test["threshold_baseline"], test[chosen]
    n_test = meta["split"]["test_rows"]
    lines = [f"On the {n_test}-row held-out test set, the selected model ({PRETTY[chosen]}) has "
             f"F1 {m['f1']:.3f} vs {b['f1']:.3f} for the baseline, recall {m['recall']:.3f} vs "
             f"{b['recall']:.3f}, and {m['fn']} vs {b['fn']} false negatives."]
    better_f1 = m["f1"] > b["f1"] + 0.01
    fewer_fn = m["fn"] < b["fn"]
    worse = m["f1"] < b["f1"] - 0.01 or m["fn"] > b["fn"]
    if better_f1 and fewer_fn:
        lines.append("It improves on the baseline on this test set. With one test split and "
                     "correlated readings, treat this as a hint that needs confirmation with more "
                     "sessions, not as proof.")
    elif worse:
        lines.append("**The ML model does not beat the simple threshold baseline on this test set.** "
                     "The threshold controller remains the better or equal choice.")
    else:
        lines.append("**The difference is within noise: this run shows no clear advantage of ML over "
                     "the simple threshold baseline.**")
    return " ".join(lines)


def write_report(cfg: Config, meta: dict, test: dict, baseline_note, figs: dict, path) -> None:
    demo = meta["data_kind"] != "REAL"
    ds, sp, cv = meta["dataset"], meta["split"], meta["cross_validation"]
    chosen = meta["selection"]["chosen"]
    L = []
    L.append("# Model Report" + (" — SYNTHETIC DEMO DATA" if demo else ""))
    L.append("")
    if demo:
        L += ["> **⚠ These results come from SYNTHETIC demo data generated by `src/make_demo_data.py`.**",
              "> They exist only to prove that the software pipeline runs end to end.",
              "> They are **not** experimental results and must not be quoted as project results.",
              "> The demo labels were produced by an invented rule plus 5 % noise.", ""]
    L.append(f"Generated automatically by `python -m src.evaluate{' --demo' if demo else ''}` on "
             f"{datetime.now().isoformat(timespec='seconds')}. Every number below comes from that run.")
    L.append("")

    L += ["## 1. Dataset summary", "",
          "| Item | Value |", "|---|---|",
          f"| Data kind | {meta['data_kind']} |",
          f"| Files | {len(meta['data_files'])} |",
          f"| Rows loaded | {meta['cleaning']['rows_loaded']} |",
          f"| Rows kept after cleaning | {meta['cleaning']['rows_kept']} |",
          f"| Collection sessions | {ds['sessions']} |",
          f"| Time range | {ds['time_range'][0]} → {ds['time_range'][1]} |",
          f"| Class distribution | WATER {ds['class_distribution']['WATER']}, NO_WATER {ds['class_distribution']['NO_WATER']} |",
          f"| Label sources | {', '.join(f'{k}: {v}' for k, v in meta['label_sources'].items())} |", ""]
    if meta["cleaning"]["dropped"]:
        L.append("Rows dropped during cleaning:")
        L.append("")
        for reason, n in meta["cleaning"]["dropped"].items():
            L.append(f"- {reason}: {n}")
        L.append("")
    if meta["warnings"]:
        L.append("**Warnings:**")
        L.append("")
        L += [f"- {w}" for w in meta["warnings"]]
        L.append("")
    L.append(f"![Readings]({figs['scatter']})")
    L.append("")

    c = meta["config"]
    L += ["## 2. Preprocessing", "",
          "- Non-numeric/missing sensor values, impossible values (soil_raw outside 0–1023, "
          "temperature/humidity outside the configured ranges), duplicates and unlabeled rows are dropped.",
          f"- Soil calibration: `soil_raw_dry = {c['soil_raw_dry']}` → 0 %, `soil_raw_wet = {c['soil_raw_wet']}` → 100 %, "
          "linear in between, clipped to 0–100 %, computed in 32-bit float (same as the Arduino).",
          f"- Model features, in order: `{', '.join(meta['feature_names'])}`. Timestamp, plant ID, "
          "session ID and soil condition text are **not** used as features.",
          "- Logistic Regression uses standard scaling (folded into the weights on export); trees need no scaling.",
          ""]

    L += ["## 3. Train/test split", "",
          f"- Strategy: **{sp['strategy']}**"
          + (" (whole sessions go to either train or test, so correlated readings from one session "
             "cannot leak across)." if sp["strategy"] == "group" else ""),
          f"- Training set: {sp['train_rows']} rows, {len(sp['train_sessions'])} session(s), "
          f"WATER {sp['train_class_distribution']['WATER']} / NO_WATER {sp['train_class_distribution']['NO_WATER']}",
          f"- Test set: {sp['test_rows']} rows, {len(sp['test_sessions'])} session(s), "
          f"WATER {sp['test_class_distribution']['WATER']} / NO_WATER {sp['test_class_distribution']['NO_WATER']}",
          f"- Test sessions: {', '.join(sp['test_sessions'])}"]
    L += [f"- Note: {n}" for n in sp["notes"]]
    L.append("")

    L += ["## 4. Baseline", "",
          f"`WATER if moisture_pct < {c['moisture_threshold_pct']} else NO_WATER`. The threshold is the "
          "team's configured value and is never fitted to data.", ""]
    if baseline_note:
        L += [f"> {baseline_note}", ""]

    L += ["## 5. Models tested", "",
          "| Model | What it is |", "|---|---|",
          "| Learned soil-only threshold | depth-1 tree on soil moisture only: shows whether ML gains "
          "come from temperature/humidity or just from a different threshold |",
          f"| Logistic Regression | linear model on all 3 features, C={c['models']['logistic_regression'].get('C')} |",
          f"| Small Decision Tree | {c['models']['decision_tree']} |",
          f"| Small Random Forest | {c['models']['random_forest']} |", ""]

    L += ["## 6. Cross-validation (training set only)", "", f"Method: {cv['method']}", ""]
    if cv["results"]:
        L += ["| Model | Accuracy | Precision | Recall | F1 | F2 | Total FN |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        for name in MODEL_ORDER:
            r = cv["results"].get(name)
            if r:
                L.append(f"| {PRETTY[name]} | " + " | ".join(
                    f"{r[k]['mean']:.3f} ± {r[k]['std']:.3f}" for k in ("accuracy", "precision", "recall", "f1", "f2"))
                    + f" | {r['fn_total']} |")
        L.append("")

    L += ["## 7. Held-out test metrics", "",
          "Positive class = WATER. A **false negative** = the plant needed water but the model said NO_WATER.", "",
          "| Model | Accuracy | Precision | Recall | F1 | FN | FP | Embedded suitability |",
          "|---|---:|---:|---:|---:|---:|---:|---|"]
    for name in MODEL_ORDER:
        if name in test:
            r = test[name]
            L.append(f"| {PRETTY[name]}{' ✔ selected' if name == chosen else ''} | {fmt(r['accuracy'])} | "
                     f"{fmt(r['precision'])} | {fmt(r['recall'])} | {fmt(r['f1'])} | {r['fn']} | {r['fp']} | "
                     f"{EMBEDDED_SUITABILITY[name][1]} |")
    L.append("")

    L += ["## 8. Confusion matrices (test set)", ""]
    for name in MODEL_ORDER:
        if name in figs.get("confusion", {}):
            L.append(f"![{PRETTY[name]}]({figs['confusion'][name]})")
    L.append("")
    L += ["| Model | TN | FP | FN | TP | Missed-watering rate (FN / actual WATER) |",
          "|---|---:|---:|---:|---:|---:|"]
    for name in MODEL_ORDER:
        if name in test:
            r = test[name]
            L.append(f"| {PRETTY[name]} | {r['tn']} | {r['fp']} | {r['fn']} | {r['tp']} | {fmt(r['false_negative_rate'])} |")
    L.append("")

    L += ["## 9. Selected model", "",
          f"**{PRETTY[chosen]}**. {meta['selection']['reason']}.", "",
          "The choice was made from cross-validation on the training set before the test set was scored.",
          "", "### Does ML beat the baseline?", "", verdict(test, chosen, meta), ""]
    if "decision_tree" in figs:
        L += [f"![Decision tree]({figs['decision_tree']})", ""]

    L += ["## 10. Embedded deployment notes", "",
          "- Export with `python -m src.export_embedded" + (" --demo" if demo else "") + "`. "
          "It writes a C header with the calibration and the model as plain `if` statements (tree) "
          "or three multiply-adds (logistic regression), then compiles it with g++ and checks it "
          "against the Python model on every train/test row plus random inputs.",
          "- The Random Forest is not exported: several trees would fit in an Uno's flash, but it is "
          "harder to check and explain.",
          "- The Arduino must average the soil sensor exactly like `sensor_logger.ino`, because that "
          "is how the training data was produced.", ""]

    L += ["## 11. Limitations", "",
          f"- Single train/test split of {sp['test_rows']} rows; test metrics have wide uncertainty.",
          "- Readings within a session are correlated, so the effective sample size is closer to the "
          "number of sessions than the number of rows.",
          "- Labels are only as good as the team's labeling procedure; if labels come from the "
          "threshold rule, the baseline is correct by construction.",
          "- One sensor, one pot/soil type: the model is not expected to transfer to other soils or plants "
          "without recalibration and new data.",
          "- DHT11 resolution is coarse (±2 °C, ±5 % RH), which limits what temperature/humidity can add."]
    if demo:
        L.append("- **All of the above was computed on synthetic demo data.**")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")


def evaluate(cfg: Config) -> dict:
    model_dir = cfg.path("model_dir")
    meta_path = model_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"{meta_path} not found. Run `python -m src.train"
                                f"{' --demo' if cfg.is_demo else ''}` first.")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    processed = cfg.path("processed_dir")
    test_df = pd.read_csv(processed / "test.csv")
    all_df = pd.concat([pd.read_csv(processed / "train.csv"), test_df], ignore_index=True)
    X_test, y_test = feature_matrix(test_df, cfg), encode_labels(test_df["irrigation_label"])

    models, baseline_note = load_models(cfg, meta)
    test = {name: classification_metrics(y_test, m.predict(X_test)) for name, m in models.items()}

    report_dir = cfg.path("report_dir")
    fig_dir = report_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs = {"confusion": {}}
    for name, r in test.items():
        p = fig_dir / f"confusion_{name}.png"
        plot_confusion(r, PRETTY[name], p)
        figs["confusion"][name] = f"figures/{p.name}"
    plot_scatter(all_df, cfg, cfg.moisture_threshold_pct, fig_dir / "readings_scatter.png")
    figs["scatter"] = "figures/readings_scatter.png"
    if "decision_tree" in models:
        plot_tree_figure(models["decision_tree"], cfg, fig_dir / "decision_tree.png")
        figs["decision_tree"] = "figures/decision_tree.png"

    evaluation = {"created_at": datetime.now().isoformat(timespec="seconds"),
                  "data_kind": meta["data_kind"], "test_metrics": test,
                  "selected": meta["selection"]["chosen"], "baseline_note": baseline_note}
    (model_dir / "evaluation.json").write_text(json.dumps(evaluation, indent=2), encoding="utf-8")
    write_report(cfg, meta, test, baseline_note, figs, report_dir / "model_report.md")
    return {"meta": meta, "test": test, "report": report_dir / "model_report.md"}


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Evaluate trained models and write the model report.")
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    try:
        out = evaluate(cfg)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    meta, test = out["meta"], out["test"]
    if meta["data_kind"] != "REAL":
        print("*** SYNTHETIC DEMO DATA - these numbers are NOT project results ***")
    print(f"Test set: {meta['split']['test_rows']} rows ({meta['split']['strategy']} split)")
    print(f"{'model':<24}{'acc':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'FN':>5}{'FP':>5}")
    for name in MODEL_ORDER:
        if name in test:
            r = test[name]
            mark = " <- selected" if name == meta["selection"]["chosen"] else ""
            print(f"{name:<24}{r['accuracy']:>7.3f}{r['precision']:>7.3f}{r['recall']:>8.3f}"
                  f"{r['f1']:>7.3f}{r['fn']:>5}{r['fp']:>5}{mark}")
    print("\n" + verdict(test, meta["selection"]["chosen"], meta).replace("**", ""))
    for w in meta["warnings"]:
        print(f"WARNING: {w}")
    print(f"\nReport written to {out['report']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
