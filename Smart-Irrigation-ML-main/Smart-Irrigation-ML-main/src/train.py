"""
Train and select a small irrigation classifier.

Steps
  1. load + clean every CSV in the data directory
  2. leakage-aware train/test split (whole sessions, or time-based)
  3. cross-validate every candidate + the threshold baseline on the TRAINING set
  4. select a model using CV results only (the test set is not looked at here)
  5. fit every candidate on the full training set and save them with metadata

Selection rule (fixed before any results are seen):
  highest mean CV score on `selection_metric` (default F2, which weights recall
  on WATER more than precision because a missed watering is worse than an
  extra one); scores within 0.01 of the best are treated as a tie and the
  tie goes to the model that is easier to run on the Arduino Uno.

Usage:
    python -m src.train            # real data in data/raw/
    python -m src.train --demo     # SYNTHETIC demo data (software testing)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold

from .baseline import ThresholdBaseline
from .config import LABELS, Config, ConfigError, add_config_args, config_from_args
from .models import EMBEDDED_SUITABILITY, build_candidates, classification_metrics
from .preprocessing import DataError, encode_labels, feature_matrix, load_dataset, split_dataset

TIE_TOLERANCE = 0.01


def small_dataset_warnings(n_rows: int, class_counts: dict, n_sessions: int) -> list[str]:
    warnings = []
    if n_rows < 200:
        warnings.append(f"Small dataset: only {n_rows} labeled rows. Metrics may be unstable.")
    minority = min(class_counts.values()) if class_counts else 0
    if minority < 30:
        warnings.append(f"Minority class has only {minority} rows. Precision/recall are unreliable.")
    if n_sessions < 5:
        warnings.append(f"Only {n_sessions} collection session(s). Readings within a session are "
                        "correlated, so the effective sample size is much smaller than the row count.")
    return warnings


def class_distribution(labels) -> dict:
    s = pd.Series(labels).value_counts()
    return {lab: int(s.get(lab, 0)) for lab in LABELS}


def make_cv_splitter(y, groups, cfg: Config):
    """Returns (splitter, description) or (None, reason) if CV is not possible."""
    n_groups = len(np.unique(groups))
    min_class = int(np.bincount(y, minlength=2).min())
    k_max = cfg.cv_folds
    if n_groups >= 3:
        k = min(k_max, n_groups)
        return (StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=cfg.random_seed),
                f"{k}-fold StratifiedGroupKFold (whole sessions held out per fold)")
    k = min(k_max, min_class)
    if k < 2:
        return None, "Cross-validation skipped: fewer than 2 examples of one class in training data."
    return (StratifiedKFold(n_splits=k, shuffle=True, random_state=cfg.random_seed),
            f"{k}-fold StratifiedKFold on rows (too few sessions for grouped CV; "
            "scores are likely optimistic because neighbouring readings are correlated)")


def cross_validate(models: dict, X, y, groups, cfg: Config) -> tuple[dict, str]:
    splitter, desc = make_cv_splitter(y, groups, cfg)
    if splitter is None:
        return {}, desc
    per_model: dict[str, list[dict]] = {name: [] for name in models}
    for train_i, val_i in splitter.split(X, y, groups):
        if len(np.unique(y[train_i])) < 2:
            continue
        for name, model in models.items():
            fitted = clone(model).fit(X[train_i], y[train_i]) if name != "threshold_baseline" else model
            per_model[name].append(classification_metrics(y[val_i], fitted.predict(X[val_i])))
    results = {}
    for name, folds in per_model.items():
        if not folds:
            continue
        results[name] = {"folds": len(folds)}
        for metric in ("accuracy", "precision", "recall", "f1", "f2"):
            vals = np.array([f[metric] for f in folds])
            results[name][metric] = {"mean": float(vals.mean()), "std": float(vals.std())}
        results[name]["fn_total"] = int(sum(f["fn"] for f in folds))
    return results, desc


def select_model(cv_results: dict, metric: str, candidates: list[str]) -> tuple[str, str]:
    scored = {n: cv_results[n][metric]["mean"] for n in candidates if n in cv_results}
    if not scored:
        return "decision_tree", ("No cross-validation results (dataset too small). Defaulted to the "
                                 "small decision tree because it is the easiest model to deploy. "
                                 "This is NOT an evidence-based choice.")
    best = max(scored.values())
    tied = [n for n, s in scored.items() if s >= best - TIE_TOLERANCE]
    chosen = max(tied, key=lambda n: (EMBEDDED_SUITABILITY[n][0], scored[n]))
    reason = f"Highest mean CV {metric.upper()} ({scored[chosen]:.3f})"
    if len(tied) > 1:
        reason = (f"Mean CV {metric.upper()} within {TIE_TOLERANCE} of the best ({best:.3f}) for "
                  f"{', '.join(sorted(tied))}; chose {chosen} as the most Arduino-friendly of these "
                  f"(its score: {scored[chosen]:.3f})")
    return chosen, reason


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def train(cfg: Config) -> dict:
    np.random.seed(cfg.random_seed)
    df, cleaning = load_dataset(cfg)
    split = split_dataset(df, cfg)
    train_df = df.iloc[split.train_idx].reset_index(drop=True)
    test_df = df.iloc[split.test_idx].reset_index(drop=True)

    X_train, y_train = feature_matrix(train_df, cfg), encode_labels(train_df["irrigation_label"])
    groups_train = train_df["session_id"].to_numpy()

    candidates = build_candidates(cfg)
    to_validate = dict(candidates)
    baseline_note = None
    try:
        to_validate["threshold_baseline"] = ThresholdBaseline.from_config(cfg)
    except ConfigError as exc:
        baseline_note = f"Threshold baseline not evaluated: {exc}"

    cv_results, cv_desc = cross_validate(to_validate, X_train, y_train, groups_train, cfg)
    chosen, reason = select_model(cv_results, cfg.selection_metric, list(candidates))

    model_dir = cfg.path("model_dir")
    (model_dir / "candidates").mkdir(parents=True, exist_ok=True)
    for name, model in candidates.items():
        model.fit(X_train, y_train)
        joblib.dump(model, model_dir / "candidates" / f"{name}.joblib")
    joblib.dump(candidates[chosen], model_dir / "model.joblib")

    processed = cfg.path("processed_dir")
    processed.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(processed / "train.csv", index=False)
    test_df.to_csv(processed / "test.csv", index=False)

    label_sources = df.get("label_source", pd.Series(dtype=str)).replace("", "unspecified")
    data_dir = cfg.path("data_dir")
    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "data_kind": "SYNTHETIC_DEMO" if cfg.is_demo else "REAL",
        "config": cfg.to_dict(),
        "feature_names": cfg.feature_names,
        "label_encoding": {lab: i for i, lab in enumerate(LABELS)},
        "positive_class": "WATER",
        "data_files": {name: file_sha256(data_dir / name) for name in cleaning.files},
        "cleaning": cleaning.as_dict(),
        "label_sources": label_sources.value_counts().to_dict(),
        "dataset": {
            "rows": len(df),
            "sessions": int(df["session_id"].nunique()),
            "class_distribution": class_distribution(df["irrigation_label"]),
            "time_range": [str(df["timestamp"].min()), str(df["timestamp"].max())],
        },
        "split": {
            "strategy": split.strategy,
            "notes": split.notes,
            "train_rows": len(train_df),
            "test_rows": len(test_df),
            "train_sessions": sorted(train_df["session_id"].unique().tolist()),
            "test_sessions": sorted(test_df["session_id"].unique().tolist()),
            "train_class_distribution": class_distribution(train_df["irrigation_label"]),
            "test_class_distribution": class_distribution(test_df["irrigation_label"]),
        },
        "cross_validation": {"method": cv_desc, "results": cv_results},
        "baseline_note": baseline_note,
        "selection": {"metric": cfg.selection_metric, "chosen": chosen, "reason": reason},
        "warnings": small_dataset_warnings(len(df), class_distribution(df["irrigation_label"]),
                                           int(df["session_id"].nunique())),
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__},
    }
    if set(label_sources.unique()) == {"threshold_rule"}:
        metadata["warnings"].append(
            "Every label was produced by the threshold rule, so the threshold baseline is correct "
            "by construction and ML cannot show a real advantage. Add independently judged labels.")
    with open(model_dir / "metadata.json", "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)
    return metadata


def print_summary(meta: dict) -> None:
    kind = meta["data_kind"]
    banner = "  *** SYNTHETIC DEMO DATA - results are NOT project results ***" if kind != "REAL" else ""
    print(f"\n=== Training summary ({kind}) ==={banner}")
    ds, sp = meta["dataset"], meta["split"]
    print(f"Rows used: {ds['rows']} from {ds['sessions']} session(s); class distribution {ds['class_distribution']}")
    print(f"Dropped during cleaning: {meta['cleaning']['dropped'] or 'none'}")
    print(f"Split: {sp['strategy']} -> train {sp['train_rows']} rows {sp['train_class_distribution']}, "
          f"test {sp['test_rows']} rows {sp['test_class_distribution']}")
    for note in sp["notes"]:
        print(f"  note: {note}")
    cv = meta["cross_validation"]
    print(f"\nCross-validation: {cv['method']}")
    if cv["results"]:
        print(f"  {'model':<24}{'acc':>7}{'prec':>7}{'recall':>8}{'F1':>7}{'F2':>7}{'FN':>5}")
        for name, r in cv["results"].items():
            print(f"  {name:<24}{r['accuracy']['mean']:>7.3f}{r['precision']['mean']:>7.3f}"
                  f"{r['recall']['mean']:>8.3f}{r['f1']['mean']:>7.3f}{r['f2']['mean']:>7.3f}{r['fn_total']:>5}")
    if meta["baseline_note"]:
        print(f"  {meta['baseline_note']}")
    print(f"\nSelected: {meta['selection']['chosen']} - {meta['selection']['reason']}")
    for w in meta["warnings"]:
        print(f"WARNING: {w}")
    print("\nNext: python -m src.evaluate" + (" --demo" if kind != "REAL" else ""))


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="Train irrigation classifiers (offline).")
    add_config_args(p)
    args = p.parse_args(argv)
    cfg = config_from_args(args)
    try:
        meta = train(cfg)
    except (DataError, ConfigError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print_summary(meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())
