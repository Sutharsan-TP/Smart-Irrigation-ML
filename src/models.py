"""
Candidate models and shared metric code.

Kept deliberately small and interpretable (no neural networks):
  soil_threshold_learned  depth-1 tree on soil moisture only (= a learned threshold)
  logistic_regression     linear model on all three features (scaled)
  decision_tree           small tree on all three features
  random_forest           a few small trees on all three features

"soil_threshold_learned" answers a useful question: if an ML model beats the
team's fixed threshold, is it because temperature/humidity help, or just
because the data suggest a better threshold?
"""

from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, fbeta_score,
                             precision_score, recall_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from .config import Config

# Higher = easier to run on an Arduino Uno. Used for the comparison table and
# as the tie-breaker during model selection.
EMBEDDED_SUITABILITY = {
    "threshold_baseline": (4, "High (one comparison)"),
    "soil_threshold_learned": (4, "Very High (one comparison)"),
    "decision_tree": (3, "Very High (nested if statements)"),
    "logistic_regression": (2, "High (3 multiply-adds)"),
    "random_forest": (1, "Medium (several trees; not exported)"),
}
EXPORTABLE = {"soil_threshold_learned", "decision_tree", "logistic_regression"}


class FeatureSubset(BaseEstimator, ClassifierMixin):
    """Fit `estimator` on a subset of the input columns (by index)."""

    def __init__(self, estimator=None, columns=(0,)):
        self.estimator = estimator
        self.columns = columns

    def fit(self, X, y):
        self.estimator_ = clone(self.estimator).fit(np.asarray(X)[:, list(self.columns)], y)
        self.classes_ = self.estimator_.classes_
        return self

    def predict(self, X):
        return self.estimator_.predict(np.asarray(X)[:, list(self.columns)])

    def predict_proba(self, X):
        return self.estimator_.predict_proba(np.asarray(X)[:, list(self.columns)])


def build_candidates(cfg: Config) -> dict:
    seed = cfg.random_seed
    m = cfg.models
    return {
        "soil_threshold_learned": FeatureSubset(DecisionTreeClassifier(max_depth=1, random_state=seed), (0,)),
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=seed, **m.get("logistic_regression", {}))),
        ]),
        "decision_tree": DecisionTreeClassifier(random_state=seed, **m.get("decision_tree", {})),
        "random_forest": RandomForestClassifier(random_state=seed, **m.get("random_forest", {})),
    }


def classification_metrics(y_true, y_pred) -> dict:
    """Metrics with WATER (=1) as the positive class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    positives = tp + fn
    return {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "f2": float(fbeta_score(y_true, y_pred, beta=2, zero_division=0)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        # share of plants that needed water but were told NO_WATER
        "false_negative_rate": float(fn / positives) if positives else float("nan"),
    }
