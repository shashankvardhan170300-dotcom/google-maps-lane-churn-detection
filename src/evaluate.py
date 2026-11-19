"""Evaluation harness: metrics + a short human-readable report.

All metrics reported here are DEMO metrics computed on synthetic data.
They describe how well the demo models separate synthetic churn from
synthetic stable traces, and must not be confused with the production
results from Shashank's Google work (+20% recall / +14% accuracy),
which are stated separately in the README.
"""

from __future__ import annotations

import json

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(model, X_test, y_test) -> dict:
    """Score one fitted model and return a metrics dict."""
    y_pred = model.predict(X_test)
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "n_test": int(len(y_test)),
        "n_predicted_churn": int(np.sum(y_pred == 1)),
        "n_actual_churn": int(np.sum(y_test == 1)),
    }


def evaluate_models(models: dict, X_test, y_test) -> dict[str, dict]:
    """Score every fitted model; returns ``{name: metrics}``."""
    return {
        name: evaluate_model(model, X_test, y_test)
        for name, model in models.items()
    }


def format_report(results: dict[str, dict], title: str = "Lane-churn detection — demo evaluation") -> str:
    """Render the metrics dict as a plain-text report."""
    lines = [title, "=" * len(title), ""]
    for name, m in results.items():
        cm = m["confusion_matrix"]
        lines += [
            f"Model: {name}",
            f"  test traces: {m['n_test']} "
            f"(actual churn: {m['n_actual_churn']}, "
            f"predicted churn: {m['n_predicted_churn']})",
            f"  accuracy:  {m['accuracy']:.4f}",
            f"  precision: {m['precision']:.4f}",
            f"  recall:    {m['recall']:.4f}",
            f"  f1:        {m['f1']:.4f}",
            "  confusion matrix [[TN, FP], [FN, TP]]:",
            f"    [[{cm['true_negative']}, {cm['false_positive']}], "
            f"[{cm['false_negative']}, {cm['true_positive']}]]",
            "",
        ]
    best = max(results, key=lambda k: results[k]["recall"])
    lines.append(
        f"Best recall: {best} "
        f"(recall {results[best]['recall']:.4f}, "
        f"accuracy {results[best]['accuracy']:.4f})"
    )
    lines.append("")
    lines.append(
        "NOTE: demo metrics on synthetic data only — see README.md for the "
        "separate production results from Google."
    )
    return "\n".join(lines)


def save_metrics_json(results: dict[str, dict], path: str) -> None:
    """Write the metrics dict to ``path`` as JSON."""
    with open(path, "w") as fh:
        json.dump(results, fh, indent=2)
