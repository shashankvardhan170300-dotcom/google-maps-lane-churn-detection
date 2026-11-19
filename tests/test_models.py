"""Smoke tests for model training and evaluation (``src.models``,
``src.evaluate``).
"""

from sklearn.model_selection import train_test_split

from src.data.synth_traces import generate_segment_traces
from src.evaluate import evaluate_models, format_report
from src.features.trace_features import FEATURE_COLUMNS, build_trace_features
from src.models.churn_classifier import build_models, train_models


def _small_dataset():
    traces = generate_segment_traces(n_segments=60, seed=21)
    features = build_trace_features(traces)
    X = features[FEATURE_COLUMNS]
    y = features["label"]
    return train_test_split(X, y, test_size=0.25, random_state=21, stratify=y)


def test_models_train_and_predict():
    X_train, X_test, y_train, y_test = _small_dataset()
    models = train_models(build_models(), X_train, y_train)
    assert set(models) == {"gradient_boosting", "logistic_regression"}
    for model in models.values():
        preds = model.predict(X_test)
        assert set(preds.tolist()) <= {0, 1}
        assert len(preds) == len(y_test)


def test_metrics_are_valid_and_recall_is_strong():
    """The synthetic churn signal is strong by design, so the demo
    models should clear a modest recall floor on held-out traces."""
    X_train, X_test, y_train, y_test = _small_dataset()
    models = train_models(build_models(), X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    for name, m in results.items():
        for key in ("accuracy", "precision", "recall", "f1"):
            assert 0.0 <= m[key] <= 1.0, f"{name}.{key} out of range"
        assert m["n_test"] == len(y_test)
        cm = m["confusion_matrix"]
        assert sum(cm.values()) == len(y_test)
    assert results["gradient_boosting"]["recall"] >= 0.70


def test_report_renders():
    X_train, X_test, y_train, y_test = _small_dataset()
    models = train_models(build_models(), X_train, y_train)
    results = evaluate_models(models, X_test, y_test)
    report = format_report(results)
    assert "gradient_boosting" in report
    assert "logistic_regression" in report
    assert "recall" in report
