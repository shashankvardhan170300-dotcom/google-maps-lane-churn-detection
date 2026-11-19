"""Churn classifiers: a gradient-boosting model plus a logistic-regression
baseline.

The production pipeline this demo is modelled on used ML detection over
per-trace features; here a GradientBoostingClassifier is the primary
model and a (scaled) LogisticRegression is the transparent baseline it
is compared against. Recall is the metric that matters most for this
problem: a missed churn event leaves a stale lane on the map, while a
false alarm only costs a review.
"""

from __future__ import annotations

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def build_models(random_state: int = 42) -> dict[str, Pipeline]:
    """Return the (unfitted) models keyed by name.

    Gradient boosting takes the raw features; logistic regression gets a
    standard scaler first because it is sensitive to feature scale.
    """
    return {
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.08,
            max_depth=3,
            subsample=0.9,
            random_state=random_state,
        ),
        "logistic_regression": Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(
                        max_iter=1000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        ),
    }


def train_models(models: dict, X_train, y_train) -> dict:
    """Fit each model on the training split and return the fitted dict."""
    fitted = {}
    for name, model in models.items():
        fitted[name] = model.fit(X_train, y_train)
    return fitted
