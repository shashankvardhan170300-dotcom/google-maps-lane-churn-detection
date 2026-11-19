#!/usr/bin/env python3
"""End-to-end lane-churn detection pipeline demo.

Runs the full flow on synthetic data:

    1. generate synthetic GNSS traces (``src.data``)
    2. build per-trace features (``src.features``)
    3. train the churn classifiers (``src.models``)
    4. evaluate and print a report (``src.evaluate``)

Usage
-----
    python scripts/run_pipeline.py [--n-segments 200] [--seed 42]
                                   [--out-dir outputs]

Outputs (when ``--out-dir`` is given): ``features.csv`` (per-trace
feature table) and ``metrics.json`` (evaluation metrics).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.model_selection import train_test_split  # noqa: E402

from src.data.synth_traces import generate_segment_traces  # noqa: E402
from src.evaluate import evaluate_models, format_report, save_metrics_json  # noqa: E402
from src.features.trace_features import FEATURE_COLUMNS, build_trace_features  # noqa: E402
from src.models.churn_classifier import build_models, train_models  # noqa: E402


def run(n_segments: int = 200, seed: int = 42, out_dir: str | None = None) -> dict:
    """Run the pipeline and return the evaluation results dict."""
    print(f"[1/4] Generating synthetic traces ({n_segments} segments, seed={seed}) ...")
    traces = generate_segment_traces(n_segments=n_segments, seed=seed)
    n_churn = int(traces.groupby("segment_id")["label"].first().sum())
    print(f"      {len(traces):,} fixes across "
          f"{traces['trace_id'].nunique():,} traces; "
          f"{n_churn} churned segments, {n_segments - n_churn} stable.")

    print("[2/4] Building per-trace features ...")
    features = build_trace_features(traces)
    print(f"      {len(features):,} feature rows x {len(FEATURE_COLUMNS)} features.")

    print("[3/4] Training models ...")
    X = features[FEATURE_COLUMNS]
    y = features["label"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=seed, stratify=y
    )
    models = train_models(build_models(random_state=seed), X_train, y_train)
    print(f"      trained {len(models)} models on {len(X_train)} traces; "
          f"testing on {len(X_test)}.")

    print("[4/4] Evaluating ...")
    results = evaluate_models(models, X_test, y_test)
    print()
    print(format_report(results))

    if out_dir:
        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        features.to_csv(out / "features.csv", index=False)
        save_metrics_json(results, str(out / "metrics.json"))
        print(f"Artifacts written to {out}/ (features.csv, metrics.json)")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Lane-churn detection pipeline demo (synthetic data)."
    )
    parser.add_argument("--n-segments", type=int, default=200,
                        help="number of road segments to simulate")
    parser.add_argument("--seed", type=int, default=42,
                        help="random seed (deterministic output)")
    parser.add_argument("--out-dir", type=str, default="outputs",
                        help="directory for features.csv / metrics.json")
    args = parser.parse_args()
    run(n_segments=args.n_segments, seed=args.seed, out_dir=args.out_dir)


if __name__ == "__main__":
    main()
