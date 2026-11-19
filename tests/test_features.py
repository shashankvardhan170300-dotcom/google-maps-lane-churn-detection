"""Tests for per-trace feature engineering (``src.features``)."""

import numpy as np
import pandas as pd

from src.data.synth_traces import generate_segment_traces
from src.features.trace_features import FEATURE_COLUMNS, build_trace_features


def _handmade_trace() -> pd.DataFrame:
    """A tiny trace with a constant +3 m offset and steady speed, so the
    expected feature values are known exactly."""
    n = 10
    return pd.DataFrame(
        {
            "trace_id": ["t1"] * n,
            "segment_id": ["s1"] * n,
            "source": ["aGMM"] * n,
            "point_idx": list(range(n)),
            "timestamp_s": [float(i) for i in range(n)],
            "x_m": [float(i * 10) for i in range(n)],
            "y_m": [3.0] * n,
            "speed_mps": [15.0] * n,
            "heading_deg": [90.0] * n,
            "gnss_offset_m": [3.0] * n,
            "label": [1] * n,
        }
    )


def test_one_row_per_trace_with_all_features():
    traces = generate_segment_traces(
        n_segments=10, traces_per_segment=4, points_per_trace=25, seed=11
    )
    features = build_trace_features(traces)
    assert len(features) == traces["trace_id"].nunique() == 40
    assert set(features.columns) == {"trace_id", "segment_id", "label"} | set(
        FEATURE_COLUMNS
    )
    assert features["label"].isin([0, 1]).all()


def test_no_missing_or_infinite_values():
    traces = generate_segment_traces(n_segments=30, seed=12)
    features = build_trace_features(traces)
    assert not features[FEATURE_COLUMNS].isna().any().any()
    assert np.isfinite(features[FEATURE_COLUMNS].to_numpy()).all()


def test_constant_offset_trace_features():
    """A trace sitting exactly 3 m off the centreline: offset stats must
    reflect that, drift and heading change must be zero."""
    features = build_trace_features(_handmade_trace())
    row = features.iloc[0]
    assert row["offset_mean"] == 3.0
    assert row["offset_std"] == 0.0
    assert row["offset_abs_mean"] == 3.0
    assert row["offset_max_abs"] == 3.0
    assert row["offset_frac_gt_2m"] == 1.0
    assert abs(row["offset_drift"]) < 1e-9  # polyfit on constant data: ~1e-15
    assert row["heading_change_mean_abs"] == 0.0
    assert row["speed_mean"] == 15.0
    assert row["n_points"] == 10
    assert row["source_is_igmm"] == 0


def test_churn_traces_show_stronger_offset_features():
    """Feature signal check: churned traces should, on average, have a
    larger mean absolute offset and larger drift than stable traces."""
    traces = generate_segment_traces(n_segments=100, seed=42)
    features = build_trace_features(traces)
    churn = features[features["label"] == 1]
    stable = features[features["label"] == 0]
    assert churn["offset_abs_mean"].mean() > stable["offset_abs_mean"].mean()
    assert (
        churn["offset_drift"].abs().mean() > stable["offset_drift"].abs().mean()
    )


def test_missing_column_raises():
    traces = generate_segment_traces(n_segments=2, seed=1).drop(
        columns=["gnss_offset_m"]
    )
    try:
        build_trace_features(traces)
    except ValueError as exc:
        assert "gnss_offset_m" in str(exc)
    else:
        raise AssertionError("expected ValueError for missing column")
