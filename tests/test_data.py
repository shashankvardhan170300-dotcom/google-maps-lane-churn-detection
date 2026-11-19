"""Tests for the synthetic trace generator (``src.data.synth_traces``)."""

import numpy as np
import pandas as pd

from src.data.synth_traces import generate, generate_segment_traces

EXPECTED_COLUMNS = {
    "trace_id",
    "segment_id",
    "source",
    "point_idx",
    "timestamp_s",
    "x_m",
    "y_m",
    "speed_mps",
    "heading_deg",
    "gnss_offset_m",
    "label",
}


def test_schema_and_row_count():
    traces = generate_segment_traces(
        n_segments=10, traces_per_segment=4, points_per_trace=20, seed=1
    )
    assert set(traces.columns) == EXPECTED_COLUMNS
    assert len(traces) == 10 * 4 * 20
    assert traces["trace_id"].nunique() == 40
    assert set(traces["source"].unique()) <= {"aGMM", "iGMM"}
    assert set(traces["label"].unique()) <= {0, 1}


def test_deterministic_with_seed():
    a = generate_segment_traces(n_segments=20, seed=7)
    b = generate_segment_traces(n_segments=20, seed=7)
    pd.testing.assert_frame_equal(a, b)


def test_different_seed_gives_different_data():
    a = generate_segment_traces(n_segments=20, seed=7)
    b = generate_segment_traces(n_segments=20, seed=8)
    assert not a.equals(b)


def test_churn_segments_have_larger_offsets():
    """The generator's core invariant: churned segments sit further off
    the mapped centreline than stable ones."""
    traces = generate_segment_traces(n_segments=100, seed=42)
    per_segment = traces.groupby(["segment_id", "label"])["gnss_offset_m"].apply(
        lambda s: np.mean(np.abs(s))
    )
    churn_mean = per_segment.xs(1, level="label").mean()
    stable_mean = per_segment.xs(0, level="label").mean()
    assert churn_mean > 2.0 * stable_mean


def test_churn_fraction_is_honoured():
    _, segments = generate(n_segments=200, churn_fraction=0.35, seed=3)
    assert segments["label"].sum() == 70  # 35% of 200
    assert (segments["n_traces"] == 8).all()


def test_timestamps_increase_within_trace():
    traces = generate_segment_traces(
        n_segments=5, traces_per_segment=2, points_per_trace=30, seed=5
    )
    for _, group in traces.groupby("trace_id"):
        ts = group.sort_values("point_idx")["timestamp_s"].to_numpy()
        assert np.all(np.diff(ts) >= 0)
