"""Per-trace feature engineering for lane-churn detection.

Design mirrors the production "per-trace streaming" approach: each
trace is processed independently as it streams in, and a fixed-size
feature vector is emitted per trace. No feature looks across traces —
a trace's features depend only on that trace's own fixes — so the
computation is embarrassingly parallel and stateful only within a
trace, exactly as in a per-trace streaming job (e.g. Apache Flume).

Features
--------
Offset (lateral GNSS-to-map offset) statistics — the core churn
signal:

- ``offset_mean`` / ``offset_std`` / ``offset_median``
- ``offset_abs_mean`` / ``offset_max_abs`` / ``offset_p95_abs``
- ``offset_frac_gt_2m``: fraction of fixes further than 2 m from the
  mapped centreline
- ``offset_drift``: slope of offset vs. along-track distance (metres
  of lateral drift per 100 m travelled) — a systematic ramp is the
  signature of a lane that moved relative to the map

Heading behaviour:

- ``heading_change_mean_abs`` / ``heading_change_std`` /
  ``heading_change_max_abs``: statistics of turn-to-turn heading
  changes (circular difference, degrees)

Motion:

- ``speed_mean`` / ``speed_std``
- ``n_points`` / ``duration_s``: trace length and duration

Provenance:

- ``source_is_igmm``: 1 when the trace came from the iGMM source,
  0 for aGMM

The segment churn label is carried through as ``label`` (target).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

#: lateral-offset threshold (m) used by ``offset_frac_gt_2m``.
OFFSET_ALERT_M = 2.0

FEATURE_COLUMNS = [
    "offset_mean",
    "offset_std",
    "offset_median",
    "offset_abs_mean",
    "offset_max_abs",
    "offset_p95_abs",
    "offset_frac_gt_2m",
    "offset_drift",
    "heading_change_mean_abs",
    "heading_change_std",
    "heading_change_max_abs",
    "speed_mean",
    "speed_std",
    "n_points",
    "duration_s",
    "source_is_igmm",
]


def _circular_diff_deg(angles: np.ndarray) -> np.ndarray:
    """Consecutive heading changes in degrees, wrapped to [-180, 180]."""
    diff = np.diff(np.asarray(angles, dtype=float))
    return (diff + 180.0) % 360.0 - 180.0


def build_trace_features(traces: pd.DataFrame) -> pd.DataFrame:
    """Compute one feature row per trace.

    Parameters
    ----------
    traces:
        Per-fix table as produced by
        :func:`src.data.synth_traces.generate_segment_traces`.

    Returns
    -------
    pandas.DataFrame with one row per ``trace_id`` and columns
    ``trace_id``, ``segment_id``, ``label`` plus ``FEATURE_COLUMNS``.
    """
    required = {
        "trace_id",
        "segment_id",
        "source",
        "timestamp_s",
        "x_m",
        "gnss_offset_m",
        "heading_deg",
        "speed_mps",
        "label",
    }
    missing = required - set(traces.columns)
    if missing:
        raise ValueError(f"trace table is missing columns: {sorted(missing)}")

    feature_rows: list[dict] = []
    for (trace_id, segment_id), group in traces.groupby(
        ["trace_id", "segment_id"], sort=True
    ):
        group = group.sort_values("point_idx")
        offset = group["gnss_offset_m"].to_numpy(dtype=float)
        heading = group["heading_deg"].to_numpy(dtype=float)
        speed = group["speed_mps"].to_numpy(dtype=float)
        along = group["x_m"].to_numpy(dtype=float)
        heading_change = _circular_diff_deg(heading)

        if len(along) > 1 and np.ptp(along) > 0:
            # Metres of lateral drift per 100 m travelled.
            slope = float(np.polyfit(along, offset, 1)[0])
            drift = slope * 100.0
        else:
            drift = 0.0

        feature_rows.append(
            {
                "trace_id": trace_id,
                "segment_id": segment_id,
                "label": int(group["label"].iloc[0]),
                "offset_mean": float(np.mean(offset)),
                "offset_std": float(np.std(offset)),
                "offset_median": float(np.median(offset)),
                "offset_abs_mean": float(np.mean(np.abs(offset))),
                "offset_max_abs": float(np.max(np.abs(offset))),
                "offset_p95_abs": float(np.percentile(np.abs(offset), 95)),
                "offset_frac_gt_2m": float(
                    np.mean(np.abs(offset) > OFFSET_ALERT_M)
                ),
                "offset_drift": drift,
                "heading_change_mean_abs": float(
                    np.mean(np.abs(heading_change))
                )
                if len(heading_change)
                else 0.0,
                "heading_change_std": float(np.std(heading_change))
                if len(heading_change)
                else 0.0,
                "heading_change_max_abs": float(
                    np.max(np.abs(heading_change))
                )
                if len(heading_change)
                else 0.0,
                "speed_mean": float(np.mean(speed)),
                "speed_std": float(np.std(speed)),
                "n_points": int(len(group)),
                "duration_s": float(
                    group["timestamp_s"].max() - group["timestamp_s"].min()
                ),
                "source_is_igmm": int((group["source"] == "iGMM").iloc[0]),
            }
        )

    features = pd.DataFrame(feature_rows)
    ordered = ["trace_id", "segment_id", "label"] + FEATURE_COLUMNS
    return features[ordered]
