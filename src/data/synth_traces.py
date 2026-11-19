"""Synthetic GPS trace generator for lane-churn detection demos.

All data produced here is SYNTHETIC, generated from scratch for this
portfolio project. It models the *shape* of the real problem — GNSS
traces observed against mapped lane geometry — without using any real
map data, real traces, or proprietary code.

Domain model
------------
A "lane segment" is a stretch of road whose mapped lane geometry is
either stable or has *churned* (the real-world lanes changed: a lane
was added/removed or the geometry shifted, so the map no longer matches
what drivers experience). Each segment is observed by several GPS
"traces" (one vehicle pass = one trace). A trace is a sequence of
position samples with:

- ``trace_id`` / ``segment_id``: which pass, which road segment
- ``source``: the trace source, ``aGMM`` or ``iGMM`` (the two trace
  sources used in the production pipeline this demo is modelled on)
- ``timestamp_s``: seconds since the trace started
- ``x_m`` / ``y_m``: local metres along/across the road
- ``speed_mps``: vehicle speed
- ``heading_deg``: direction of travel, derived from positions
- ``gnss_offset_m``: lateral offset of the GNSS fix from the mapped
  lane centreline — the key churn signal
- ``label``: 1 if the segment has churned, 0 if it is stable

Churn signature (what the detector learns): on a churned segment the
lateral offset develops a systematic bias that grows along the segment
(the map and the road diverge), heading becomes more erratic, and a
large fraction of fixes sit far from the mapped centreline. Stable
segments show offsets centred on zero with small noise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SOURCES = ("aGMM", "iGMM")

#: metres of lateral offset beyond which a fix is counted as "far off"
#: the mapped centreline.
OFFSET_ALERT_M = 2.0


def generate_segment_traces(
    n_segments: int = 200,
    traces_per_segment: int = 8,
    points_per_trace: int = 60,
    churn_fraction: float = 0.35,
    segment_length_m: float = 400.0,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic trace table.

    Parameters
    ----------
    n_segments:
        Number of road segments to simulate.
    traces_per_segment:
        Vehicle passes observed per segment.
    points_per_trace:
        GNSS fixes per trace.
    churn_fraction:
        Share of segments whose lane geometry has churned.
    segment_length_m:
        Length of each segment in metres.
    seed:
        Random seed — same seed always produces identical data.

    Returns
    -------
    pandas.DataFrame with one row per GNSS fix. Columns: ``trace_id``,
    ``segment_id``, ``source``, ``point_idx``, ``timestamp_s``,
    ``x_m``, ``y_m``, ``speed_mps``, ``heading_deg``,
    ``gnss_offset_m``, ``label``.
    """
    rng = np.random.default_rng(seed)

    # Segment-level truth: which segments churned.
    n_churn = int(round(n_segments * churn_fraction))
    churned = np.zeros(n_segments, dtype=bool)
    churned[:n_churn] = True
    rng.shuffle(churned)

    along = np.linspace(0.0, segment_length_m, points_per_trace)
    rows: list[dict] = []

    for seg in range(n_segments):
        is_churn = bool(churned[seg])
        # Churned segments: the road moved relative to the map, so the
        # lateral offset has a systematic bias that grows along the
        # segment. Stable segments: offset stays centred on zero.
        # Some churn is early-stage (weak signal); some stable segments
        # have a temporary excursion (e.g. a roadworks detour) that
        # looks churn-like for a stretch. Both make the demo problem
        # realistically hard instead of trivially separable.
        weak_churn = is_churn and bool(rng.random() < 0.30)
        excursion = (not is_churn) and bool(rng.random() < 0.10)
        if is_churn:
            bias = rng.uniform(0.8, 3.5) * rng.choice([-1.0, 1.0])
            if weak_churn:
                bias *= rng.uniform(0.3, 0.6)
            drift = rng.uniform(0.2, 2.0) * np.sign(bias)
            offset_noise = rng.uniform(0.9, 1.5)
            heading_noise = rng.uniform(2.0, 3.0)
        else:
            bias = rng.uniform(-0.5, 0.5)
            drift = rng.uniform(-0.3, 0.3)
            offset_noise = rng.uniform(0.6, 1.1)
            heading_noise = rng.uniform(0.6, 1.2)

        for t in range(traces_per_segment):
            trace_id = f"trace-{seg:04d}-{t:02d}"
            source = SOURCES[int(rng.integers(0, len(SOURCES)))]
            # Per-trace jitter so traces on the same segment differ.
            trace_bias = bias + rng.normal(0.0, 0.25)
            trace_drift = drift + rng.normal(0.0, 0.15)

            offsets = (
                trace_bias
                + trace_drift * (along / segment_length_m)
                + rng.normal(0.0, offset_noise, size=points_per_trace)
            )
            # Multipath spikes: a few fixes jump several metres (urban
            # canyon / reflected signals). Affects churned and stable
            # traces alike, so max/p95 features alone cannot separate
            # the classes.
            spike = rng.random(points_per_trace) < 0.04
            n_spike = int(spike.sum())
            if n_spike:
                offsets[spike] += (
                    rng.choice([-1.0, 1.0], size=n_spike)
                    * rng.uniform(3.0, 6.0, size=n_spike)
                )
            # Temporary excursion on some stable segments (roadworks
            # detour): a stretch of fixes sits off the centreline even
            # though the lane geometry has not churned.
            if excursion:
                lo, hi = points_per_trace // 3, points_per_trace // 2
                offsets[lo:hi] += 2.5
            speed = np.clip(
                rng.normal(15.0, 3.0, size=points_per_trace), 5.0, 25.0
            )
            # Heading derived from successive positions plus noise.
            heading = np.empty(points_per_trace)
            heading[0] = 90.0
            dy = np.diff(offsets)
            dx = np.diff(along)
            heading[1:] = 90.0 + np.degrees(np.arctan2(dy, dx))
            heading += rng.normal(0.0, heading_noise, size=points_per_trace)
            heading = (heading + 180.0) % 360.0 - 180.0

            # Timestamps from speed and point spacing.
            spacing = segment_length_m / max(points_per_trace - 1, 1)
            dt = spacing / np.maximum(speed, 1.0)
            timestamps = np.concatenate([[0.0], np.cumsum(dt)[:-1]])

            for i in range(points_per_trace):
                rows.append(
                    {
                        "trace_id": trace_id,
                        "segment_id": f"segment-{seg:04d}",
                        "source": source,
                        "point_idx": i,
                        "timestamp_s": round(float(timestamps[i]), 3),
                        "x_m": round(float(along[i]), 2),
                        "y_m": round(float(offsets[i]), 3),
                        "speed_mps": round(float(speed[i]), 2),
                        "heading_deg": round(float(heading[i]), 2),
                        "gnss_offset_m": round(float(offsets[i]), 3),
                        "label": int(is_churn),
                    }
                )

    return pd.DataFrame(rows)


def generate(
    n_segments: int = 200,
    traces_per_segment: int = 8,
    points_per_trace: int = 60,
    churn_fraction: float = 0.35,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Generate traces plus a segment-level summary table.

    Returns
    -------
    (traces, segments): the per-fix trace table and one row per segment
    with its churn label and trace count.
    """
    traces = generate_segment_traces(
        n_segments=n_segments,
        traces_per_segment=traces_per_segment,
        points_per_trace=points_per_trace,
        churn_fraction=churn_fraction,
        seed=seed,
    )
    segments = (
        traces.groupby("segment_id")["label"]
        .agg(label="first", n_traces="size")
        .reset_index()
    )
    segments["n_traces"] = segments["n_traces"] // points_per_trace
    return traces, segments
