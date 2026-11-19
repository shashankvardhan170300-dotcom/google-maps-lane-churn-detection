# Google Maps Lane-Churn Detection

A lane-churn detection ML pipeline in Python: synthetic GNSS traces in,
per-trace features, a churn classifier, and an evaluation report out.

## What this is — read first

This is a **portfolio reconstruction** built from scratch on **synthetic
data**. It demonstrates the *approach* Shashank Narayandas used at Google
— it is **not Google code, not Google data, and not a copy of any
production system**. No proprietary code, map data, or real traces are
used anywhere in this repository.

**Production results vs. demo results — do not mix these up:**

| | |
|---|---|
| **Production outcome at Google** (ML Engineer, Contract, May 2024 – Apr 2025) | Leading two new trace sources (**aGMM** and **iGMM**) into the lane-churn detection pipeline and rebuilding trace processing into per-trace Flume streaming increased global lane-churn detection **recall by 20%** and **accuracy by 14%**. These are real production results from his Google work. |
| **Demo results in this repo** | The models below score **recall 0.92 / accuracy 0.96** on **synthetic** data (see [Demo results](#demo-results)). These numbers describe the demo only. They are not the production figures above. |

## The problem

Google Maps has to know when the lanes on a road have changed — a lane
added or removed, geometry shifted — because a stale map sends drivers
the wrong way. The signal comes from GNSS traces: millions of vehicle
passes whose positions are compared against the mapped lane geometry.
When the observed traces systematically diverge from the map, the lane
has *churned* and the map needs updating.

The hard parts, and what this pipeline mirrors:

- **Per-trace streaming.** Traces arrive one at a time and are processed
  independently — in production this was rebuilt as per-trace Flume
  streaming, so no trace's features depend on any other trace.
- **Two trace sources.** The production pipeline ingested the **aGMM**
  and **iGMM** trace sources; the demo generates traces from both and
  carries the source as a feature.
- **The churn signature.** A churned lane shows a systematic lateral
  (GNSS-to-map) offset that grows along the segment, erratic heading,
  and many fixes far from the mapped centreline. Stable lanes sit close
  to the centreline with small noise.
- **Messy reality.** Real GNSS data has multipath spikes, early-stage
  churn with a weak signal, and temporary detours that look like churn.
  The synthetic generator includes all three, so the demo problem is
  realistically hard rather than trivially separable.

## How it works

```
synthetic GNSS traces          per-trace features           classifier
(src/data/synth_traces.py) --> (src/features/) -----------> (src/models/)
  trace_id, segment_id,          offset stats, drift,          GradientBoosting
  timestamps, x/y, speed,        heading changes, speed,       + LogisticRegression
  heading, GNSS offset,          source                        baseline
  source, churn label
                                     |
                                     v
                              evaluation report
                              (src/evaluate.py)
                              recall / precision /
                              accuracy / F1 /
                              confusion matrix
```

**Features (16 per trace)** — offset mean/std/median, mean/max/95th
percentile absolute offset, fraction of fixes > 2 m off the centreline,
offset drift (metres of lateral drift per 100 m travelled), heading-change
mean/std/max, speed mean/std, trace length/duration, and the trace
source. Recall is the metric that matters most here: a missed churn
event leaves a stale lane on the map; a false alarm only costs a review.

## Demo results

Default run: 200 segments × 8 traces, 25% held out as test (400 traces,
140 churned). Seed 42.

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| GradientBoosting | 0.9550 | 0.9485 | 0.9214 | 0.9348 |
| LogisticRegression (baseline) | 0.9550 | 0.9485 | 0.9214 | 0.9348 |

Confusion matrix (gradient boosting), `[[TN, FP], [FN, TP]]`:
`[[253, 7], [11, 129]]` — 11 missed churn events, 7 false alarms.

Both models converge to similar performance on this synthetic problem:
after per-trace aggregation the offset features separate the classes
well, and the boundary is close to linear. On noisier real-world data
the gradient-boosting model is the one expected to pull ahead. Metrics
are stable across seeds (recall 0.92–0.98 for seeds 42, 7, 123).

> Again: these are **demo metrics on synthetic data**. The production
> figures from Google are **+20% recall / +14% accuracy** — stated
> separately above, never as this demo's results.

## Quickstart

```bash
git clone <this-repo>
cd google-maps-lane-churn-detection

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# End-to-end: generate traces -> features -> train -> evaluate
python scripts/run_pipeline.py

# Options
python scripts/run_pipeline.py --n-segments 300 --seed 7 --out-dir outputs
```

Artifacts land in `outputs/` (`features.csv`, `metrics.json`).

## Tests

```bash
python -m pytest tests/ -v
```

14 tests, all passing: generator determinism (seeded output is
byte-identical), schema and churn-fraction checks, the core invariant
that churned segments sit further off the centreline, feature-builder
correctness on a hand-built trace (including exact expected values),
and model training/evaluation smoke tests with a recall floor.

## Project structure

```
google-maps-lane-churn-detection/
├── README.md
├── LICENSE                     # MIT
├── requirements.txt
├── notebooks/
│   └── 01_pipeline_walkthrough.ipynb
├── scripts/
│   └── run_pipeline.py         # end-to-end entry point
├── src/
│   ├── data/
│   │   └── synth_traces.py     # synthetic GPS trace generator
│   ├── features/
│   │   └── trace_features.py   # per-trace feature engineering
│   ├── models/
│   │   └── churn_classifier.py # GradientBoosting + LogReg baseline
│   └── evaluate.py             # metrics + report
└── tests/
    ├── test_data.py            # generator: determinism, schema, signal
    ├── test_features.py        # feature builder: exact values, invariants
    └── test_models.py          # training smoke test, metric validity
```

## The synthetic data

`src/data/synth_traces.py` generates every fix from scratch with a
seeded NumPy generator (same seed → identical data). Each segment is
labelled churned or stable; churned segments get a systematic lateral
offset bias that ramps along the segment, noisier headings, and (for
~30% of them) a weak early-stage signal; stable segments stay centred
on zero, with ~10% showing a temporary roadworks-style excursion.
~4% of fixes get a multipath spike of 3–6 m regardless of label.
Fields mirror a real trace feed: trace/segment IDs, source (aGMM/iGMM),
timestamps, local x/y in metres, speed, heading, GNSS offset, label.

## Author

**Shashank Narayandas** — Data & ML Engineer. At Google (ML Engineer,
Contract, May 2024 – Apr 2025) he designed and implemented LLM-based
detection algorithms for Google Maps, led the aGMM and iGMM trace
sources into the lane-churn detection pipeline (+20% recall / +14%
accuracy), and rebuilt trace processing into per-trace Flume streaming.
M.S. Data Science, Webster University. Claude Certified Architect
(Foundations).

- LinkedIn: https://www.linkedin.com/in/shashank-ai-ml
- Email: shashank.n.aiml@gmail.com

## License

MIT — see [LICENSE](LICENSE).
