# TransitCheck

Classifying Kepler Objects of Interest as **confirmed planets** or **false positives**
from engineered light-curve features, using a gradient-boosted tree.

The question is not "can this discover a new planet". Every object already has a
disposition assigned by the Kepler team. The question is whether a machine-learning
pre-filter can **reduce the human review burden** in citizen-science transit vetting.

---

## Headline result

On a held-out test set of **434 KOIs**, split by star so no star appears in both
training and test:

| class | precision | recall |
|---|---|---|
| FALSE POSITIVE | 0.94 | 0.80 |
| CONFIRMED | 0.66 | 0.88 |

**117 of 133 real planets found. 16 missed. 61 false alarms.** ROC AUC 0.914 on this
split, 0.933 ± 0.008 across 25 further star-grouped splits.

Read as a review filter — the framing that matches the question:

> Of 434 KOIs, the model auto-rejects **256 (59% of the queue)** and loses
> **16 of 133 real planets (12%)** in doing so. A human reviews the remaining 178.

For honest context: [AstroNet](https://iopscience.iop.org/article/10.3847/1538-3881/aa9e09)
(Shallue & Vanderburg 2018) reaches AUC 0.988 on the same mission and the same
question, trained on 15,737 examples against 2,224 here. That gap is real. See
[METHODOLOGY.md §7](METHODOLOGY.md) for why Planet Hunters TESS is *not* a fair
comparison despite being the project's original benchmark.

---

## What's here

| file | what it is |
|---|---|
| `transitcheck_features.py` | the extraction pipeline — download, clean, flatten, fold, bin, measure |
| `run_1000.py`, `run_batch2.py`, `drive_batch2.sh` | batch runners with checkpointing and resource guards |
| `features_2000.csv` | the feature table: 2,289 rows / 2,000 stars |
| `transitCheck_training_ipynb.ipynb` | model training and evaluation (Colab) |
| `transitcheck_model.joblib` | the trained model |
| `analyse_run.py`, `diagnose_depth_bias.py`, `plot_depth_vs_occupancy.py` | analysis and diagnostics |
| `verify_model.py`, `verify_discrepancies.py` | reproduce the reported metrics and test their significance |
| `METHODOLOGY.md` | full writeup: pipeline, model, results, limitations |
| `CLAUDE.md` | design reasoning and a detailed correction log |

---

## Reproducing

Requires Python 3.11+, `lightkurve`, `pandas`, `numpy`, `scikit-learn`, `matplotlib`.

```bash
# extraction (downloads from MAST; ~6-9 hours and ~7 GB of cache per 1000 stars)
python transitcheck_features.py          # dry run: sample 100 stars, no downloads
python transitcheck_features.py --run    # extract 100 stars
python run_1000.py --run                 # first 1,000-star batch
bash drive_batch2.sh                     # second batch, chunked with disk/RAM guards

# analysis
python analyse_run.py features_2000.csv  # full report
python verify_model.py                   # reproduce the model and its metrics
python verify_discrepancies.py           # multi-split significance checks
```

Extraction is **resumable**. Every row is checkpointed as it completes, failures are
classified (`transient`, `nodata`, `diskfull`, `corruptcache`, `precheck`), and
re-running skips completed work while retrying only what should be retried.

The catalogue (`MyProject_sync.csv`) is the NASA Exoplanet Archive KOI cumulative
table, included so the pipeline runs without a separate download.

---

## Honest limitations

- **`asymmetry` carries no measurable signal** — importance +0.0066 ± 0.0090, smaller
  than its own uncertainty, and flat from 1,118 rows to 2,224. Transit asymmetry is
  not measurable at Kepler noise levels with this method. Kept and reported as a
  negative result.
- **Measured depth runs 8.8% shallow** against the catalogue, stably. Definitional
  (median of binned flux vs the minimum of a fitted model), uniform, and invisible to
  a tree. Diagnosed and deliberately not corrected.
- **2,289 of 7,587 available rows used** — extraction is download-bound. Doubling from
  1,000 to 2,000 stars moved accuracy by +0.003, which is the main evidence that the
  sample is not badly unrepresentative.
- **The top-ranked feature is catalogue-derived**, not from the light curve.
  `duration_over_period` alone scores 0.70 against a 0.69 majority-class baseline, so
  it is not carrying the model by itself — but it ranks first, and that is worth
  knowing.

[METHODOLOGY.md §9](METHODOLOGY.md) documents three development bugs that each
produced a *plausible number correlated with the label* rather than crashing. That
section is the most transferable thing in this repository.
