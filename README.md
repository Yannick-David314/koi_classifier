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

The model uses **six light-curve features only** — no quantity from the catalogue's
transit-model fit reaches the classifier.

| class | precision | recall |
|---|---|---|
| FALSE POSITIVE | 0.89 | 0.74 |
| CONFIRMED | 0.58 | 0.80 |

**106 of 133 real planets found. 27 missed. 78 false alarms.** ROC AUC 0.856 on this
split, 0.876 ± 0.012 across 25 further star-grouped splits.

Read as a review filter — the framing that matches the question:

> Of 434 KOIs, the model auto-rejects **250 (58% of the queue)** and loses
> **27 of 133 real planets (20%)** in doing so. A human reviews the remaining 184.

### The result that matters more than the classifier

| feature set | accuracy (25 splits) | ROC AUC |
|---|---|---|
| **light-curve only (6)** | **0.7825 ± 0.0134** | 0.8764 |
| light-curve + catalogue geometry (8) | 0.8959 ± 0.0107 | 0.9595 |
| **catalogue only (2)** | **0.7748 ± 0.0162** | 0.8552 |

Six features extracted from four years of photometry are **statistically
indistinguishable** from two numbers read out of a catalogue row
(+0.008 ± 0.019, t = 2.0). An earlier version of this README claimed the light-curve
features carried the performance; the ablation shows they do not.

They are, however, **complementary**: 0.78 and 0.77 separately, **0.90 together**. The
model deliberately excludes the catalogue features so that this comparison is genuine
rather than a model measured against a subset of itself.

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
| `train_model.py` | trains and saves the final six-feature model |
| `ablate_features.py` | the feature-family ablation above |
| `transitcheck_model.joblib` | the trained model (model + feature list + threshold) |
| `transitCheck_training_ipynb.ipynb` | original Colab training run — **superseded**: it uses the earlier seven-feature set that included `duration_over_period` |
| `analyse_run.py`, `diagnose_depth_bias.py`, `plot_depth_vs_occupancy.py` | analysis and diagnostics |
| `verify_model.py`, `verify_discrepancies.py` | reproduce the earlier reported metrics and test their significance |
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
python analyse_run.py features_2000.csv  # full extraction report
python train_model.py                    # train the final model, print its metrics
python ablate_features.py                # feature-family ablation, 25 paired splits
python verify_discrepancies.py           # multi-split significance checks
```

Extraction is **resumable**. Every row is checkpointed as it completes, failures are
classified (`transient`, `nodata`, `diskfull`, `corruptcache`, `precheck`), and
re-running skips completed work while retrying only what should be retried.

The catalogue (`MyProject_sync.csv`) is the NASA Exoplanet Archive KOI cumulative
table, included so the pipeline runs without a separate download.

---

## Honest limitations

- **`asymmetry` carries no measurable signal** — importance +0.0043 ± 0.0129, smaller
  than its own uncertainty, and flat from 1,118 rows to 2,224. Transit asymmetry is
  not measurable at Kepler noise levels with this method. Kept and reported as a
  negative result.
- **Measured depth runs 8.8% shallow** against the catalogue, stably. Definitional
  (median of binned flux vs the minimum of a fitted model), uniform, and invisible to
  a tree. Diagnosed and deliberately not corrected.
- **2,289 of 7,587 available rows used** — extraction is download-bound. Doubling from
  1,000 to 2,000 stars moved accuracy by +0.003, which is the main evidence that the
  sample is not badly unrepresentative.
- **Light-curve features alone are not better than the catalogue.** See the ablation
  above. This is the project's most uncomfortable finding, and the reason the model
  excludes catalogue features — any other configuration hides it.

[METHODOLOGY.md §9](METHODOLOGY.md) documents three development bugs that each
produced a *plausible number correlated with the label* rather than crashing. That
section is the most transferable thing in this repository.
