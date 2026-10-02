# TransitCheck — Roadmap & Status

*Updated after restricting the model to six light-curve features (CLAUDE.md §11).*

## You are here: Phase 3 core complete.

You have a trained, honestly-evaluated model. What remains in Phase 3 is
communicating the result, not improving it.

**Final result (six light-curve features):** 0.80 recall, 0.58 precision on
CONFIRMED planets, threshold 0.25, on a star-grouped held-out test set of 434 rows.
106 of 133 real planets found · 27 missed · 78 false alarms · ROC AUC 0.856.
Reproduced from the saved model on 2026-10-01 under Python 3.14 / scikit-learn 1.9.0.

*(The previous figures here -- 0.88 recall, 0.66 precision, 117 of 133 found -- came
from the seven-feature model that included `duration_over_period`, a catalogue
quantity. See §11 of CLAUDE.md for why it was removed.)*

---

## Phase 0 — Framing
- [x] Chose the project, tied to real interests
- [x] Reframed to an answerable question: can an ML pre-filter reduce human review burden for citizen-science transit vetting?
- [x] Identified Planet Hunters TESS as the benchmark

## Phase 1 — Learn the tools on one star (Kepler-10) — ✅ complete
- [x] Environment set up; pulled and plotted a real light curve
- [x] Learned folding, binning, flattening, and the window-length-vs-cadence trap
- [x] Correctly found Kepler-10b's transit by hand

## Phase 2 — Build the data pipeline — ✅ complete
- [x] Six engineered features, each reasoned from transit physics
- [x] Depth tuned against `koi_depth` — median |error| ~10%
- [x] Transit-masked flattening, epoch centering, local + global views
- [x] `secondary_depth` variant question settled with data (`min2`, 3.81× discrimination)
- [x] Multi-KOI contamination hypothesis confirmed and fixed via sibling masking
- [x] Error handling, caching, checkpointing, throttling, cache eviction
- [x] Scaled to 2,000 stars / 2,289 rows across two runs
- [x] X/y assembled, leakage columns excluded and asserted
- [x] Grouped train/test split verified disjoint

## Phase 3 — Train & evaluate — ✅ core complete
- [x] `HistGradientBoostingClassifier` fit on grouped split
- [x] NaNs left unimputed — missingness is informative
- [x] Threshold chosen on a validation split, not on test
- [x] Precision/recall reported, not accuracy
- [x] Permutation importances computed on the test set
- [x] ~~**Catalogue-only baseline run** — geometry alone scores 0.63 accuracy (base rate 0.64); the light-curve pipeline carries the performance~~ — **wrong**: that baseline was one column, a subset of the model it was compared against (CLAUDE.md §11.6)
- [x] **Feature-set ablation** over 25 star-grouped splits: light-curve only 0.7825, catalogue only (2 columns) 0.7748 — statistically indistinguishable; together 0.8959
- [x] Model restricted to the six light-curve features, so the comparison above is genuine
- [x] Model and notebook saved
- [ ] Benchmark against Planet Hunters TESS's published precision/recall
- [ ] Document methodology and limitations

### Improvement levers — all three tested, all null
| lever | result |
|---|---|
| More data (1,000 → 2,000 stars) | No change in recall |
| More features (`koi_impact`) | Within noise (±3 pts) |
| Hyperparameter tuning | CV 0.8806 vs 0.8754 default — not adopted |

*These were tested on the earlier seven-feature model, before `duration_over_period`
was removed. They have not been re-run on the six-feature model.*

Three independent signals that the model isn't underfit or starved. It extracts
close to all the signal these seven features contain. Further gains would need a
different *kind* of feature, not more of the same.

### Feature importances (six-feature model, test set)
```
depth                  +0.0627 ± 0.0154
secondary_depth        +0.0541 ± 0.0130
snr                    +0.0300 ± 0.0142
odd_even_diff          +0.0190 ± 0.0122
width                  +0.0185 ± 0.0156
asymmetry              +0.0043 ± 0.0129
```
With `duration_over_period` gone, `depth` takes the top spot. `asymmetry` stays
within noise of zero, consistent with the finding below.

### Findings worth reporting
- **`asymmetry` is not measurable at Kepler noise levels.** Flat at 1,118 rows, still flat at 2,224. Relative asymmetry tracks SNR near-monotonically — it measures noise, not geometry. A real negative result.
- **`width` came off the floor with more data** (−0.0005 → +0.0185, three times its spread). The earlier null was sample size.
- **Depth bias is −8.8%, stable across 2,000 stars.** Definitional: median-over-window vs the catalogue's model-fit minimum. Uniform, so invisible to a tree.
- **Three bugs produced plausible label-correlated numbers rather than crashes:** `koi_fpflag_*` as features, CANDIDATE rows labelled negative, and `MaskedNDArray` coercing to 0.0. Each would have made the model look *better*.

## Phase 4 — Backend (candidate queue, database) — ⬜ not started
The natural next step: run the pipeline on the 1,977 CANDIDATE rows, score them,
rank by probability. That's the first output a human could act on — and the thing
the project was for.

## Phase 5 — Frontend review UI, pilot with your club — ⬜ not started
