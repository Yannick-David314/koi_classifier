# TransitCheck — Roadmap & Status

*Updated after training on the 2,000-star dataset.*

## You are here: Phase 3 core complete.

You have a trained, honestly-evaluated model. What remains in Phase 3 is
communicating the result, not improving it.

**Final result:** 0.88 recall, 0.66 precision on CONFIRMED planets, threshold
0.25, on a star-grouped held-out test set of 434 rows.
117 of 133 real planets found · 16 missed · 61 false alarms.

---

## Phase 0 — Framing
- [x] Chose the project, tied to real interests
- [x] Reframed to an answerable question: can an ML pre-filter reduce human review burden for citizen-science transit vetting?
- [x] Identified Planet Hunters TESS as the benchmark
- [ ] Reach out to a faculty mentor *(still recommended)*
- [ ] Read PHT's published methodology in full

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
- [x] **Catalogue-only baseline run** — geometry alone scores 0.63 accuracy (base rate 0.64); the light-curve pipeline carries the performance
- [x] Model and notebook saved
- [ ] Benchmark against Planet Hunters TESS's published precision/recall
- [ ] Document methodology and limitations

### Improvement levers — all three tested, all null
| lever | result |
|---|---|
| More data (1,000 → 2,000 stars) | No change in recall |
| More features (`koi_impact`) | Within noise (±3 pts) |
| Hyperparameter tuning | CV 0.8806 vs 0.8754 default — not adopted |

Three independent signals that the model isn't underfit or starved. It extracts
close to all the signal these seven features contain. Further gains would need a
different *kind* of feature, not more of the same.

### Feature importances (2,224 rows, test set)
```
duration_over_period   +0.1248 ± 0.0138
secondary_depth        +0.0620 ± 0.0099
snr                    +0.0446 ± 0.0110
depth                  +0.0225 ± 0.0099
width                  +0.0185 ± 0.0060
odd_even_diff          +0.0158 ± 0.0118
asymmetry              +0.0066 ± 0.0090
```

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
