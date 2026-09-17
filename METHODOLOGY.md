# TransitCheck — Methodology

Classifying Kepler Objects of Interest as confirmed planets or false positives from
engineered light-curve features.

---

## 1. What this project does, and what it does not

**The question.** Can a machine-learning pre-filter reduce the human review burden in
citizen-science transit vetting?

**Not** "discover a new planet". Every object here already has a disposition assigned
by the Kepler team. The model is asked to reproduce that judgement from the light
curve, so that a reviewer's time can be spent on the cases that need it.

That framing matters for how the results should be read. A classifier that removes
half the review queue while rarely discarding a real planet is useful even if its
accuracy is unremarkable, and a classifier with excellent accuracy that quietly drops
planets is not.

---

## 2. Data

| | |
|---|---|
| Catalogue | NASA Exoplanet Archive KOI cumulative table, 9,564 rows |
| Labelled subset | 7,587 rows (2,748 CONFIRMED, 4,839 FALSE POSITIVE) over 6,641 stars |
| Excluded | 1,977 CANDIDATE rows — the label is genuinely unknown, not negative |
| Light curves | Kepler long-cadence PDCSAP, downloaded from MAST via `lightkurve` |
| **Used here** | **2,289 rows over 2,000 stars; 2,224 extracted successfully** |

The 2,000 stars were drawn in two independent batches of 1,000, sampled by star rather
than by row, at the catalogue's natural class proportions (63.4% false positive). The
second batch excluded every `kepid` already present in the first.

### Why 2,289 of 7,587

Extraction is download-bound: roughly 6.8 GB of cached FITS and six to nine hours per
thousand stars. The batches were sized to what the hardware could do, not to a
statistical requirement. The gap between 2,289 and 7,587 is the largest single caveat
on everything below — see §8.

---

## 3. Pipeline

Per star: **download → clean → flatten → fold → bin → measure.**

1. **Download.** All available quarters, **pinned to long cadence (1800 s)**. Kepler
   returns 60 s short-cadence products for some targets, and stitching both together
   produces a light curve whose time axis runs backwards at the seam and whose median
   cadence reflects whichever product contributed more points. 24.9% of rows are on
   stars where this would have happened.
2. **Clean.** Sort by time, drop duplicate timestamps, then asymmetric outlier removal
   (`sigma_lower=20`, `sigma_upper=5`). Lenient downward on purpose: a real deep
   eclipse *is* a statistical outlier, and a symmetric cut shaves the bottom off it.
3. **Flatten.** Savitzky–Golay detrending over a window of three transit durations,
   sized from each star's own measured cadence. The transit is masked out of the trend
   fit, so the fit has no reason to dip toward it.
4. **Fold and bin.** Fold on the catalogue ephemeris, cut a window of four transit
   durations, bin into a nominal 201 bins.
5. **Measure.** Seven features, below.

Transit times are always computed arithmetically from the catalogue's period and
epoch, never found by searching the flux for dips. Searching would catch every cosmic
ray and miss every shallow transit.

---

## 4. Features

| feature | what it measures | source |
|---|---|---|
| `depth` | median in-transit flux vs out-of-transit baseline | light curve |
| `width` | longest contiguous run of bins below half-depth | light curve |
| `asymmetry` | ingress-side depth minus egress-side depth, signed | light curve |
| `snr` | depth divided by MAD-based out-of-transit noise | light curve |
| `odd_even_diff` | depth of odd- vs even-numbered transits | light curve |
| `secondary_depth` | deepest qualifying dip elsewhere in the orbit | light curve |
| `duration_over_period` | transit duration ÷ orbital period | **catalogue** |

Three design notes worth stating because they are not obvious:

- **`secondary_depth` uses "deepest run of ≥ 2 bins", not "longest run".** Three
  selection rules were recorded side by side and compared on real data. Longest-run
  fires on 52.9% of confirmed planets against 67.5% of false positives — a ratio of
  1.28, which is close to useless. Deepest-with-minimum-length fires on 11.1% against
  42.4%, a ratio of **3.81**, on dips 2.2× deeper.
- **Sibling KOIs are masked out.** On a multi-planet star, folding on one planet's
  period scatters the *other* planets' transits across arbitrary phase, where they
  form runs deep enough to be misreported as secondary eclipses. Before masking, 0 of
  6 detected "secondaries" on multi-KOI stars sat near phase ±0.5 where a real one
  would be.
- **NaNs are never imputed.** `HistGradientBoostingClassifier` learns a default
  direction for missing values at each split. The missingness is informative — `width`
  goes missing precisely on weak transits — so filling it would erase signal.

---

## 5. Model and evaluation

```
model       HistGradientBoostingClassifier, scikit-learn defaults
threshold   0.25, chosen on a validation split carved from TRAINING stars only
split       GroupShuffleSplit on kepid, 20% test, never touched until the end
test set    434 rows / 133 CONFIRMED / 301 FALSE POSITIVE
```

**Grouping on `kepid` is load-bearing.** 99 of the 1,000 stars in each batch carry
more than one KOI row. A random row split would put the same star on both sides and
leak its noise characteristics from train into test.

The threshold was lowered from 0.5 to 0.25 deliberately. For a review-queue filter,
missing a real planet is worse than passing a false positive to a human, so the
operating point trades precision for recall.

---

## 6. Results

### Classification report (test set, n = 434)

| class | precision | recall | support |
|---|---|---|---|
| FALSE POSITIVE | 0.94 | 0.80 | 301 |
| CONFIRMED | 0.66 | 0.88 | 133 |

```
confusion matrix        predicted FP   predicted CONF
actual FALSE POSITIVE        240             61
actual CONFIRMED              16            117
```

**117 of 133 real planets found. 16 missed. 61 false alarms.**

### As a review filter — the framing that matches the project's question

```
queue before filtering                434 KOIs
auto-rejected as false positive       256  (59.0% of the queue removed)
real planets lost in that rejection    16 of 133  (12.0%)
left for human review                 178, of which 117 are planets
```

### Permutation importances (test set)

```
duration_over_period   +0.1248 ± 0.0138
secondary_depth        +0.0620 ± 0.0099
snr                    +0.0446 ± 0.0110
depth                  +0.0225 ± 0.0099
width                  +0.0185 ± 0.0060
odd_even_diff          +0.0158 ± 0.0118
asymmetry              +0.0066 ± 0.0090
```

### Ranking quality

ROC AUC **0.914** on the reported split; **0.933 ± 0.008** across 25 further
star-grouped splits. The reported split is about two standard deviations less
favourable than typical, so the headline numbers are, if anything, pessimistic.

### Things that did not help

- **Doubling the data, 1,000 → 2,000 stars.** Mean accuracy 0.858 → 0.862 across 25
  splits. Within noise.
- **Hyperparameter search**, 32 configurations. Cross-validated score 0.8806 against
  0.8754 for defaults. Not adopted — the gain does not survive the uncertainty.

---

## 7. Comparison to published work

### Planet Hunters TESS is not a fair comparison, and forcing one would mislead

PHT II reports recovering **85% of TOIs larger than 4 Earth radii and 51% of those
between 3 and 4**. Those are the most quoted numbers in the project, and they should
not be set beside the results above, for four reasons:

1. **Different mission.** Kepler stared at one field for four years; TESS covers a
   sector for ~27 days. Every feature here depends on stacking many transits.
2. **Different task.** PHT's number is *detection* — finding transit-like events in
   light curves nobody has flagged. This project starts from a KOI that already has an
   ephemeris and asks a different question: is it a planet or a false positive?
3. **No published precision.** PHT's pipeline produces a *ranked list*, of which the
   top 500 per sector go to human vetting. Precision is not defined the same way, so
   there is no second number to compare against.
4. **Different labels.** `koi_disposition` is the product of years of systematic
   vetting including centroid and pixel-level diagnostics. TOI status is not that.

### A fairer framing: state the operational claim, and compare like with like

**Against NotPlaNET** (Poleo et al. 2024), a CNN built to cut PHT's vetting burden —
the closest analogue in *purpose*. It reduces light curves needing manual vetting by
up to a third, with essentially no planet candidates lost (zero misclassified in 16 of
18 sectors). This project removes **59% of the queue but loses 12% of the planets**.
Comparable in kind, clearly worse in the trade. Note NotPlaNET's positive class is
"transit or eclipse" versus instrumental junk — eclipsing binaries count as
*positives* there and as *negatives* here, so the metrics still are not interchangeable.

**Against AstroNet** (Shallue & Vanderburg 2018) — same mission, same catalogue, same
planet-vs-false-positive question, and therefore the only genuinely comparable
benchmark. AstroNet reports **AUC 0.988** and 96.0% accuracy, against **0.914–0.933**
here. That is a real and substantial gap, and it is the honest headline comparison.

AstroNet trained on 15,737 examples against 2,224 here, and learns transit shape
directly from binned arrays rather than from seven hand-engineered numbers. The gap is
what one would expect from those two facts.

---

## 8. Limitations

### `asymmetry` does not work, and that is a result

Permutation importance **+0.0066 ± 0.0090** — smaller than its own uncertainty, and
consistent with zero. This is not a tuning failure. Relative asymmetry
(`|asymmetry| / depth`) falls near-monotonically as SNR rises: the noisiest stars show
the largest apparent asymmetry and the cleanest show the least. If it were measuring
real transit geometry it would not track SNR that way.

The finding was flat at 1,118 rows and still flat at 2,224. **Transit asymmetry is not
measurable at Kepler noise levels with this method.** Recorded rather than buried,
because a feature that demonstrably carries no signal is worth knowing about.

### The −8.8% depth bias

Measured depth runs a median **8.8% shallower** than the catalogue's, stable across
2,000 stars. Diagnosed, deliberately not corrected:

- It is **definitional**. `koi_depth` is the flux lost at the *minimum* of a fitted
  transit model; this measures a *median* across the central half of the transit. Limb
  darkening curves the transit floor, so a median over that window necessarily sits
  above the minimum.
- Narrowing the window from `duration/4` to `/6` cuts the bias to −5.7% and then
  **plateaus** — about 45% of the offset is window width and the rest is the
  definitional gap, which no window choice closes.
- It is **uniform**, not systematic in a way that matters. A flat multiplicative
  offset is invisible to a tree, which learns thresholds from this data, not from the
  catalogue. Correcting it would trade a harmless bias for real noise on the sparsest
  rows.

An impact-parameter test was run to check whether grazing (more V-shaped) transits
were more affected. They are not — Spearman ρ = −0.095, no relationship. The
shape-based explanation was tested and did not hold.

### Coverage: 2,289 of 7,587 available rows

30% of the labelled catalogue. The sample is a random draw by star at natural class
proportions, so it should be unbiased, but every number here carries the uncertainty
of a 30% sample. The strongest evidence that this is not badly wrong is that doubling
from 1,000 to 2,000 stars moved almost nothing.

### Other caveats

- **Secondary-eclipse masking assumes circular orbits.** The masked region is phase
  ±0.5; eccentric orbits shift a real secondary off centre. Partial fix.
- **The top feature is not a light-curve feature.** `duration_over_period` comes
  entirely from catalogue columns. Alone, it scores 0.70 accuracy against a 0.69
  majority-class baseline — barely better than guessing — so it is not carrying the
  model by itself. But it *is* ranked first, and that tension deserves stating.
- **`asymmetry` and `width` are retained** despite weak importance. They cost nothing
  and may earn their place on higher-SNR data.

---

## 9. Three bugs that produced plausible numbers instead of crashing

The most transferable lesson from this project. All three would have *improved*
apparent performance, and none would have raised an error.

### `koi_fpflag_*` as features

The catalogue ships four false-positive flags. They are not correlated with the label
— they **are** the label, recorded by the same vetting process. A model trained on
them reads the answer key and scores near-perfectly.

### CANDIDATE rows labelled negative

`y = (disposition == 'CONFIRMED')` turns all 1,977 CANDIDATE rows into `y = 0`. A
candidate is not a false positive; its label is unknown. The bug trains the model to
call genuinely uncertain objects false positives, and the resulting accuracy looks
fine because the model has learned a self-consistent, wrong rule.

### `MaskedNDArray` coercing to 0.0

`binned.flux.value` is an astropy `MaskedNDArray`, not a plain array. `np.nanmedian`
over a fully-masked slice returns a *masked scalar*, and `float()` of that is **0.0**,
not `NaN`. So when a long-period KOI had every in-transit bin empty:

```
depth = baseline − 0.0 = baseline ≈ 1.0
```

A reported 100% dimming — physically impossible — on 3 of 104 rows. The guard tested
whether any *bins* fell in the transit window, not whether those bins held any *data*.

**All three affected rows were FALSE POSITIVE.** The bogus value was correlated with
the label, so a tree would have learned "depth ≈ 1.0 ⇒ false positive" and been
rewarded for it.

### The pattern

None of these crashed. Each produced a plausible number that happened to correlate
with the answer. The defences that actually caught them were: refusing to let
catalogue-derived columns into the feature matrix without justifying each one;
asserting physical impossibility (a depth of 1.0 is not a thing); and checking class
composition whenever a subgroup looked unusual.

A crash is a gift. A plausible wrong number is the expensive kind of bug.

---

## 10. Reproducing this

```bash
python transitcheck_features.py            # dry run: sample 100 stars, no downloads
python transitcheck_features.py --run      # extract features for 100 stars
python run_1000.py --run                   # first 1,000-star batch
bash drive_batch2.sh                       # second batch, chunked with resource guards

python analyse_run.py features_2000.csv    # full analysis report
python verify_model.py                     # reproduce the model and its metrics
python verify_discrepancies.py             # multi-split significance checks
```

Extraction is resumable: results are checkpointed per row, and re-running skips
completed work. Expect ~6–9 hours and ~7 GB of cache per thousand stars.
