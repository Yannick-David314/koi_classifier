# TransitCheck — project context for Claude Code

Read this fully before touching anything. It carries the reasoning behind every
tuning decision in the pipeline, because most of them look arbitrary otherwise
and several were arrived at empirically rather than from first principles.

---

## 1. How I want to work

**I am learning, not just shipping.** I have little programming background and
I'm building this to understand it. That shapes how I want you to help:

- **Explain before implementing.** Tell me what a change does and why, and give
  me the chance to attempt it, before writing finished code for me.
- **When I hand you a wrong attempt, tell me what's wrong and why** rather than
  silently replacing it with a correct version.
- **Don't skip the reasoning to get to a working answer faster.** If a decision
  has a tradeoff, say what the tradeoff is even if you've already picked a side.
- **Push back on me.** If something I've decided is wrong or fragile, say so.

That said, for this handoff I *have* deliberately delegated implementation of the
remaining preprocessing (see §7). Build it — but explain what you built.

---

## 2. What the project is

Classify Kepler Objects of Interest (KOIs) as `CONFIRMED` planets or
`FALSE POSITIVE` from their light curves, using a gradient-boosted tree on
engineered features.

The honest framing: this is not "discover a new planet." It's "can an ML
pre-filter reduce the human review burden for citizen-science transit vetting?"
Planet Hunters TESS is the real benchmark to compare against, not a competitor.

**Why trees and not a CNN.** A CNN (the Shallue & Vanderburg "AstroNet"
approach) learns transit *shape* directly from a binned array, because its kernel
reads neighbouring bins together. A tree treats every column as independent — it
has no idea bin 100 and bin 101 are adjacent — so raw bin arrays are a poor fit
for it. Hence engineered features: I do the shape-recognition myself and hand the
tree a handful of meaningful numbers. AstroNet also trained on 15,737 examples;
at ~1,000 stars a from-scratch CNN would likely overfit. CNN is a later
stretch goal, not abandoned — the local/global view machinery is already there
for it.

---

## 3. The data

- **Catalogue:** `MyProject_sync.csv` in the project root. This is the full KOI
  cumulative table (~9,500 rows) from the NASA Exoplanet Archive.
- **Light curves:** downloaded live from MAST via `lightkurve`.
- **Environment:** Windows, venv at `.venv/`, Jupyter notebooks.

### Columns that matter

| Column | Meaning | Units |
|---|---|---|
| `kepid` | Kepler Input Catalog ID (the star) | int |
| `koi_disposition` | the label: CONFIRMED / FALSE POSITIVE / CANDIDATE | str |
| `koi_period` | orbital period | days |
| `koi_time0bk` | transit epoch (first transit centre) | BKJD |
| `koi_duration` | transit duration, roughly first-to-last contact | **hours** |
| `koi_depth` | catalogue transit depth | ppm |
| `koi_fpflag_*` | false-positive vetting flags | binary |

**Units are the single biggest source of bugs here.** `koi_duration` is in hours,
`koi_period` is in days, and Lightkurve's `.phase` is in days by default. Nearly
every `/24` or `* 24` in the code is an hours↔days conversion. Do not remove one
without tracing what it's converting.

### Leakage — read this before building X

These must **never** enter the feature matrix:

- **`koi_depth`** — the catalogue's own depth measurement, produced by the same
  vetting process that produced the labels. It's kept in the output table *only*
  as a ground-truth check on my measured depth, and must be dropped before
  training.
- **`koi_fpflag_*`** — these flags literally *are* the false-positive
  determination. Including them would be training a model to read the answer key.
- **`kepid`** — an arbitrary ID. A tree will happily split on it and learn
  nothing real.

---

## 4. The pipeline

Per star: **download → clean → flatten (transit-masked) → fold → bin → measure**.

The working implementation is in `transitcheck_features.py`. Every constant is
at the top of that file. What follows is why each one is what it is.

### 4.1 Download and clean

```python
lk.search_lightcurve(f"KIC {kepid}", author='Kepler').download_all().stitch()
```

- `download_all().stitch()` not `[0].download()` — a single Kepler quarter
  (~90 days) gives too few transits to average noise down. Shallow transits are
  invisible without stacking all quarters.
- `author='Kepler'` restricts to the official pipeline products. Other authors
  exist (community pipelines) and mixing them is inconsistent.
- Lightkurve defaults to **PDCSAP** flux (systematics-corrected), which is what
  we want. Verify with `star_lc.meta.get('FLUX_ORIGIN')` rather than trusting it.

**Outlier removal is asymmetric on purpose:**

```python
remove_outliers(sigma_lower=20, sigma_upper=5)
```

A real deep eclipse *is* statistically an outlier. A symmetric `sigma=5` cut
would shave the bottom off genuine transits. So: strict upward (cosmic rays,
instrument glitches), very lenient downward (protect real dips).

### 4.2 Transit masking

**The mask is computed from arithmetic on timestamps, never from the flux.**
The catalogue gives period and epoch, so transit times are predictable:
`epoch`, `epoch + period`, `epoch + 2×period`, … Searching the flux for dips
would catch every cosmic ray and miss every shallow transit.

```python
temp_fold = star_lc.fold(period=period, epoch_time=epoch_time, normalize_phase=True)
fractional_duration = (duration / 24) / period
phase_mask = np.abs(temp_fold.phase.value) < (fractional_duration * 1.5)
transit_mask = np.isin(star_lc.time.value, temp_fold.time_original.value[phase_mask])
```

**Three gotchas here, all of which have already bitten once:**

1. **`normalize_phase=True` is only on this fold.** It makes `.phase`
   dimensionless (−0.5 to 0.5, a fraction of the period) so it can be compared
   against `fractional_duration`. Every *other* fold in the codebase omits it, so
   `.phase` is in days there. These two conventions coexist deliberately. If you
   touch either, check which one you're in.
2. **`time_original`** exists because folding keeps the real timestamps in a
   separate column. The phase test selects rows in folded space; `time_original`
   translates them back to real time.
3. **`np.isin` is a membership test, not positional.** Folding reorders rows by
   phase, so a positional mask would be meaningless. `np.isin` returns something
   aligned to `star_lc`'s own order and length, which is what makes it safe to
   `&` against other masks later.

### 4.3 Flattening

```python
cadence_minutes = np.median(np.diff(star_lc.time.value)) * 24 * 60
window_length = int((3 * 60 * duration) / cadence_minutes)
window_length = max(window_length, 5)
window_length += 1 if window_length % 2 == 0 else 0
star_lc.flatten(window_length=window_length, mask=transit_mask)
```

- `flatten` removes slow drift (star spots, instrument drift) by sliding a
  window, fitting a smooth trend, and dividing it out.
- **The window is specified in data points, not time**, so it must be derived
  from each star's own measured cadence. Kepler long cadence is ~29.4 min
  nominally, but measure it rather than assume — stitched quarters and gaps vary.
- **3 transit durations** is the target width: wide enough that the fit doesn't
  follow the dip, narrow enough to still track real stellar variability.
- **Must be odd** — Savitzky-Golay windows centre on a single point.
- **Floor of 5** — the filter needs a window wider than its polynomial order.
- **`mask=transit_mask` is the important part.** It tells the trend fit to ignore
  in-transit points, so the fit has no reason to dip toward the transit or
  overshoot recovering from it. Without it you get a characteristic artefact:
  a shrunken dip followed immediately by a fake upward bump. That was visible in
  early plots and this is what fixed it.
- `flatten` only modifies **flux**. The time array is untouched, so anything
  computed from timestamps gives identical results before or after.

### 4.4 Fold, slice, bin

`prepare_transit()` folds, cuts a window around the transit, bins it, and returns
`(depth, flux, phase, baseline)`. Every shape feature works from these outputs so
they all describe the same curve rather than each recomputing their own.

- `epoch_time` **must** be passed to `fold()`. Without it, phase 0 is the first
  timestamp, not the transit centre.
- Window = 4 transit durations, centred. Gives the transit plus surrounding
  baseline.
- `n_bins = 201` (Shallue & Vanderburg's local-view convention).

**Bin size reasoning, which took a while to get right:**

The original approach was a fixed *time* bin (1 day), which is coarser than the
entire transit for nearly every KOI and washed the signal out completely. Then
duration-scaled bins. Then fixed *count*. The rules that survived:

- Never bin finer than the star's native cadence — a bin holding 0–1 raw points
  isn't averaging anything.
- Bin size should scale with transit duration, not be a fixed number of days.
- A fixed bin *count* only matters when the array itself is the model input
  (i.e. for a CNN). For engineered features the array collapses to one number, so
  fixed count is not required.

### 4.5 Known inconsistency, left deliberately

`prepare_transit` uses a fixed `n_bins=201`, while `measure_secondary_depth`
computes bin width from physics (`duration / bins_per_duration`). These are
inconsistent, and I know it.

Left as-is because: the depth measurement is already validated against
`koi_depth` at that setting, and the `duration/4` in-transit window is entangled
with the bin width — changing one invalidates the tuning of the other. The gain
would be consistency, not accuracy.

**Revisit only if the scaled batch surfaces a real problem** (very short or very
long durations where 201 bins across 4 durations is badly wrong). Don't refactor
it pre-emptively.

The plotting/local-view code also uses fixed `n_bins` — that one is *correct*,
since consistent resolution across stars is what makes plots comparable by eye.

---

## 5. The six features

### depth — **validated**
Median of in-transit binned flux vs out-of-transit baseline.

In-transit window is `duration/4` — the flat bottom only. This was tuned
empirically against `koi_depth`:

| in-transit window | result |
|---|---|
| `np.nanmin` (single lowest bin) | overshot by 15–71%, worst on shallow transits |
| `duration/2` median | undershot 6–51%, worst on deep transits |
| `duration/4` median | within ~15% on most rows, no systematic direction |

The `nanmin` failure mode is the important lesson: on a shallow transit the
lowest bin is part real dip, part downward noise, so it always overshoots.
**Anywhere you're tempted to use `min` on a noisy binned curve, use a median over
the relevant region instead.** This applies to every feature here.

Exact catalogue agreement is not the goal — the catalogue fits a full transit
model, we take a median of binned flux. Those are different definitions. What
matters is that the measurement is *consistent* across stars.

### width — **validated**
Longest contiguous run of bins below half-depth, converted to hours.

Three ways to measure this, and why the third won:
1. Count all below-threshold bins → strays inflate the count.
2. Span first to last crossing → one stray at the edge stretches it enormously.
3. **Longest contiguous run** → a real dip is unbroken; a noise spike is a run of
   1 and loses. No need to restrict where you look, so an unusually long grazing
   eclipse still gets measured in full.

`find_longest_true()` implements this with a `np.diff` trick. The `False` padding
at both ends is load-bearing: it guarantees every run has an opening `+1` and
closing `-1`, and the leading pad's index shift exactly cancels `np.diff`'s
off-by-one, so returned indices land on the original mask. Returns
inclusive-start, exclusive-end.

**Takes `depth` as its threshold input**, so a biased depth propagates into a
biased width.

### asymmetry — **implemented, appears noise-dominated**
Splits in-transit bins at phase 0, compares median depth on each side.

- Window is `duration/2`, not `/4` — asymmetry lives in the ingress/egress
  slopes, which the narrower depth window deliberately excludes.
- **Signed, not absolute.** A tree can ignore the sign by splitting at zero, but
  can't recover it if thrown away.
- See §6 — validation suggests this is measuring noise, not geometry, at current
  signal levels.

### snr — **implemented**
`depth` divided by MAD-based noise from out-of-transit bins (window `duration/2`,
so the noise sample is clear of the slopes). Returns NaN if noise is 0 or NaN.

**MAD not standard deviation.** MAD = median of absolute deviations from the
median, × 1.4826. The constant rescales it onto a std-dev scale for clean data,
so the two agree when nothing is wrong — but one bad value, squared, dominates a
std-dev while barely moving a MAD.

**This SNR is not comparable to Kepler's own detection significance**, which is a
multi-event statistic across all transits. Low values here don't mean marginal
detections.

### odd_even_diff — **implemented**
Depth of odd-numbered transits vs even-numbered ones, absolute difference. A real
planet's transits are identical; a blended eclipsing binary often alternates.

**The split must happen before folding.** Folding stacks every transit onto one
shared phase axis, which erases which transit a point came from. Binning erases
it further.

```python
transit_number = np.round((time - epoch_time) / period).astype(int)
```

That one division yields two things: the **whole number** is which transit, the
**fractional leftover** is the phase. Rounding works because the wobble (hours of
transit width) is tiny against the gap between transits (days).

Then both conditions must hold — in-transit **and** the right parity:
```python
even_filter = transit_mask & (transit_number % 2 == 0)
```
Build the complete mask first, apply it once. Applying `transit_mask`, then
trying to filter the result again, is a type/length mismatch.

Uses **25 bins**, coarse on purpose: each group holds half the data, and only one
number is needed, not a resolved shape.

### secondary_depth — **implemented, not separating; see §6**
Deepest contiguous dip anywhere outside the primary transit. An eclipsing binary
eclipses twice per orbit; a planet effectively once.

- **Needs the global view** (whole folded period) — a secondary sits roughly half
  a period away, far outside the local window.
- Searches everywhere outside the primary rather than checking phase 0.5, because
  eccentric orbits shift the secondary off-centre.
- Blanks ±2 durations around phase 0. Generous on purpose: the primary's sloped
  edges are deep enough to win any "deepest dip" search and be misreported.
- Threshold is 3 × noise. For normal noise, a random point falls 3 spreads below
  the mean ~0.1% of the time; the contiguity requirement adds a second layer, so
  a false hit needs several *consecutive* noise bins to cross.
- Returns **0.0** when nothing is found — "no secondary" is a real answer, not a
  failure.
- Returned **raw**, not divided by noise. Deliberate: the ratio can be computed
  afterward, the raw depth can't be recovered from a ratio.

---

## 6. Validation results — 10-star batch

5 CONFIRMED (rows 0–4), 5 FALSE POSITIVE (rows 5–9). All ran without errors.

| # | kepid | disp | koi_depth | measured (ppm) | width/dur | snr | secondary |
|---|---|---|---|---|---|---|---|
| 0 | 10797460 | CONF | 615.8 | 583 | 0.82 | 6.25 | 0.0 |
| 1 | 10797460 | CONF | 874.8 | 747 | 0.62 | 3.00 | 1.95e-4 |
| 2 | 10854555 | CONF | 603.3 | 517 | 0.30 | 2.65 | 0.0 |
| 3 | 10872983 | CONF | 1517.5 | 1478 | 0.92 | 7.80 | 0.0 |
| 4 | 10872983 | CONF | 686.0 | 635 | 0.86 | 4.42 | 1.08e-4 |
| 5 | 10848459 | FP | 8079.2 | 7046 | 0.60 | 97.5 | 0.0 |
| 6 | 6721123 | FP | 233.7 | 180 | 0.48 | 3.15 | 4.47e-5 |
| 7 | 10419211 | FP | 17984.3 | 15964 | 0.60 | 122.6 | 3.99e-4 |
| 8 | 10464078 | FP | 8918.7 | 9704 | 0.72 | 62.6 | 4.94e-4 |
| 9 | 10480982 | FP | 74284.0 | 66326 | 0.58 | 604.5 | 0.0 |

**depth separates well, with one important exception.** CONFIRMED sit at
~500–1500 ppm; four of five FPs at 7,000–66,000 ppm (eclipsing-binary territory —
a Jupiter around a Sun-like star is only ~1%). But **row 6 is a FALSE POSITIVE at
180 ppm**, shallower than every confirmed planet. Not all FPs are binaries — some
are centroid offsets or background blends, which can be arbitrarily shallow. A
"deep ⇒ false positive" rule would misclassify it. This is exactly why there are
six features.

**width behaves soundly.** All ratios below 1 (half-depth width < first-to-last
contact, as expected). High-SNR rows cluster tightly at 0.58–0.72; low-SNR rows
scatter 0.30–0.62. Consistent where signal is strong, noisy where it isn't — the
right failure mode.

**snr separates but is not independent.** It's `depth / noise`, so it's highly
correlated with depth by construction. Trees handle correlated features fine, but
don't read it as confirmation.

**asymmetry appears noise-dominated.** `|asymmetry| / depth` falls near
monotonically as SNR rises — the noisiest star shows the largest apparent
asymmetry, the cleanest the smallest. If it were measuring real geometry it
wouldn't track SNR like that. Not a bug; transit asymmetry just isn't measurable
at these signal levels. Keep the feature (it may earn its place on stronger
signals) but expect little from it now.

**secondary_depth is not separating, and there's a suspicious pattern.**
Nonzero on 2 CONFIRMED and 3 FP; zero on row 9, the 7.4% eclipsing binary where a
secondary is most expected.

Rows 0/1 are the same star and rows 3/4 are the same star — both **multi-planet
systems**. In each pair, the first entry reads 0.0 and the second reads nonzero.

**Hypothesis (2 cases, not a conclusion):** folding on one planet's period lets
the *other* planet's transits form a run deep enough to cross the 3× threshold.

**Direct check I'd like run:** plot the global view for row 1 and see where the
detected dip sits in phase. Near ±period/2 ⇒ genuine secondary. Somewhere
arbitrary ⇒ contamination. Row 9's global view is worth eyeballing too (though
not every binary has a detectable secondary — it depends on the temperature ratio
of the two stars).

---

## 7. What I'm handing over

Everything from here up to (not including) model training.

### 7.1 Error handling

Not yet implemented. My plan, for you to build and improve on:

- **Wrap the whole per-star body** in try/except, not just the download — a bad
  epoch fails later in the pipeline, not at download time.
- **Catch broadly** (`except Exception`) as a deliberate temporary measure, and
  record `type(e).__name__` plus the message in an `error` column. At ~1,000
  stars I don't yet know what breaks; this finds out empirically so the catch can
  be narrowed later. Comment it as temporary so it doesn't read as sloppiness.
- **Keep failed rows with NaN features** rather than skipping, so failures can be
  counted and inspected. Needs a `FEATURE_NAMES` constant so failed rows have the
  same columns as successful ones.
- **Print failures as they happen** — a batch this size shouldn't stay silent.

**Known failure modes to expect:**
- Search returns nothing for that KIC.
- MAST timeout or dropped connection.
- **Missing `koi_time0bk` (NaN)** — this one fails *silently*, producing a
  garbage fold rather than an error. It needs its own `pd.isna` guard *before*
  the try, not inside it.
- `flatten` fails if `window_length` exceeds the number of data points.
- Odd/even subsets can be empty if a star has only one or two transits.
- Very long periods with few observed transits generally.

### 7.2 Scaling to ~1,000+ stars

From the full ~9,500-row cumulative table.

**Class balance: your call.** I don't have a strong view. Tradeoffs as I
understand them: natural proportions match deployment reality and keep
precision/recall meaningful; balanced sampling can help the model learn the
minority class. Pick one, explain why, and make it a parameter I can flip.

**Things that matter at this scale but not at 10:**

- **Cache downloads to disk.** Re-downloading on every run is brutal at 1,000
  stars, and I will be rerunning as features change.
- **Cache per `kepid`, not per row.** Multi-planet systems mean the *same* star
  appears in several KOI rows (rows 0/1 and 3/4 above). Downloading its light
  curve once per row is pure waste.
- **Checkpoint incrementally.** Write results as they come in, so a crash at row
  800 doesn't lose everything.
- **Be polite to MAST.** Don't hammer it in parallel without throttling.
- Expect the runtime to be dominated by downloads, not computation.

### 7.3 Assemble X and y

```python
X = feature_table[FEATURE_NAMES]
y = (feature_table['disposition'] == 'CONFIRMED').astype(int)
```

- Drop `kepid`, `koi_depth_ppm`, `koi_duration_hours`, `disposition`, `error`
  from X. See §3 on leakage.
- **Keep `kepid` accessible separately** — needed for the group split below.

**Model choice: `HistGradientBoostingClassifier` (scikit-learn). Decided.**

Several features return `np.nan` by design — `width` when nothing crosses the
half-depth threshold, `snr` when the noise estimate is 0 or NaN.

**Do not impute or fill these.** `HistGradientBoostingClassifier` handles NaN
natively: at each split it learns a default direction for missing values and
picks whichever reduces error, so missingness itself becomes usable information.
That matters here because **these NaNs are not random** — `width` goes missing
precisely on weak or noisy transits, and `snr` when the noise estimate fails.
Both carry real information about the star. Filling them with a median would
erase exactly the signal the model could use.

Chosen over XGBoost/LightGBM/CatBoost only to avoid an extra dependency; at
~1,000 rows and 6 features any of them would be fine and differences would be
within noise. The older `GradientBoostingClassifier` is ruled out — it errors on
NaN outright.

Training itself is still out of scope (§8) — this decision is recorded here so
the preprocessing doesn't add an imputation step it doesn't need.

### 7.4 Train/test split — set it up, don't train

**Split by star, grouped on `kepid`, not by row.** Rows 0/1 and 3/4 share a
`kepid`; a random row split would put the same star on both sides and leak its
noise characteristics from train into test. `GroupShuffleSplit` or `GroupKFold`.

Set the split up and verify no `kepid` appears on both sides. **Stop there** —
actual training is later.

### 7.5 Validation items to revisit once scaled

- **Bin count is provisional.** The 25/201 numbers were chosen against a 10-star
  batch whose sparsest case was ~4.5 points/bin. A 1,000-star batch will contain
  something sparser. Worth reporting the points-per-bin distribution so the floor
  can be reset with real margin.
- **Relative asymmetry.** If raw `asymmetry` doesn't separate classes, try
  `asymmetry / depth`. Given §6, I suspect neither will help much yet.
- **Noise-normalised secondary depth.** If raw doesn't separate, try dividing by
  noise. Note `snr` already carries noise info, so a tree may combine them itself.
- **The multi-planet contamination check** in §6 — this is the one I most want
  answered, since it may mean `secondary_depth` is measuring the wrong thing.

---

## 8. Out of scope for now

Model training, hyperparameter tuning, evaluation metrics, backend, frontend.
Phase 3+ in `TransitCheck-Roadmap.md`. Don't start these.

Also parked: the CNN path. The local/global view code stays as-is so it's
available later, but nothing should be built *for* it now. In particular, do
**not** add the −1/+1 depth normalisation from the AstroNet global view — that
convention deliberately erases depth, which is currently the single most
discriminating feature.

---

## 9. If something here is wrong

This document was written from a long working session and reflects what I
believed at the time. If you find something that contradicts the code, or a
decision that doesn't hold up, say so rather than working around it. Several of
the numbers above were arrived at by trial against `koi_depth` and are not
sacred — but tell me before changing them, and tell me what moved as a result.

---

## 10. Corrections — findings from the section 7 handover

*Appended, not merged. Sections 1–9 are left exactly as written so the record
shows what was believed at the time versus what turned out to be true. Each entry
says what the doc claimed, what the code or data actually does, and what changed.*

All row counts below refer to the 7,587 CONFIRMED/FALSE POSITIVE rows in
`MyProject_sync.csv` (9,564 rows total: 2,748 CONFIRMED, 4,839 FALSE POSITIVE,
1,977 CANDIDATE), covering 6,641 distinct stars.

### 10.1 Short-cadence data was being stitched in with long-cadence — **fixed**

§4.1 guards against mixing *authors* but not *cadences*. `search_lightcurve(author='Kepler')`
returns 60 s short-cadence products alongside 1800 s long-cadence ones for some
targets, and `download_all()` takes all of them. For KIC 6721123 — **row 6 of the
§6 validation table** — the search returns 20 products: three short-cadence Q3
chunks first, then all 17 long-cadence quarters.

`LightCurveCollection.stitch()` is a plain `vstack` in collection order; it does
**not** sort by time. So that star's stitched light curve:

- runs backwards at the seam (Q3 → Q1),
- contains Q3 twice, at two cadences,
- has a median time difference of ~1 minute, because one SC quarter contributes
  ~130,000 points against ~73,000 for all 17 LC quarters combined.

That last one propagates into §4.3: `window_length = 3 × 60 × 5.022 / 1 = 904`
points. In the 1-min region that is the intended ~3 durations. In the 30-min
region it is ~452 hours ≈ 19 days ≈ 90 durations — so most of the curve was
effectively not detrended at all.

What survived: `np.isin` in §4.2 (membership, order-free — that note was
load-bearing) and `fold()` (pointwise). What did not: `flatten` and everything
downstream of it.

**Row 6 is the 233.7 ppm FALSE POSITIVE the §6 argument rests on**, and it has the
largest relative depth shortfall in the shallow group (−23%). The astrophysics in
§6 may well still hold — shallow FPs from centroid offsets and background blends
are real — but the evidence for it currently comes from a row processed through a
broken detrend. Re-measure it before relying on it.

**Fixed:** cadence pinned to `KEPLER_LONG_CADENCE_SECONDS = 1800`, one search
filtered locally, plus an explicit time sort and duplicate-timestamp drop after
`stitch()` regardless. The trade-off is losing genuine 1-min resolution on the
subset of stars that have it; consistency across stars wins, which is §4.1's own
argument. `cadences_seen` records what each target actually had, so the scale of
the exposure becomes measurable rather than assumed.

### 10.2 `secondary_depth` finds the *longest* dip, not the deepest — **recorded, not changed**

§5, §6 and the function's own docstring all said "deepest contiguous dip".
`find_longest_true` takes `argmax` of run *lengths*, so the code picks the widest
run crossing 3σ and reports that run's median depth.

This matters for the §6 hypothesis. A genuine secondary is short (~4 bins at
`bins_per_duration=4`); another planet's transit, or a broad variability wiggle,
can be much wider. The rule is biased toward exactly the contaminants §6
suspects, and against the signal it wants.

**Not changed.** Three variants are now recorded side by side, each with the phase
of the dip it actually found: `secondary_depth` (longest run, unchanged),
`secondary_depth_deepest` (deepest of any length, the fragile control — a
single-bin run is its own median), and `secondary_depth_deepest_min2` (deepest run
of ≥ 2 bins). The scaled run settles it with data.

### 10.3 `SHAPE_WINDOW_FRACTION`'s comment was wrong — **fixed**

The comment claimed `width` used it. `measure_asymmetry` and `measure_snr` do;
`measure_width` never references it and scans the whole local view. The §5 prose
("No need to restrict where you look") was right; the comment was wrong. Comment
corrected.

### 10.4 The `koi_time0bk` NaN guard in §7.1 guards something that never happens

Across all 9,564 rows: `koi_period`, `koi_time0bk` and `koi_duration` have **zero**
NaNs. The only column that actually goes missing is `koi_depth` (363 rows overall,
259 in the CONFIRMED/FALSE POSITIVE subset) — which is the §6 validation column,
not a feature, so it degrades a comparison rather than breaking a run.

The guard is implemented anyway: it is free, and the failure it prevents would be
silent. But the empirical claim behind it does not hold for this catalogue.

### 10.5 §4.5's trigger has already fired — **feature guarded, row kept**

§4.5 defers the fixed-201-bins vs physics-derived-bin-width inconsistency until
"very short or very long durations" appear. They already have:

| condition | rows | share |
|---|---|---|
| `4 × duration ≥ period` | **217** | 2.9% |
| `duration / period > 0.1` | 1,102 | 14.5% |
| `duration > 24 h` | 154 | 2.0% |
| `duration < 1 h` | 150 | 2.0% |

The first row failed **silently**. When `4 × duration ≥ period`,
`measure_secondary_depth`'s `outside_primary` mask selects nothing, `np.nanmedian`
of an empty slice returns NaN, `robust_noise` returns NaN, and the function
returned `0.0` — which is the encoding for *"no secondary found"*. So 217 rows
reported a confident astrophysical answer to a question that was never asked.

**Fixed as a per-feature guard, not a row rejection.** Only `secondary_depth` is
broken there; `depth`, `width`, `asymmetry` and `snr` are degraded, not
meaningless, so the row stays and that one feature returns NaN (which
`HistGradientBoostingClassifier` consumes natively). The same reasoning now
applies when the noise estimate fails for any other reason.

This is a continuum, not a threshold: by `duration/period > 0.1` the 4-duration
window already covers >40% of the orbit, so `baseline` picks up transit wings and
`depth` drifts low well before the hard failure. `duration_over_period` is
recorded on every row; plot depth error against it, using `koi_depth` as
reference, once the run finishes.

### 10.6 "Never bin finer than the star's native cadence" is not enforced, and cannot be as written

Local bin width is `duration × 4 / 201` hours. Reaching Kepler long cadence
(29.4 min) would need `duration ≥ 24.6 h`, so **98.1% of rows bin finer than
native cadence**. The rule as stated in §4.4 is violated essentially always.

It works anyway because folding stacks all transits: the quantity that matters is
raw points *per bin after folding*, not per bin per transit. §7.5's instinct was
right. Rough pre-run estimate from `koi_num_transits × bin_width / 29.4 min`:

```
percentiles (1/5/25/50/75/95):  0.0 / 1.2 / 8.7 / 23.5 / 67.4 / 221.6 points per bin
rows below 4.5 pts/bin (the 10-star floor):   996
rows below 1.0 pts/bin:                        318
```

Treat these as an estimate — they assume long cadence throughout (now accurate,
given 10.1) and uniform coverage, and 681 rows have no `koi_num_transits` at all.
`local_points_per_bin_median`, `n_local_bins` and `n_local_bins_filled` are now
measured per row; compare the measurements against this estimate before anyone
touches `LOCAL_N_BINS`.

Two sub-notes acted on:
- **`koi_num_transits == 0` on 208 rows.** Nothing was observed, so there is no
  transit to measure. Now a precheck rejection, before any download. NaN passes
  through — unknown is not zero.
- **`LOCAL_N_BINS = 201` is nominal, not guaranteed.** `bin(time_bin_size=...)`
  emits however many bins fit the data actually present, so the array is not
  reliably length-201. Harmless for features (each collapses to one number), but a
  real obstacle for the parked CNN path in §8, which needs fixed-length input.
  Noted in the constant.

### 10.7 §7.3's `y` would have silently mislabelled CANDIDATEs — **fixed**

`y = (feature_table['disposition'] == 'CONFIRMED').astype(int)` turns all 1,977
CANDIDATE rows into `y = 0` — trained as false positives. §3 lists three
disposition values; §7.2 says "from the full ~9,500-row cumulative table" and
never says to exclude one. The filter is now explicit and upstream, in
`sample_kois`, which is the only place it appears.

### 10.8 §4.1's `FLUX_ORIGIN` check was advice, never code — **fixed**

Now verified on every star and recorded as `flux_origin`, with a warning printed
on mismatch. This matters more given 10.1: `stitch()` uses
`metadata_conflicts='silent'`, so post-stitch metadata comes from whichever
product happened to be first in the collection.

### 10.9 A secondary at phase ±0.5 arrived as two runs, not one — **fixed**

Verified on a synthetic star with a known 400 ppm secondary. Phase −0.5 and +0.5
are the same physical point but sit at opposite ends of the phase array, so a
secondary centred there is split into two runs of half the length:

```
bins   0-2   len=2  phase frac -0.4969..-0.4906  depth=0.000399
bins 158-160 len=2  phase frac +0.4906..+0.4969  depth=0.000407
```

Each fragment's depth is right; the run length is halved.

This was not a caveat but a bug in the search, and it biased the 10.2 comparison
toward the answer we already suspect is wrong: both fragments recovered the depth
correctly (399 and 407 ppm against a true 400), so the *deepest* variants were
robust to the split and *longest-run* was not. Worse, it made the min2 variant
untestable — a 4-bin secondary splitting into two 2-bin runs still passes a ≥ 2
filter, but a 2-bin secondary splitting into two 1-bin runs is discarded entirely
as noise.

**Fixed.** `measure_secondary_depth` now rolls `flux` and `phase` by `len(flux)//2`
before building `below_threshold`, putting the wrap point at the array centre. The
primary is split across the array edges instead, which is free because it is
blanked either way. The 4-bin secondary above now arrives as a single 4-bin run,
and the 2-bin case survives the min2 filter.

One consequence the roll creates: a run straddling the array centre has phase
values like `[+2.45, +2.48, -2.48, -2.45]`, which a plain mean averages to ~0
instead of ±0.5. `circular_mean_phase_fraction` averages the angles instead —
correct at the wrap, and identical to a plain mean everywhere else.

### 10.10 `flatten` partially removed the secondary eclipse — **partially fixed**

`transit_mask` protects only the **primary**. The secondary is unmasked, so the
Savitzky-Golay trend follows it and divides part of it out. On the same synthetic
star, with a 19-point (9.3 h ≈ 3.1 duration) window:

```
raw        secondary recovered at 102%
flattened  secondary recovered at  66%
```

So `secondary_depth` was systematically low by an amount set by the window-to-
secondary width ratio. This is the same artefact §4.3 describes for the primary.

**Partially fixed.** `build_transit_mask(include_secondary=True)` now also masks
phase ±0.5 ± 1.5 durations, and `flatten_star` uses that mask. On the synthetic
star the secondary now recovers at **99.6%** of truth, up from 66%.

Two things to keep in mind:

- **It is only partial.** The secondary is assumed to sit at phase ±0.5, which is
  true for a circular orbit and wrong for an eccentric one. `secondary_depth` is
  recorded as measured, and `secondary_phase_fraction` reports where the dip
  actually was — that is what tells us how often the assumption holds.
- **Two masks now exist and they are not interchangeable.** The primary-only mask
  feeds `measure_odd_even_diff`, which must not see secondary-eclipse points; the
  primary-plus-secondary mask feeds `flatten`. `detrend_masked_fraction` records
  how much is withheld from the trend fit (15.0% on the synthetic star, where
  duration/period = 0.025).

This does touch a validated path, so it is re-validated by the run itself: the
output table compares measured depth against `koi_depth` on every row, so if
withholding more points from the trend fit degrades primary depth agreement, it
shows up in the same table.

### 10.11 Smaller items

- §4's filename (`transitcheck_features.py`) and §8's `TransitCheck-Roadmap.md`
  now both match the files on disk (the roadmap is `TransitCheck_Roadmap.md`, with
  an underscore).
- §5's depth-window labels (`duration/2` vs `duration/4`) are **half-widths** in
  the code: `|phase| < (duration/24)/4` spans `duration/2` in total. Internally
  consistent, but ambiguous in the prose. Now stated in the constants.
- `build_feature_table` no longer requires a clean `0..n-1` index. It iterates the
  DataFrame directly and keys on `kepoi_name`, which is the stable unique KOI
  identifier. `kepoi_name` is therefore carried through, and joins the §7.3 drop
  list.
- `measure_width`'s threshold inverts if `depth` is negative (no dip found, flux
  above baseline): "below half depth" then selects most of the window. Left
  unchanged because it would move validated numbers; watch for it in the scaled
  run's depth distribution.
- The roadmap's Phase 2 item *"Add normalization so every star's transit depth is
  on a comparable scale"* needs reconciling with §8, which rules out the AstroNet
  −1/+1 normalisation precisely because it erases depth. Not acted on.

### 10.12 Masking the secondary broke rows with long transits — **found and fixed during the 100-star run**

A regression introduced by the 10.10 fix, caught by the §7.1 broad catch on its
first outing. Masking primary ±1.5 durations *and* secondary ±1.5 durations covers
the whole orbit once `duration/period >= 1/6`, leaving `flatten` no baseline to fit
and raising `ValueError: cannot reshape array of size 0`.

Scale: **507 of 7587 rows** cross that threshold, against **96** that the
primary-only mask would have broken on its own (`duration/period >= 1/3`). So the
10.10 fix would have cost an extra 411 rows at full scale.

**Fixed** with a fallback chain in `extract_features`, stepping down rather than
losing the row: secondary+primary → primary only → no detrending at all. The last
rung is acceptable because `stitch()` has already normalised the flux, so it is a
degradation rather than a catastrophe. `MIN_UNMASKED_FRACTION_FOR_DETREND = 0.25`
sets the floor, and `detrend_fallback` records which rung each row landed on. In the
100-star batch: 94 rows normal, 7 `primary_only`, 3 `none`.

### 10.13 A masked-array coercion reported 100% transit depths — **fixed**

Three of 104 rows in the first run came back with a measured depth of ~1,000,000 ppm
— a 100% dimming, which is not a thing. All three were long-period KOIs
(e.g. K05241.01: 273.7 d period, 8.4 h duration, 4 observed transits) where the local
view is sparse enough that **every bin inside the transit window is empty**.

The mechanism, which is worth understanding because it is silent:

- `binned.flux.value` is an astropy `MaskedNDArray`, not a plain ndarray.
- `np.nanmedian` over a fully-masked slice returns a *masked scalar*.
- `float()` of a masked scalar is **0.0**, not NaN.
- So `depth = baseline - 0.0 = baseline ≈ 1.0`.

The existing guard tested `in_transit.any()` — whether any *bins* fell in the
window — not whether those bins held any *data*. A long-period KOI can have 25 bins
inside the transit window and zero measurements among them: the window exists, the
measurement does not.

**Fixed** in two places. `prepare_transit` and `measure_secondary_depth` now convert
with `np.asarray(..., dtype=float)`, and the guards test
`np.isfinite(flux[region]).any()` instead of `region.any()`.

Note precisely what the cast does: it strips the **mask** and keeps the data
underneath. It works here only because `bin()` writes NaN under the mask for empty
bins — it is not a general masked-to-NaN conversion. The `isfinite` guards carry the
real weight.

Why this one mattered more than its 3/104 rate suggests: all three affected rows were
FALSE POSITIVE, so the bogus value was *correlated with the label*. A tree would have
learned "depth ≈ 1.0 ⇒ FALSE POSITIVE" and scored well on it. This is the failure
mode that looks like success — the same shape as §10.7, and the reason §3's leakage
discipline is worth the friction.

### 10.14 Sibling-KOI contamination of `secondary_depth` — **confirmed, then fixed**

§6's hypothesis, from 2 cases, was that folding on one planet's period lets another
planet's transits form a run deep enough to cross the 3σ threshold. The 100-star run
confirms it:

| | detected a "secondary" | of those, near phase ±0.5 |
|---|---|---|
| multi-KOI stars | 6 of 12 | **0** |
| single-KOI stars | 69 of 86 | 29 (42%) |

On multi-planet systems every detected dip sits at an arbitrary phase. That is the
predicted contamination signature, not a genuine secondary eclipse.

**Fixed.** `build_sibling_mask` computes each sibling KOI's transit times from its own
catalogue `koi_period`/`koi_time0bk` — predictable arithmetic, same principle as §4.2 —
and those points are dropped before the global-view fold. Removing points beats
blanking a phase region, because a sibling on a different period does not land at a
fixed phase here: it smears.

Verified on a synthetic star with a sibling at P/2 and **no** real secondary: unmasked
it reports a fake 2000 ppm "secondary" at phase −0.24; masked it correctly reports 0.0.

**A refinement the synthetic test produced:** contamination needs *near-commensurate*
periods. A sibling at an incommensurate period (3.1 d against 5 d) smears across phase
and never forms a run at all. So this is not a uniform tax on every multi-planet
system — it bites where period ratios are close to simple fractions.

Siblings are also added to the detrend mask, for the §4.3 reason: any dip the trend fit
can see is one it partly divides out. That makes "the mask covers the whole orbit"
likelier, so the 10.12 fallback chain gained a rung and is now
**secondary+siblings → secondary only → primary only → none**, with `detrend_fallback`
recording which fired.

Not addressed: `measure_odd_even_diff` still uses the primary-only mask, so a sibling
transit overlapping the primary's in-transit window would leak in. That needs an
overlap, which is rarer, and it is untested.

### 10.15 `secondary_depth` now uses the min2 variant — **decided by the data**

The 10.2 comparison, settled at 100 stars:

| variant | detections | median depth | fires on CONFIRMED | on FALSE POSITIVE | ratio |
|---|---|---|---|---|---|
| longest run | 75/98 | 170 ppm | 66% | 83% | 1.26x |
| deepest, any length | 75/98 | 200 ppm | 66% | 83% | — |
| **deepest, ≥ 2 bins** | 38/98 | **427 ppm** | **23%** | **48%** | **2.08x** |

Longest-run fires on two-thirds of confirmed planets, which means it is mostly finding
noise. min2 finds half as many dips, but they are 2.5x deeper and discriminate twice
as well.

**`secondary_depth` is now `secondary_depth_deepest_min2`.** All three stay as
diagnostic columns so the comparison can be re-checked at 1000 stars rather than
resting on 38 detections.

### 10.16 The −9.7% depth bias — diagnosed, deliberately not fixed

Two tests against `koi_depth`, both on cached data. Hypothesis: `koi_depth` is the flux
lost at the *minimum* of a fitted model, while we take a median across `|phase| <
duration/4` (a span of `duration/2`); limb darkening curves the transit floor, so a
median over that window sits above the minimum.

**Test 1 — impact parameter. The prediction failed.** Grazing transits should be more
V-shaped and so more undermeasured. They are not:

```
impact 0.00-0.30 (central)   n=26   median signed -12.2%
impact 0.30-0.60             n=20                  -8.0%
impact 0.60-0.85             n=21                  -4.8%
impact 0.85-1.00 (grazing)   n=19                 -10.4%
Spearman rho = -0.095   (i.e. no relationship)
```

The bands are non-monotonic and the most *central* transits are the most
undermeasured — the opposite of the prediction. Transit shape via impact parameter
does not explain the offset.

**Test 2 — window width. Partly confirmed, with a floor.**

```
duration/4   median signed -9.7%   median |err| 10.4%   25 bins
duration/6                 -5.7%                 6.6%   17 bins
duration/8                 -5.3%                 6.6%   13 bins
duration/12                -5.3%                 6.5%    9 bins
```

It shrinks and then **plateaus at −5.3%**, so window width accounts for roughly
45% of the offset (4.4 of 9.7 points) and a ~5% floor survives any narrowing. The
floor is consistent with the definitional gap — median of binned flux versus a model
minimum — which no window choice can close.

**Left at `duration/4`.** Honest accounting of that call: narrowing to /6 does improve
median |error| from 10.4% to 6.6%, which is a real gain, and on the sparse rows the
cost is modest (IQR 18.9% → 21.6%). The reason to decline is not that narrowing is
harmful but that the gain is *agreement with a catalogue number the model never sees*.
A flat multiplicative offset is invisible to a tree, which splits on thresholds learned
from this data. §5 said consistency matters more than agreement; this is that case.

**Test 3 — the real sparsity cliff is at ≤ 1 point/bin, not 4.5.**

```
points/bin   exactly 0    n= 3   median |err| 23.7%
             (0, 1]       n= 3                43.2%
             (1, 4.5]     n=13                10.6%
             (4.5, 15]    n=26                11.0%
             (15, 50]     n=27                 9.7%
             > 50         n=25                10.4%
Spearman rho (points/bin vs |residual|) = -0.031
```

Above 1 point per bin the error is flat at ~10% — sparse rows are **not** measurably
worse, so `LOCAL_N_BINS = 201` stays and §4.5's concern is closed on this evidence.
Below it, the measurement collapses: −24%, −43%. The 10-star batch's 4.5 floor was
simply the wrong number; the cliff is an order of magnitude lower.

**Recommended, not done:** return NaN for `depth` when
`local_points_per_bin_median <= 1`, the same guard-the-feature pattern as 10.5. It
would affect ~6% of rows here and an estimated ~4% of the full catalogue, and it turns
a silently 40%-wrong number into an honest missing one. Flagged rather than applied
because it changes a validated path and the evidence is 6 rows.

*(One caveat on how these numbers were produced: the first version of the Test 3
banding used `pd.cut` with a left edge of 0, which is left-open, so every
`points/bin == 0` row was silently dropped from all bands and the "<1" label actually
meant `(0, 1]`. Corrected above. Worth noting as the same class of error the rest of
this section is about — it did not crash, it just quietly excluded the most interesting
rows.)*

### 10.17 The 1000-star run

1148 KOI rows across 1000 stars. **1118 succeeded (97.4%), 30 precheck rejections
(`koi_num_transits == 0`), zero unexplained failures.** Reached in five passes; the
resume path did the work.

**What broke, and why it was not code.** The first pass produced 37 `unknown`
failures with one root cause: the disk filled at row ~1100. 27 rows raised
`OSError: ENOSPC` directly and the next 10 read back the half-written FITS. The
download cache is **6.8 GB for 1000 stars, so ~48 GB for the full 7587-row
catalogue** — a real constraint to plan for, not an incidental.

Both now classify rather than landing in `unknown`: `diskfull` and `corruptcache`,
the latter evicting the bad file via `evict_star_cache` before giving up, because a
truncated FITS reads back identically forever. A second corrupt-cache signature
turned up during recovery (`Error in reading Data product`, distinct from `Not
recognized as a supported data product`) and is matched too. One star showed the
cascade clearly: a connection reset mid-download truncated a file, which then
poisoned the other four KOI rows on that star until the cache entry was evicted.

**Throttling worked.** `MAST_SEARCH_DELAY_SECONDS = 1.5` with bounded retry: 15
retries fired across 1148 rows and only 8 rows ended transient, against 60
consecutive ConnectionErrors on the unthrottled 100-star pass 3.

**10.1 confirmed at scale: 278 of 1118 rows (24.9%) sit on stars whose search
returns both cadences.** One row in four would have been detrended against a
cadence estimate derived from the wrong products.

**Depth.** Median |error| 9.7%, 75.2% within 15% (up from 69.1% at 100 stars),
median signed −8.2%. The 10.5 continuum shows up cleanly — median |error| rises
9.2% → 9.2% → 10.2% → 12.0% across `duration/period` bands `<0.02`, `0.02-0.05`,
`0.05-0.10`, `>0.10` — degradation, not failure, which is what that section
predicted.

**Points per bin.** Measured p50 = 21.0 against the 23.5 estimated in 10.6; p5 = 1.0
against 1.2. The estimate was close. 18.7% of rows fall below 4.5 points/bin and
2.7% below 1.0. `n_local_bins` ranges 48–201, confirming 201 is nominal.

**The secondary variants, at 10x the sample.** 10.15's decision holds and
strengthens:

| variant | detections | median depth | fires on CONF | on FP | ratio |
|---|---|---|---|---|---|
| longest run | 662/1070 | 150 ppm | 52.9% | 67.5% | 1.28x |
| deepest, any length | 662/1070 | 161 ppm | 52.9% | 67.5% | 1.28x |
| **deepest, ≥ 2 bins** | 324/1070 | **353 ppm** | **11.1%** | **42.4%** | **3.81x** |

min2's discrimination *improved* with sample size, 2.08x → **3.81x**, and the median
detected depth splits 76 ppm on CONFIRMED against 489 ppm on FALSE POSITIVE — a
factor of 6.4. Longest-run still fires on more than half of confirmed planets.

**Sibling masking worked, partially.** Before the fix, multi-KOI stars put 0 of 6
detected dips near phase ±0.5. After it, 13 of 41 (32%), against 149 of 283 (53%)
on single-KOI stars. Most of the contamination is gone; a gap remains, and 10.18
explains it.

### 10.18 CANDIDATE siblings are not masked — **found in the 1000-star results, fixed for future runs**

`build_feature_table` defaults its sibling lookup to the batch it was given, and
`sample_kois` has already filtered that batch to CONFIRMED/FALSE POSITIVE (10.7).
So a star's **CANDIDATE** KOI rows were never masked — and a candidate planet
contaminates the secondary search exactly as much as a confirmed one does.

Measured in the 1000-star table:

```
rows on a star with >= 1 CANDIDATE sibling    detected 12/54    near +/-0.5:  3/12 = 25%
rows on a star with none                      detected 312/1016 near +/-0.5: 159/312 = 51%
```

Half the near-0.5 rate, on the rows with an unmasked sibling. That is the residual
gap in 10.17's sibling result.

**Fixed for future runs** by passing the unfiltered catalogue as `sibling_table` in
`run_1000.py`. The exclusion list and the label filter are different questions and
should never have shared a source — CANDIDATEs are excluded from *training* because
their label is unknown, which is no reason to pretend their transits are not there.

**Not re-run.** 55 of 1118 rows (4.9%) are affected in the current table. Re-running
just those stars is cheap given the cache, but it is a judgement call about whether
the existing table is good enough to proceed on.

### 10.19 The CANDIDATE-sibling re-run — fix confirmed, but the headline comparison was confounded

The 55 affected rows were re-run with the unfiltered catalogue as `sibling_table`.
All 55 succeeded. On those rows the fix did what it should:

```
                     detected      near +/-0.5
before (no cand sib masking)  12/54      3/12 = 25%
after  (cand siblings masked)  6/54      2/6  = 33%
```

**Half the detections disappeared** — 6 of 12 were CANDIDATE-sibling transits being
reported as secondary eclipses.

But the whole-table multi-vs-single comparison barely moved: 32% → **33%**, against a
53% single-KOI baseline. **That comparison was confounded, and the confound is large.**

```
             CONFIRMED   FALSE POSITIVE     CONFIRMED share
multi-KOI          224               55                 80%
single-KOI         190              649                 23%
```

Multi-KOI rows are 80% confirmed planets; single-KOI rows are 77% false positives.
**A genuine secondary eclipse at phase ±0.5 comes from an eclipsing binary**, and
eclipsing binaries are almost all in the single-KOI population. So the raw gap was
mostly comparing a planet population against a binary population — multi-KOI stars
*should* show a lower near-0.5 rate, contamination or not.

Controlled for disposition, with 95% Wilson intervals:

```
CONF / multi      7/23  = 30%  [16%, 51%]
CONF / single     7/17  = 41%  [22%, 64%]
FALS / multi      7/19  = 37%  [19%, 59%]
FALS / single   140/259 = 54%  [48%, 60%]
```

The point estimates still sit lower for multi-KOI, by 11 and 17 points. But the
multi-KOI samples are 17–23 detections and the intervals overlap the single-KOI ones
in both dispositions. **There is no longer statistically distinguishable evidence of
residual sibling contamination.** Something may remain — ephemeris drift or TTVs in
resonant multi-planet systems would put sibling transits outside a linear-ephemeris
±1.5-duration mask, which is the mechanism I would look at first — but this sample
cannot show it. Revisit at full catalogue scale, where the multi-KOI FALSE POSITIVE
cell would hold roughly 7x more rows.

*(A test that did not work, recorded so it is not repeated: measuring the circular
distance from each detection to the nearest sibling transit phase. Over a 1470-day
baseline sibling transits land at essentially every phase, so 89% of off-0.5
detections and 93% of near-0.5 detections were both within 0.02 of one. The statistic
has no discriminating power.)*

### 10.20 Depth error vs bin occupancy — **question closed; 10.16's recommendation retracted**

Measured on 1055 rows of the 1000-star table (`depth_vs_occupancy.png`):

```
points/bin        n   med |err|  med signed     IQR   p90 |err|
0                23      12.3%      -12.3%   26.8%      66.0%
(0,1]            42      19.1%      -15.9%   28.1%      50.0%
(1,2]            32      14.4%      -11.7%   28.5%      55.2%
(2,4.5]          72      10.3%       -8.2%   13.2%      33.5%
(4.5,10]        143       9.6%       -7.1%    8.8%      21.2%
(10,25]         253       7.9%       -7.3%    8.7%      19.6%
(25,60]         217       9.1%       -8.5%    9.7%      18.8%
(60,150]        168      10.0%       -7.9%   10.9%      20.5%
>150            105      12.2%      -10.4%   19.4%      41.1%

Spearman rho (occupancy vs |error|) = -0.028
below 4.5 pts/bin:  12.9% vs 9.3% for the rest   (+3.5 points)
below 1.0 pts/bin:  12.3% vs 9.6% for the rest   (+2.7 points)
```

**The relationship is U-shaped, not monotonic, which is why the overall correlation is
zero.** Both ends are worse than the middle, for unrelated reasons:

- **The sparse end is genuinely starved.** Long period, few transits, little to stack.
- **The dense end is not about density at all.** Occupancy correlates with
  `duration/period` at **rho = +0.972** (and with period at −0.892), because high
  occupancy means a short period, and a short period means the transit occupies a large
  fraction of the orbit. The `>150` bucket has a median period of 0.86 d and
  `duration/period` of 0.19 — it is the 10.5 geometry continuum, showing up under a
  different name.

**Answering the question plainly: sparse rows are worse, but only modestly, and it is
not a cliff.** The two sparsest buckets run 12–19% median error against a ~8–9% floor
in the middle — a few points, not an order of magnitude. The tails are the real story:
p90 reaches 50–66% at low occupancy against ~19% in the middle, so sparse rows are not
biased so much as unreliable.

**`LOCAL_N_BINS = 201` stays.** §4.5's concern is closed on this evidence.

**Retracting 10.16's recommendation.** That section proposed returning NaN for `depth`
when `local_points_per_bin_median <= 1`, on the strength of a 43.2% median error in
that band. At 100 stars that band held **6 rows**. At 1000 stars it holds 65, and the
error is 12–19%, not 43%. The cliff was small-sample noise and the guard is not
justified — it would blank ~6% of rows to remove a few points of median error.

One option deliberately not taken: making occupancy a *feature*, so the tree could
learn to discount sparse rows. At rho = +0.972 with `duration/period`, adding it would
effectively re-introduce `koi_duration`/`koi_period` into X, which §7.3's drop list
excludes. It stays a diagnostic.
