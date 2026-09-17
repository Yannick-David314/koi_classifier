"""Report on a feature-extraction run.

Answers the four questions the 100-star run was commissioned to answer:
  1. error-kind histogram
  2. measured depth vs koi_depth agreement
  3. points-per-bin, measured rather than estimated
  4. the three secondary variants and their phase fractions

Reads the checkpoint written by build_feature_table. Pure analysis — no network.

    python analyse_run.py [checkpoint.csv]
"""

import sys

import numpy as np
import pandas as pd

import transitcheck_features as tf


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def pct(x, n):
    return f"{x}/{n} ({x / n:.1%})" if n else f"{x}/0"


def main(path):
    df = pd.read_csv(path).drop_duplicates(subset='kepoi_name', keep='last')
    df['error_kind'] = df['error_kind'].fillna('')
    ok = df[df['error_kind'] == ''].copy()

    rule("1. RUN SUMMARY AND ERROR KINDS")
    print(f"rows: {len(df)}   stars: {df['kepid'].nunique()}   "
          f"succeeded: {pct(len(ok), len(df))}")
    print(f"class balance (succeeded rows): "
          f"{ok['disposition'].value_counts().to_dict()}")

    print("\nerror_kind histogram:")
    for kind, count in df['error_kind'].value_counts().items():
        label = kind if kind else '(success)'
        print(f"  {label:<12} {count:>4}  {count / len(df):>6.1%}")

    failed = df[df['error_kind'] != '']
    if len(failed):
        print("\nfailure messages (first 60 chars, grouped):")
        msgs = failed['error'].fillna('').str.slice(0, 60).value_counts()
        for msg, count in msgs.items():
            print(f"  {count:>3}x  {msg}")

    # --- what the cadence pin actually caught -------------------------------
    rule("1b. CADENCE MIXING (was section 10.1 real at scale?)")
    seen = ok['cadences_seen'].fillna('').astype(str)
    mixed = seen.str.contains(r'\|')
    print(f"stars whose search returned MORE than one cadence: {pct(int(mixed.sum()), len(ok))}")
    print("cadence sets returned:")
    for value, count in seen.value_counts().items():
        print(f"  {value!r:<16} {count:>4}")
    bad_origin = ok[ok['flux_origin'].fillna('').str.lower() != tf.EXPECTED_FLUX_ORIGIN]
    print(f"rows whose FLUX_ORIGIN was not {tf.EXPECTED_FLUX_ORIGIN}: {len(bad_origin)}")

    # --- 2. depth agreement -------------------------------------------------
    rule("2. MEASURED DEPTH vs koi_depth")
    d = ok.dropna(subset=['depth', 'koi_depth_ppm']).copy()
    d = d[d['koi_depth_ppm'] > 0]
    d['measured_ppm'] = d['depth'] * 1e6
    d['rel_err'] = (d['measured_ppm'] - d['koi_depth_ppm']) / d['koi_depth_ppm']

    print(f"rows with both measured and catalogue depth: {len(d)}")
    print(f"  median relative error : {d['rel_err'].median():+.1%}")
    print(f"  mean   relative error : {d['rel_err'].mean():+.1%}   "
          f"(a systematic offset, if any, shows here)")
    print(f"  median |relative error|: {d['rel_err'].abs().median():.1%}")
    print(f"  within 15% (the 10-star bar): {pct(int((d['rel_err'].abs() <= 0.15).sum()), len(d))}")
    print(f"  within 25%                  : {pct(int((d['rel_err'].abs() <= 0.25).sum()), len(d))}")
    print(f"  negative depth (no dip found): {int((ok['depth'] < 0).sum())}")

    # Screen for the 10.13 signature specifically, NOT for "deep".
    #
    # That bug coerced a fully-masked in-transit median to 0.0, so depth came out as
    # baseline, i.e. ~1.0 -- a reported 100% dimming. An earlier version of this check
    # used depth > 0.5, which is far too aggressive: K07367.01 measures 0.554 and the
    # catalogue agrees at 730,498 ppm (73%). That is a real, very deep eclipsing
    # binary, not a bug. Scoping to ~1.0 is what distinguishes the two.
    SIGNATURE_FLOOR = 0.9
    suspect = ok[ok['depth'] > SIGNATURE_FLOOR]
    deepest = ok['depth'].max()
    print(f"  deepest measured depth: {deepest:.4f} "
          f"({deepest * 1e6:,.0f} ppm)" if pd.notna(deepest) else "  no depths")
    print(f"  rows matching the 10.13 signature (depth > {SIGNATURE_FLOOR}): {len(suspect)}")
    for _, r in suspect.iterrows():
        cat_ppm = r['koi_depth_ppm']
        agrees = pd.notna(cat_ppm) and cat_ppm > 0 and abs(r['depth'] * 1e6 - cat_ppm) / cat_ppm < 0.5
        print(f"    {r['kepoi_name']}  measured {r['depth'] * 1e6:,.0f} ppm  "
              f"catalogue {cat_ppm if pd.notna(cat_ppm) else 'n/a'}  "
              f"-> {'catalogue agrees, likely real' if agrees else 'NO catalogue support, investigate'}")

    print("\n  |rel err| by catalogue depth:")
    bins = [0, 500, 1000, 5000, 20000, 1e9]
    labels = ['<500 ppm', '500-1k', '1k-5k', '5k-20k', '>20k']
    d['depth_band'] = pd.cut(d['koi_depth_ppm'], bins=bins, labels=labels)
    for band, g in d.groupby('depth_band', observed=True):
        print(f"    {band:<10} n={len(g):>3}  median |err| {g['rel_err'].abs().median():.1%}"
              f"   median signed {g['rel_err'].median():+.1%}")

    print("\n  |rel err| vs duration/period (the section 10.5 continuum):")
    d['dop_band'] = pd.cut(d['duration_over_period'],
                           bins=[0, 0.02, 0.05, 0.1, 1.0],
                           labels=['<0.02', '0.02-0.05', '0.05-0.10', '>0.10'])
    for band, g in d.groupby('dop_band', observed=True):
        print(f"    {band:<10} n={len(g):>3}  median |err| {g['rel_err'].abs().median():.1%}"
              f"   median signed {g['rel_err'].median():+.1%}")

    print(f"\n  detrend_masked_fraction: median {ok['detrend_masked_fraction'].median():.1%}, "
          f"max {ok['detrend_masked_fraction'].max():.1%}")

    # --- 3. points per bin --------------------------------------------------
    rule("3. POINTS PER BIN, MEASURED (section 10.6)")
    ppb = ok['local_points_per_bin_median'].dropna()
    print("estimated before the run (koi_num_transits x bin_width / 29.4 min):")
    print("  p1=0.0  p5=1.2  p25=8.7  p50=23.5  p75=67.4  p95=221.6")
    if len(ppb):
        qs = np.percentile(ppb, [1, 5, 25, 50, 75, 95])
        print("measured:")
        print("  p1={:.1f}  p5={:.1f}  p25={:.1f}  p50={:.1f}  p75={:.1f}  p95={:.1f}"
              .format(*qs))
        print(f"\n  rows below 4.5 points/bin (the 10-star floor): "
              f"{pct(int((ppb < 4.5).sum()), len(ppb))}")
        print(f"  rows below 1.0 points/bin                    : "
              f"{pct(int((ppb < 1).sum()), len(ppb))}")

    filled = ok.dropna(subset=['n_local_bins', 'n_local_bins_filled'])
    if len(filled):
        frac = filled['n_local_bins_filled'] / filled['n_local_bins']
        print(f"\n  bins actually populated: median {frac.median():.1%}, "
              f"min {frac.min():.1%}")
        print(f"  rows with any empty bin: {pct(int((frac < 1).sum()), len(filled))}")
        print(f"  n_local_bins (nominal 201): min {int(filled['n_local_bins'].min())}, "
              f"median {int(filled['n_local_bins'].median())}, "
              f"max {int(filled['n_local_bins'].max())}")

    # --- 4. secondary variants ---------------------------------------------
    rule("4. THE THREE SECONDARY VARIANTS (sections 10.2, 10.9)")
    variants = [
        ('secondary_depth_longest', 'secondary_phase_fraction_longest', 'longest run'),
        ('secondary_depth_deepest', 'secondary_phase_fraction_deepest', 'deepest, any length'),
        ('secondary_depth_deepest_min2', 'secondary_phase_fraction_deepest_min2',
         f'deepest, >={tf.SECONDARY_MIN_RUN_BINS} bins  <-- FEATURE'),
    ]

    guarded = ok[ok['secondary_guard'].fillna('') != '']
    print(f"rows where the search was guarded out: {pct(len(guarded), len(ok))}")
    # Group by cause rather than listing every distinct blanking fraction.
    causes = guarded['secondary_guard'].fillna('').str.replace(
        r'covers [\d.]+ of', 'covers N of', regex=True)
    for reason, count in causes.value_counts().items():
        print(f"  {count:>4}x  {reason}")

    print(f"\nsibling masking (section 10.14):")
    multi_rows = ok[ok['n_siblings'].fillna(0) > 0]
    print(f"  rows with >=1 sibling KOI: {pct(len(multi_rows), len(ok))}")
    if len(multi_rows):
        print(f"  sibling_masked_fraction: median "
              f"{multi_rows['sibling_masked_fraction'].median():.1%}, "
              f"max {multi_rows['sibling_masked_fraction'].max():.1%}")
    print(f"  detrend_fallback: {ok['detrend_fallback'].fillna('').value_counts().to_dict()}")

    print("\n{:<26} {:>8} {:>8} {:>10} {:>12}".format(
        "variant", "found", "zero", "med depth", "near +/-0.5"))
    for depth_col, phase_col, label in variants:
        v = ok.dropna(subset=[depth_col])
        found = v[v[depth_col] > 0]
        ph = found[phase_col].dropna().abs()
        near = int((ph > 0.45).sum())
        med = found[depth_col].median() if len(found) else float('nan')
        print("{:<26} {:>8} {:>8} {:>10} {:>12}".format(
            label, f"{len(found)}/{len(v)}", int((v[depth_col] == 0).sum()),
            f"{med * 1e6:.0f} ppm" if np.isfinite(med) else "-",
            f"{near}/{len(ph)}" if len(ph) else "-"))

    print("\nphase-fraction distribution of detected dips (|frac|):")
    print("  a genuine secondary sits near 0.5; anything else is contamination")
    for depth_col, phase_col, label in variants:
        v = ok[(ok[depth_col].notna()) & (ok[depth_col] > 0)]
        ph = v[phase_col].dropna().abs()
        if not len(ph):
            print(f"  {label:<26} (none found)")
            continue
        hist, _ = np.histogram(ph, bins=[0, 0.1, 0.2, 0.3, 0.4, 0.45, 0.5001])
        print(f"  {label:<26} " + "  ".join(
            f"{lo}-{hi}:{c}" for (lo, hi), c in zip(
                [('0', '.1'), ('.1', '.2'), ('.2', '.3'), ('.3', '.4'),
                 ('.4', '.45'), ('.45', '.5')], hist)))

    print("\nseparation by disposition (median depth of detected secondaries, ppm):")
    for depth_col, _, label in variants:
        v = ok[(ok[depth_col].notna())]
        parts = []
        for disp in ('CONFIRMED', 'FALSE POSITIVE'):
            g = v[v['disposition'] == disp]
            nonzero = g[g[depth_col] > 0]
            parts.append(f"{disp[:4]}: {len(nonzero)}/{len(g)} found, "
                         f"med {nonzero[depth_col].median() * 1e6:.0f}" if len(nonzero)
                         else f"{disp[:4]}: 0/{len(g)} found")
        print(f"  {label:<26} " + " | ".join(parts))

    # --- multi-planet contamination check (section 6) -----------------------
    rule("5. MULTI-PLANET CONTAMINATION CHECK (section 6 hypothesis)")
    multi = ok[ok['n_koi_for_star'] > 1]
    single = ok[ok['n_koi_for_star'] <= 1]
    print(f"rows on multi-KOI stars: {len(multi)}   on single-KOI stars: {len(single)}")
    for depth_col, phase_col, label in variants:
        for name, group in (('multi ', multi), ('single', single)):
            v = group.dropna(subset=[depth_col])
            if not len(v):
                continue
            found = v[v[depth_col] > 0]
            ph = found[phase_col].dropna().abs()
            near = int((ph > 0.45).sum())
            print(f"  {label:<26} {name}  found {len(found):>3}/{len(v):<3} "
                  f"near +/-0.5: {near}/{len(ph) if len(ph) else 0}")

    # --- feature health -----------------------------------------------------
    rule("6. FEATURE HEALTH (NaN rates on successful rows)")
    for name in tf.FEATURE_NAMES:
        col = ok[name]
        print(f"  {name:<18} NaN {col.isna().sum():>3}/{len(ok):<3} "
              f"({col.isna().mean():>5.1%})   "
              f"median {col.median():.6g}" if col.notna().any()
              else f"  {name:<18} all NaN")

    print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tf.DEFAULT_CHECKPOINT)
