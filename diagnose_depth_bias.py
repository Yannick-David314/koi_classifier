"""Diagnose the systematic -9.7% depth offset against koi_depth.

Leading explanation is definitional, not a bug: koi_depth is the flux lost at the
MINIMUM of a fitted Mandel-Agol model (one point), while we take a median across
|phase| < duration/4 (a span of duration/2). Limb darkening curves the transit
floor, so mid-transit is deeper than the window edges and a median over the window
is systematically shallower than the minimum.

Two tests, both free because the FITS are already cached:

  1. Correlate the signed residual against koi_impact. Grazing transits (high
     impact parameter) are more V-shaped, so a median over the central half loses
     proportionally more. If the bias tracks impact, shape is the explanation.
  2. Re-measure depth at duration/6 and /8 on the same rows. If the bias shrinks
     monotonically toward zero, it is window width and nothing more mysterious.

This is a diagnostic, NOT a change. The window stays at /4: narrowing it means
fewer bins in the median, and 24% of rows are already below 4.5 points per bin, so
it would trade a harmless uniform bias for real noise on the sparsest quarter of
the data. A flat multiplicative offset is invisible to a tree, which splits on
thresholds learned from this data, not from the catalogue.

    python diagnose_depth_bias.py [checkpoint.csv]
"""

import sys
import warnings

import numpy as np
import pandas as pd

import transitcheck_features as tf

warnings.filterwarnings('ignore')


def rule(title):
    print(f"\n{'=' * 72}\n{title}\n{'=' * 72}")


def spearman(a, b):
    """Rank correlation without pulling in scipy."""
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    denom = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / denom) if denom else np.nan


def main(path):
    table = pd.read_csv(path).drop_duplicates(subset='kepoi_name', keep='last')
    table['error_kind'] = table['error_kind'].fillna('')
    ok = table[table['error_kind'] == ''].copy()

    cat = pd.read_csv("MyProject_sync.csv", low_memory=False)
    cols = ['kepoi_name', 'koi_impact', 'koi_period', 'koi_time0bk',
            'koi_duration', 'koi_depth', 'koi_prad', 'koi_ror']
    ok = ok.merge(cat[cols], on='kepoi_name', how='left', suffixes=('', '_cat'))

    d = ok.dropna(subset=['depth', 'koi_depth_ppm']).copy()
    d = d[(d['koi_depth_ppm'] > 0) & (d['depth'] > 0)]
    d['rel'] = (d['depth'] * 1e6 - d['koi_depth_ppm']) / d['koi_depth_ppm']

    rule("TEST 1 — does the bias track koi_impact (transit shape)?")
    di = d.dropna(subset=['koi_impact'])
    print(f"rows with an impact parameter: {len(di)}")
    print(f"overall median signed residual: {d['rel'].median():+.1%}")
    print()
    print("A grazing transit (impact -> 1) is more V-shaped, so a median over the")
    print("central half should lose proportionally MORE depth => residual more negative.")
    print()
    bands = pd.cut(di['koi_impact'], [0, 0.3, 0.6, 0.85, 1.0, 10],
                   labels=['0-0.3 central', '0.3-0.6', '0.6-0.85', '0.85-1.0 grazing', '>1.0'])
    print(f"  {'impact band':<20} {'n':>4} {'median signed':>15} {'median |err|':>14}")
    for band, g in di.groupby(bands, observed=True):
        print(f"  {str(band):<20} {len(g):>4} {g['rel'].median():>+14.1%} "
              f"{g['rel'].abs().median():>13.1%}")

    rho = spearman(di['koi_impact'], di['rel'])
    print(f"\n  Spearman rho (impact vs signed residual): {rho:+.3f}")
    print("  negative rho => more grazing means more undermeasured => shape explains it")

    rule("TEST 2 — does the bias shrink as the depth window narrows?")
    print("Re-measuring depth at duration/4 (current), /6 and /8 on the same rows.")
    print("Monotonic shrink toward zero => window width, nothing more mysterious.\n")

    fractions = [4, 6, 8, 12]
    rows = []
    cache = tf.StarCache(download_dir=tf.DEFAULT_DOWNLOAD_DIR)
    attempted = 0

    for _, r in d.sort_values('kepid').iterrows():
        attempted += 1
        try:
            star_lc, _ = cache.get(r['kepid'])
            period, epoch = r['koi_period'], r['koi_time0bk']
            duration = r['koi_duration']

            transit_mask = tf.build_transit_mask(star_lc, period, epoch, duration)
            secondary_mask = tf.build_transit_mask(star_lc, period, epoch, duration,
                                                   include_secondary=True)
            mask = secondary_mask
            if (~mask).mean() < tf.MIN_UNMASKED_FRACTION_FOR_DETREND:
                mask = transit_mask
            if (~mask).mean() < tf.MIN_UNMASKED_FRACTION_FOR_DETREND:
                mask = None
            flat, _ = tf.flatten_star(star_lc, mask, duration)

            folded = flat.fold(period=period, epoch_time=epoch)
            half_window_days = (duration * tf.LOCAL_WINDOW_DURATIONS / 2) / 24
            local = folded[np.abs(folded.phase.value) < half_window_days]
            if len(local) == 0:
                continue
            binned = local.bin(
                time_bin_size=(duration * tf.LOCAL_WINDOW_DURATIONS / tf.LOCAL_N_BINS) * tf.u.hour)
            flux = np.asarray(binned.flux.value, dtype=float)
            phase = np.asarray(binned.phase.value, dtype=float)

            entry = {'kepoi_name': r['kepoi_name'], 'koi_depth_ppm': r['koi_depth_ppm'],
                     'koi_impact': r['koi_impact'],
                     'ppb': r['local_points_per_bin_median']}
            for frac in fractions:
                it = np.abs(phase) < (duration / 24) / frac
                if not np.isfinite(flux[it]).any() or not np.isfinite(flux[~it]).any():
                    entry[f'd{frac}'] = np.nan
                    entry[f'n{frac}'] = 0
                    continue
                base = float(np.nanmedian(flux[~it]))
                entry[f'd{frac}'] = base - float(np.nanmedian(flux[it]))
                entry[f'n{frac}'] = int(np.isfinite(flux[it]).sum())
            rows.append(entry)
        except Exception as e:
            print(f"  [skip] {r['kepoi_name']}: {type(e).__name__}: {e}")

    w = pd.DataFrame(rows)
    w.to_csv('depth_window_scan.csv', index=False)
    print(f"re-measured {len(w)} of {attempted} rows (saved to depth_window_scan.csv)\n")
    print(f"  {'window':<16} {'n':>4} {'median signed':>15} {'median |err|':>14} "
          f"{'med bins used':>14}")
    for frac in fractions:
        col = f'd{frac}'
        g = w.dropna(subset=[col])
        rel = (g[col] * 1e6 - g['koi_depth_ppm']) / g['koi_depth_ppm']
        label = f"duration/{frac}"
        print(f"  {label:<16} {len(g):>4} {rel.median():>+14.1%} "
              f"{rel.abs().median():>13.1%} {g[f'n{frac}'].median():>13.0f}")

    print("\n  the cost of narrowing, on the sparsest rows (points/bin < 4.5):")
    sparse = w[w['ppb'] < 4.5]
    print(f"  {'window':<16} {'n':>4} {'median signed':>15} {'spread (IQR)':>14} "
          f"{'med bins used':>14}")
    for frac in fractions:
        col = f'd{frac}'
        g = sparse.dropna(subset=[col])
        if not len(g):
            continue
        rel = (g[col] * 1e6 - g['koi_depth_ppm']) / g['koi_depth_ppm']
        iqr = rel.quantile(0.75) - rel.quantile(0.25)
        print(f"  duration/{frac:<7} {len(g):>4} {rel.median():>+14.1%} "
              f"{iqr:>13.1%} {g[f'n{frac}'].median():>13.0f}")
    print("\n  a shrinking median with a WIDENING spread is the trade being refused:")
    print("  less bias, more noise, on exactly the rows that can least afford it.")

    rule("TEST 3 — is depth error worse on sparse rows? (gates LOCAL_N_BINS)")
    s = d.dropna(subset=['local_points_per_bin_median'])
    bands = pd.cut(s['local_points_per_bin_median'], [0, 1, 4.5, 15, 50, 1e9],
                   labels=['<1', '1-4.5', '4.5-15', '15-50', '>50'])
    print(f"  {'points/bin':<14} {'n':>4} {'median signed':>15} {'median |err|':>14} "
          f"{'spread (IQR)':>14}")
    for band, g in s.groupby(bands, observed=True):
        iqr = g['rel'].quantile(0.75) - g['rel'].quantile(0.25)
        print(f"  {str(band):<14} {len(g):>4} {g['rel'].median():>+14.1%} "
              f"{g['rel'].abs().median():>13.1%} {iqr:>13.1%}")
    rho_s = spearman(s['local_points_per_bin_median'], s['rel'].abs())
    print(f"\n  Spearman rho (points/bin vs |residual|): {rho_s:+.3f}")
    print("  ~0 => sparse rows are not measurably worse => 201 bins stays, concern closed")

    print()


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else tf.DEFAULT_CHECKPOINT)
