"""Does depth error track bin occupancy?

Plots median |depth error| against measured points-per-bin, bucketed, to settle
whether the sparse rows are measurably worse than the rest -- the open question
from CLAUDE.md 4.5 / 7.5 / 10.6 / 10.16.

Buckets are built with an explicit left edge below zero, because pd.cut's default
left-open interval silently drops every `points/bin == 0` row -- which are exactly
the rows the question is about. See the caveat in 10.16.

    python plot_depth_vs_occupancy.py [checkpoint.csv]
"""

import sys
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import transitcheck_features as tf

warnings.filterwarnings('ignore')

EDGES = [-0.001, 0.001, 1, 2, 4.5, 10, 25, 60, 150, 1e9]
LABELS = ['0', '(0,1]', '(1,2]', '(2,4.5]', '(4.5,10]',
          '(10,25]', '(25,60]', '(60,150]', '>150']


def spearman(a, b):
    ra = pd.Series(a).rank().to_numpy()
    rb = pd.Series(b).rank().to_numpy()
    ra, rb = ra - ra.mean(), rb - rb.mean()
    den = np.sqrt((ra ** 2).sum() * (rb ** 2).sum())
    return float((ra * rb).sum() / den) if den else np.nan


def main(path):
    d = pd.read_csv(path).drop_duplicates(subset='kepoi_name', keep='last')
    d['error_kind'] = d['error_kind'].fillna('')
    ok = d[d['error_kind'] == ''].copy()

    v = ok.dropna(subset=['depth', 'koi_depth_ppm', 'local_points_per_bin_median']).copy()
    v = v[(v['koi_depth_ppm'] > 0) & (v['depth'] > 0)]
    v['rel'] = (v['depth'] * 1e6 - v['koi_depth_ppm']) / v['koi_depth_ppm']
    v['abs_rel'] = v['rel'].abs()
    v['bucket'] = pd.cut(v['local_points_per_bin_median'], EDGES, labels=LABELS)

    print(f"rows with depth, koi_depth and occupancy: {len(v)}")
    print(f"rows assigned to a bucket: {int(v['bucket'].notna().sum())} "
          f"(0 dropped means the edges are right)\n")

    print(f"  {'points/bin':<12} {'n':>5} {'med |err|':>10} {'med signed':>11} "
          f"{'IQR':>8} {'p90 |err|':>10}")
    stats = []
    for band, g in v.groupby('bucket', observed=True):
        iqr = g['rel'].quantile(0.75) - g['rel'].quantile(0.25)
        stats.append({
            'band': str(band), 'n': len(g),
            'med_abs': g['abs_rel'].median(), 'med_signed': g['rel'].median(),
            'iqr': iqr, 'p90': g['abs_rel'].quantile(0.90),
        })
        print(f"  {str(band):<12} {len(g):>5} {g['abs_rel'].median():>9.1%} "
              f"{g['rel'].median():>+10.1%} {iqr:>7.1%} {g['abs_rel'].quantile(0.90):>9.1%}")

    s = pd.DataFrame(stats)

    # --- the two thresholds the question is actually about ---
    print("\n  the two thresholds in question:")
    for label, sparse_mask in (
        ('below 4.5 pts/bin', v['local_points_per_bin_median'] < 4.5),
        ('below 1.0 pts/bin', v['local_points_per_bin_median'] < 1.0),
    ):
        sp, rest = v[sparse_mask], v[~sparse_mask]
        if not len(sp):
            continue
        print(f"    {label:<20} n={len(sp):>4}  med |err| {sp['abs_rel'].median():>6.1%}"
              f"   vs rest n={len(rest):>4} med |err| {rest['abs_rel'].median():>6.1%}"
              f"   delta {sp['abs_rel'].median() - rest['abs_rel'].median():>+6.1%}")

    rho_all = spearman(v['local_points_per_bin_median'], v['abs_rel'])
    nz = v[v['local_points_per_bin_median'] > 1]
    rho_nz = spearman(nz['local_points_per_bin_median'], nz['abs_rel'])
    print(f"\n  Spearman rho (occupancy vs |err|), all rows      : {rho_all:+.3f}")
    print(f"  Spearman rho, excluding rows at <= 1 point/bin   : {rho_nz:+.3f}")

    # --- plot ---
    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(9.5, 7.5), sharex=True,
        gridspec_kw={'height_ratios': [3, 1], 'hspace': 0.12})

    x = np.arange(len(s))
    ax1.axhspan(0, 0.15, color='#2a9d8f', alpha=0.10, zorder=0)
    ax1.axhline(0.15, color='#2a9d8f', lw=1, ls='--', zorder=1,
                label='15% agreement bar (CLAUDE.md §5)')

    overall = v['abs_rel'].median()
    ax1.axhline(overall, color='#264653', lw=1.2, ls=':', zorder=1,
                label=f'overall median |error| = {overall:.1%}')

    colors = ['#e76f51' if b in ('0', '(0,1]') else '#264653' for b in s['band']]
    ax1.plot(x, s['med_abs'], '-', color='#264653', lw=1.4, zorder=2, alpha=0.6)
    ax1.scatter(x, s['med_abs'], s=90, c=colors, zorder=3, edgecolor='white', lw=1.2)
    ax1.errorbar(x, s['med_abs'], yerr=[np.zeros(len(s)), s['p90'] - s['med_abs']],
                 fmt='none', ecolor='#8d99ae', lw=1, capsize=3, zorder=2,
                 label='median → p90 of |error|')

    ax1.set_ylabel('median |depth error| vs koi_depth')
    ax1.set_title('Depth accuracy against measured bin occupancy\n'
                  f'{len(v)} rows from the 1000-star run', loc='left', fontsize=11)
    ax1.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax1.legend(loc='upper right', fontsize=8.5, framealpha=0.95)
    ax1.grid(axis='y', alpha=0.25)
    ax1.set_ylim(bottom=0)

    for xi, row in zip(x, s.itertuples()):
        if row.band in ('0', '(0,1]'):
            ax1.annotate(f'{row.med_abs:.0%}', (xi, row.med_abs),
                         textcoords='offset points', xytext=(0, 11),
                         ha='center', fontsize=9, color='#e76f51', weight='bold')

    ax2.bar(x, s['n'], color='#a8dadc', edgecolor='#457b9d', lw=0.8)
    for xi, n in zip(x, s['n']):
        ax2.annotate(str(n), (xi, n), textcoords='offset points', xytext=(0, 2),
                     ha='center', fontsize=8, color='#1d3557')
    ax2.set_ylabel('rows')
    ax2.set_xlabel('measured points per bin (local view)')
    ax2.set_xticks(x)
    ax2.set_xticklabels(s['band'])
    ax2.grid(axis='y', alpha=0.25)
    ax2.set_ylim(0, s['n'].max() * 1.25)

    fig.savefig('depth_vs_occupancy.png', dpi=150, bbox_inches='tight',
                facecolor='white')
    print("\nwrote depth_vs_occupancy.png")


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else 'features_1000.csv')
