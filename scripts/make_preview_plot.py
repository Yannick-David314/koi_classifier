"""Make the repo's social preview image: a confirmed planet beside an eclipsing binary.

Every step that touches the light curve is the pipeline's own -- load_star, the
detrend mask extract_features builds, flatten_star, prepare_transit -- so what is
plotted is the curve the features were measured from, not a lookalike.

Run from the project root:
    .venv/bin/python scripts/make_preview_plot.py

Writes social_preview.png (2400 x 1000 px, transparent) to the project root.
"""
import os
import sys

import matplotlib
matplotlib.use('Agg')          # render straight to file; no window needed
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, MaxNLocator
import numpy as np
import pandas as pd

# This script lives in scripts/, one level below transitcheck_features.py. Python
# only looks for imports next to the script being run, so put the project root on
# the search path before importing the pipeline.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
import transitcheck_features as tf  # noqa: E402

CATALOGUE_PATH = os.path.join(PROJECT_ROOT, 'MyProject_sync.csv')
OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'social_preview.png')

# (kepoi_name, panel label, series colour)
PANELS = (
    # Kepler-422 b: highest-SNR confirmed planet in features_2000.csv after filtering
    # for a single-KOI star, normal detrending, no secondary, depth within 10% of
    # koi_depth. Swap for 'K00910.01' (Kepler-721 b, ~1,080 ppm) to show a typical
    # small planet at the cost of a noisier dip.
    ('K00022.01', 'Confirmed planet', '#2a78d6'),
    # KIC 10480982, the 7.4% eclipsing binary from the 10-star validation batch.
    ('K00744.01', 'False positive (eclipsing binary)', '#eb6834'),
)

# Neutral ink for everything that isn't data. The background is transparent, so it
# has to read on whatever is behind it: this grey holds >= 4:1 contrast on both white
# and GitHub's dark theme (#0d1117).
TEXT_COLOUR = '#777670'

FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI = 12, 5, 200     # 2400 x 1000 px, 2.4 : 1
TICK_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
PANEL_LABEL_FONT_SIZE = 20


def local_view(row, catalogue):
    """Run one KOI through the pipeline and return its local view in plot units.

    In:  row       -- pandas Series, the KOI's catalogue row
         catalogue -- DataFrame, the full KOI table (siblings are looked up here,
                      unfiltered, for the CLAUDE.md 10.18 reason)
    Out: dict of ndarrays: raw_hours, raw_ppm, bin_hours, bin_ppm
         hours are from transit centre; ppm is flux change relative to baseline
    """
    period = float(row['koi_period'])            # days
    epoch_time = float(row['koi_time0bk'])       # BKJD
    duration = float(row['koi_duration'])        # hours

    star_lc, _ = tf.load_star(int(row['kepid']), download_dir=tf.DEFAULT_DOWNLOAD_DIR)

    # The same mask extract_features uses on its first rung: primary, expected
    # secondary, and every sibling KOI's transits withheld from the trend fit.
    detrend_mask = (
        tf.build_transit_mask(star_lc, period, epoch_time, duration, include_secondary=True)
        | tf.build_sibling_mask(star_lc, tf.siblings_for(row, catalogue))
    )
    # extract_features steps down a fallback chain when this mask leaves too little
    # baseline to fit. Neither star here is close (transits under 3% of the orbit),
    # so stop loudly rather than quietly plot a curve the pipeline would not produce.
    if (~detrend_mask).mean() < tf.MIN_UNMASKED_FRACTION_FOR_DETREND:
        raise RuntimeError(f"{row['kepoi_name']} would need a detrend fallback; "
                           "pick a KOI with a shorter transit relative to its period")

    flat_lc, _ = tf.flatten_star(star_lc, detrend_mask, duration)

    depth, bin_flux, bin_phase, baseline = tf.prepare_transit(flat_lc, period, epoch_time, duration)
    if not np.isfinite(depth):
        raise RuntimeError(f"{row['kepoi_name']}: prepare_transit could not measure a depth")

    # prepare_transit returns only the binned curve, so take the raw points from the
    # same fold and the same window it uses internally (as plot_local_view does).
    folded = flat_lc.fold(period=period, epoch_time=epoch_time)          # phase in days
    half_window_days = (duration * tf.LOCAL_WINDOW_DURATIONS / 2) / 24
    raw = folded[np.abs(folded.phase.value) < half_window_days]

    raw_phase = np.asarray(raw.phase.value, dtype=float)
    raw_flux = np.asarray(raw.flux.value, dtype=float)
    keep_raw = np.isfinite(raw_flux)
    keep_bin = np.isfinite(bin_flux)      # empty bins come back as NaN

    print(f"{row['kepoi_name']}: depth {depth * 1e6:,.0f} ppm "
          f"(koi_depth {row['koi_depth']:,.0f}), {keep_raw.sum():,} raw points, "
          f"{keep_bin.sum()} filled bins")

    # Depth is "baseline minus flux" as a fraction, so the same subtraction, times a
    # million, puts every point on the scale the depth feature is reported in.
    return {
        'raw_hours': raw_phase[keep_raw] * 24,
        'raw_ppm': (raw_flux[keep_raw] - baseline) * 1e6,
        'bin_hours': bin_phase[keep_bin] * 24,
        'bin_ppm': (bin_flux[keep_bin] - baseline) * 1e6,
    }


def thousands_with_true_minus(value, _position):
    """Tick text like '-60,000', using a real minus sign rather than a hyphen."""
    return f"{value:,.0f}".replace('-', '−')


def draw_panel(ax, view, label, colour):
    """Raw points faint, binned points on top, one panel's styling."""
    # Raw points: tiny and nearly transparent, so thousands of them read as a haze
    # whose thickness shows the noise, not as individual dots. rasterized=True
    # stores this layer as pixels, which matters if the figure is ever saved as
    # PDF/SVG -- thousands of vector dots make a huge, slow file.
    ax.scatter(view['raw_hours'], view['raw_ppm'], s=5, color=colour, alpha=0.12,
               linewidths=0, rasterized=True, zorder=1)

    # Binned points: what the features are measured from, so they lead.
    ax.scatter(view['bin_hours'], view['bin_ppm'], s=22, color=colour,
               linewidths=0, zorder=3)

    # Fit the y-range to the binned curve plus the bulk of the raw scatter. Using the
    # raw extremes would let a handful of stray points set the scale; percentiles
    # ignore them (they are clipped off-panel, not deleted).
    low = min(view['bin_ppm'].min(), np.percentile(view['raw_ppm'], 1))
    high = max(view['bin_ppm'].max(), np.percentile(view['raw_ppm'], 99))
    pad = 0.08 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    ax.set_xlim(view['raw_hours'].min(), view['raw_hours'].max())

    ax.set_xlabel('Hours from transit centre', fontsize=AXIS_LABEL_FONT_SIZE, color=TEXT_COLOUR)
    ax.set_ylabel('Flux change (ppm)', fontsize=AXIS_LABEL_FONT_SIZE, color=TEXT_COLOUR)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))   # few ticks: big labels need room
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_with_true_minus))
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_with_true_minus))
    ax.tick_params(labelsize=TICK_FONT_SIZE, colors=TEXT_COLOUR, length=0, pad=8)

    # Recessive frame: keep only the left and bottom axis lines, and one faint set
    # of horizontal gridlines so depths can be read off without a box around them.
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(TEXT_COLOUR)
        ax.spines[side].set_alpha(0.5)
    ax.grid(axis='y', color=TEXT_COLOUR, alpha=0.18, linewidth=1)
    ax.set_axisbelow(True)
    ax.patch.set_alpha(0)

    # Above the panel, not inside it: a deep dip fills the bottom of the plot and the
    # baseline runs along the top, so there is no corner the data reliably leaves
    # empty. Neutral ink; the colour of the points below carries the identity, so
    # the label is not relying on colour alone.
    ax.text(0, 1.05, label, transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE,
            color=TEXT_COLOUR, ha='left', va='bottom')


def main():
    catalogue = pd.read_csv(CATALOGUE_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    # Fixed margins instead of bbox_inches='tight', which would crop the canvas
    # and break the exact 2400 x 1000 size.
    # The gap between panels has to hold the right panel's tick labels ("-80,000")
    # and its rotated axis label, hence the wide wspace.
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.17, top=0.87, wspace=0.42)

    for ax, (kepoi_name, label, colour) in zip(axes, PANELS):
        row = catalogue.loc[catalogue['kepoi_name'] == kepoi_name].iloc[0]
        draw_panel(ax, local_view(row, catalogue), label, colour)

    fig.savefig(OUTPUT_PATH, dpi=DPI, transparent=True)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == '__main__':
    main()
