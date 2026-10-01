"""Make the repo's preview images: a confirmed planet beside an eclipsing binary.

Every step that touches the light curve is the pipeline's own -- load_star, the
detrend mask extract_features builds, flatten_star, prepare_transit -- so what is
plotted is the curve the features were measured from, not a lookalike.

Two figures come out of one pipeline run, drawn from the same data:

  social_preview.png       2560 x 1280, 2:1, full axes. Stands alone: GitHub's
                           social preview shape, safe for LinkedIn's crop.
  social_preview_card.png  2560 x 984, made to sit under the title band of a
                           designed 1280 x 640 card, placed at 1200 x 461 card px
                           (1 card px = ~2.13 image px). Text is sized for that scale.

Run from the project root:
    .venv/bin/python scripts/make_preview_plot.py
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
FULL_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'assets', 'social_preview.png')
CARD_OUTPUT_PATH = os.path.join(PROJECT_ROOT, 'assets', 'social_preview_card.png')

PANELS = (
    # Kepler-422 b: highest-SNR confirmed planet in features_2000.csv after filtering
    # for a single-KOI star, normal detrending, no secondary, depth within 10% of
    # koi_depth. Swap for 'K00910.01' (Kepler-721 b, ~1,080 ppm) to show a typical
    # small planet at the cost of a noisier dip.
    {'kepoi_name': 'K00022.01',
     'label': 'Confirmed planet',
     'card_title': 'Confirmed planet', 'card_subtitle': 'Kepler-422 b',
     'colour': '#2a78d6', 'card_colour': '#3987e5'},
    # KIC 10480982, the 7.4% eclipsing binary from the 10-star validation batch.
    {'kepoi_name': 'K00744.01',
     'label': 'False positive (eclipsing binary)',
     'card_title': 'False positive', 'card_subtitle': 'Eclipsing binary',
     'colour': '#eb6834', 'card_colour': '#d95926'},
)
# 'colour' and 'card_colour' are the same two hues (slots 1 and 2 of a palette
# validated for colour-blind readers), stepped for a light and a dark surface. The
# full figure is transparent and must work on either; the card is known to be dark.

# ---- full figure -------------------------------------------------------------

# Neutral ink for everything that isn't data. The background is transparent, so it
# has to read on whatever is behind it: this grey holds >= 4:1 contrast on both white
# and GitHub's dark theme (#0d1117).
TEXT_COLOUR = '#777670'

# Exactly 2:1, GitHub's social preview shape, at twice its recommended 1280 x 640 so
# it stays sharp when downscaled.
FIG_WIDTH_IN, FIG_HEIGHT_IN, DPI = 12.8, 6.4, 200   # 2560 x 1280 px

# LinkedIn previews the same image at 1.91:1, cropping it from the centre. That trims
# (1 - 1.91/2) / 2 = 2.25% of the width off each side, so nothing may sit in the outer
# SAFE_SIDE_FRACTION of either edge. 3.5% leaves a margin over the 2.25% crop.
SAFE_SIDE_FRACTION = 0.035
TICK_FONT_SIZE = 16
AXIS_LABEL_FONT_SIZE = 18
PANEL_LABEL_FONT_SIZE = 20

# ---- card figure -------------------------------------------------------------

# Shaped to the space under the card's title band rather than 2:1, so it fills that
# space's width instead of its height. The designer places it at 1200 x 461 card px.
CARD_ASPECT = 2.6
CARD_WIDTH_IN = 12.8                                   # 2560 px at DPI
CARD_HEIGHT_IN = CARD_WIDTH_IN / CARD_ASPECT

# The card's background is near-black (#0a0b0d), so this can be lighter than
# TEXT_COLOUR: ~11:1 contrast there instead of ~4:1. Do not use the card figure on a
# light background.
CARD_TEXT_COLOUR = '#c3c2b7'

# Sizes in points; at DPI 200 one point is ~2.78 image px, and the card shows the
# image at 1/2.13 scale, so 1 pt here is ~1.3 card px. The card's own title is 60
# card px, and everything here must sit clearly below it in the hierarchy.
CARD_TITLE_FONT_SIZE = 18           # ~50 image px, ~24 card px
CARD_SUBTITLE_FONT_SIZE = 13        # ~36 image px, ~17 card px
CARD_DEPTH_FONT_SIZE = 20           # ~56 image px, ~26 card px
CARD_TICK_FONT_SIZE = 15            # ~42 image px, ~20 card px (designer's floor: 40 / 19)
CARD_AXIS_LABEL_FONT_SIZE = 16      # ~44 image px, ~21 card px

# The card puts its tagline just above this image and its tech line, centred, across
# the bottom. These strips must stay empty; make_card_figure checks them.
CARD_CLEAR_TOP_PX = 30
CARD_CLEAR_BOTTOM_PX = 90


def local_view(row, catalogue):
    """Run one KOI through the pipeline and return its local view in plot units.

    In:  row       -- pandas Series, the KOI's catalogue row
         catalogue -- DataFrame, the full KOI table (siblings are looked up here,
                      unfiltered, for the CLAUDE.md 10.18 reason)
    Out: dict -- raw_hours, raw_ppm, bin_hours, bin_ppm (ndarrays) and depth (float,
         fractional, the pipeline's own depth feature). Hours are from transit
         centre; ppm is flux change relative to baseline.
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
        'depth': depth,
    }


def true_minus(text):
    """Swap hyphens for a real minus sign, which lines up better with digits."""
    return text.replace('-', '−')


def thousands_with_true_minus(value, _position):
    """Tick text like '-60,000', with a real minus sign."""
    return true_minus(f"{value:,.0f}")


def draw_points(ax, view, colour):
    """The data layer both figures share: raw points faint, binned points on top.

    Out: (low, high) -- the y-range fitted to the data, in ppm
    """
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
    ax.set_xlim(view['raw_hours'].min(), view['raw_hours'].max())
    ax.patch.set_alpha(0)
    return low, high


def style_axes(ax, colour, tick_size, label_size):
    """Axis labels, ticks and a recessive frame -- shared by both figures."""
    ax.set_xlabel('Hours from transit centre', fontsize=label_size, color=colour)
    ax.set_ylabel('Flux change (ppm)', fontsize=label_size, color=colour)

    ax.yaxis.set_major_locator(MaxNLocator(nbins=5))   # few ticks: big labels need room
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
    ax.yaxis.set_major_formatter(FuncFormatter(thousands_with_true_minus))
    ax.xaxis.set_major_formatter(FuncFormatter(thousands_with_true_minus))
    ax.tick_params(labelsize=tick_size, colors=colour, length=0, pad=8)

    # Recessive frame: keep only the left and bottom axis lines, and one faint set
    # of horizontal gridlines so depths can be read off without a box around them.
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(colour)
        ax.spines[side].set_alpha(0.5)
    ax.grid(axis='y', color=colour, alpha=0.18, linewidth=1)
    ax.set_axisbelow(True)


def style_full_panel(ax, low, high, label):
    """Full axes, for the stand-alone 2:1 figure."""
    pad = 0.08 * (high - low)
    ax.set_ylim(low - pad, high + pad)
    style_axes(ax, TEXT_COLOUR, TICK_FONT_SIZE, AXIS_LABEL_FONT_SIZE)

    # Above the panel, not inside it: a deep dip fills the bottom of the plot and the
    # baseline runs along the top, so there is no corner the data reliably leaves
    # empty. Neutral ink; the colour of the points below carries the identity, so
    # the label is not relying on colour alone.
    ax.text(0, 1.05, label, transform=ax.transAxes, fontsize=PANEL_LABEL_FONT_SIZE,
            color=TEXT_COLOUR, ha='left', va='bottom')


def style_card_panel(ax, low, high, title, subtitle, depth):
    """Axes as in the full figure, a header above, and the depth as one number.

    The depth number stays alongside the axes because the two panels have separate
    y-scales: the dips look the same size, and the number states the difference
    without the viewer having to compare tick labels.
    """
    # Extra room under the dip for the depth label, which sits inside the axes.
    span = high - low
    ax.set_ylim(low - 0.32 * span, high + 0.06 * span)
    style_axes(ax, CARD_TEXT_COLOUR, CARD_TICK_FONT_SIZE, CARD_AXIS_LABEL_FONT_SIZE)

    # The pipeline's depth feature, not the catalogue's koi_depth: the card shows
    # what this pipeline measures. On the binary's V-shaped dip the two differ
    # (6.6% vs 7.4%) because depth is a median over the central duration/2 -- the
    # bias in CLAUDE.md 10.16 -- so the lowest bins sit visibly below this number.
    ax.text(0, low - 0.06 * span, true_minus(f"-{depth * 100:.1f}%"),
            fontsize=CARD_DEPTH_FONT_SIZE, fontweight='bold', color=CARD_TEXT_COLOUR,
            ha='center', va='top')

    # Header and sub-header stacked above the axes, positioned in points from the
    # axes' top-left corner so the gaps stay fixed whatever the font sizes are. The
    # header aligns with the y-axis line, so it lines up with the data below it.
    ax.annotate(subtitle, xy=(0, 1), xycoords='axes fraction',
                xytext=(0, 10), textcoords='offset points',
                fontsize=CARD_SUBTITLE_FONT_SIZE, color=CARD_TEXT_COLOUR, alpha=0.75,
                ha='left', va='bottom')
    ax.annotate(title, xy=(0, 1), xycoords='axes fraction',
                xytext=(0, 10 + CARD_SUBTITLE_FONT_SIZE + 6), textcoords='offset points',
                fontsize=CARD_TITLE_FONT_SIZE, fontweight='bold', color=CARD_TEXT_COLOUR,
                ha='left', va='bottom')


def content_span(fig, width_in, height_in):
    """Where everything actually drawn sits, as fractions of the canvas.

    get_tightbbox measures the box around every artist -- axes, tick labels, axis
    labels, panel text -- so this catches a long label creeping outward after a
    font-size or wording change, not just the margins set in subplots_adjust.

    Out: (left, right, bottom, top), each a fraction of the canvas
    """
    fig.canvas.draw()
    drawn = fig.get_tightbbox(fig.canvas.get_renderer())     # inches
    return (drawn.x0 / width_in, drawn.x1 / width_in,
            drawn.y0 / height_in, drawn.y1 / height_in)


def make_full_figure(views):
    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    # Fixed margins instead of bbox_inches='tight', which would crop the canvas
    # and break the exact 2560 x 1280 size.
    # The gap between panels has to hold the right panel's tick labels ("-80,000")
    # and its rotated axis label, hence the wide wspace.
    fig.subplots_adjust(left=0.14, right=0.96, bottom=0.14, top=0.88, wspace=0.42)

    for ax, panel, view in zip(axes, PANELS, views):
        low, high = draw_points(ax, view, panel['colour'])
        style_full_panel(ax, low, high, panel['label'])

    # Stop if any label or data would be cut off by LinkedIn's 1.91:1 crop.
    left, right, _, _ = content_span(fig, FIG_WIDTH_IN, FIG_HEIGHT_IN)
    print(f"full: content spans {left:.1%} to {right:.1%} of the width "
          f"(must stay within {SAFE_SIDE_FRACTION:.1%} to {1 - SAFE_SIDE_FRACTION:.1%})")
    if left < SAFE_SIDE_FRACTION or right > 1 - SAFE_SIDE_FRACTION:
        raise RuntimeError("content reaches into the strip LinkedIn crops; widen the "
                           "left/right margins in subplots_adjust")

    fig.savefig(FULL_OUTPUT_PATH, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {FULL_OUTPUT_PATH}")


def make_card_figure(views):
    fig, axes = plt.subplots(1, 2, figsize=(CARD_WIDTH_IN, CARD_HEIGHT_IN))
    # Margins hold the tick and axis labels plus the clear strips the card needs:
    # its tagline above the image and its tech line across the bottom. The gap
    # between panels holds the right panel's y tick labels and axis label.
    fig.subplots_adjust(left=0.105, right=0.99, bottom=0.215, top=0.83, wspace=0.24)

    for ax, panel, view in zip(axes, PANELS, views):
        low, high = draw_points(ax, view, panel['card_colour'])
        style_card_panel(ax, low, high, panel['card_title'], panel['card_subtitle'],
                         view['depth'])

    # Nothing may fall off the canvas, or into the strips the card's tagline and
    # tech line occupy.
    height_px = CARD_HEIGHT_IN * DPI
    left, right, bottom, top = content_span(fig, CARD_WIDTH_IN, CARD_HEIGHT_IN)
    clear_bottom_px = bottom * height_px
    clear_top_px = (1 - top) * height_px
    print(f"card: content spans x {left:.1%}-{right:.1%}; clear strips "
          f"{clear_top_px:.0f} px top (need {CARD_CLEAR_TOP_PX}), "
          f"{clear_bottom_px:.0f} px bottom (need {CARD_CLEAR_BOTTOM_PX})")
    if left < 0 or right > 1:
        raise RuntimeError("card content runs off the canvas sideways; adjust "
                           "left/right in subplots_adjust")
    if clear_top_px < CARD_CLEAR_TOP_PX or clear_bottom_px < CARD_CLEAR_BOTTOM_PX:
        raise RuntimeError("card content reaches into the tagline or tech-line strip; "
                           "adjust top/bottom in subplots_adjust")

    fig.savefig(CARD_OUTPUT_PATH, dpi=DPI, transparent=True)
    plt.close(fig)
    print(f"wrote {CARD_OUTPUT_PATH}")


def main():
    catalogue = pd.read_csv(CATALOGUE_PATH)

    views = []
    for panel in PANELS:
        row = catalogue.loc[catalogue['kepoi_name'] == panel['kepoi_name']].iloc[0]
        views.append(local_view(row, catalogue))

    make_full_figure(views)
    make_card_figure(views)


if __name__ == '__main__':
    main()
