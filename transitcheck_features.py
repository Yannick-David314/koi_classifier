"""
TransitCheck — feature extraction pipeline
==========================================

Turns a KOI catalogue row into six engineered features describing the shape of
its transit, for a gradient-boosted tree classifier.

Pipeline order per star:
    download -> clean -> flatten (transit-masked) -> fold -> bin -> measure

Feature summary:
    depth            how much the star dims during transit
    width            how long the dip lasts, measured at half-depth
    asymmetry        is the dip lopsided (ingress vs egress)
    snr              depth relative to the out-of-transit noise
    odd_even_diff    do odd- and even-numbered transits differ in depth
    secondary_depth  deepest dip anywhere outside the primary transit

Every output row also carries DIAGNOSTIC_NAMES columns. Those are measurements
*about* the run, not features — they exist so the open questions in CLAUDE.md
section 7.5 can be answered from real data rather than from argument, and they
must never enter the model matrix. See CLAUDE.md section 10 for what was
corrected against the code and why.

Usage:
    python transitcheck_features.py            # dry run: sample + precheck only, no downloads
    python transitcheck_features.py --run      # actually download and extract
"""

import os
import shutil
import sys
import time

import numpy as np
import pandas as pd
import lightkurve as lk
import astropy.units as u


# ---------------------------------------------------------------------------
# Tunable constants
#
# These were chosen deliberately; see the notes beside each one before changing.
# ---------------------------------------------------------------------------

LOCAL_N_BINS = 201          # int   — nominal bins across the local view (Shallue & Vanderburg convention)
                            #         Nominal, not guaranteed: bin() emits however many bins fit the data
                            #         actually present, so the array is not reliably length-201. Harmless
                            #         for features (each collapses to one number) but a real obstacle for
                            #         the parked CNN path, which needs fixed-length input.
LOCAL_WINDOW_DURATIONS = 4  # int   — local view spans this many transit durations, centred on the dip
DEPTH_WINDOW_FRACTION = 4   # int   — depth measured within duration/4 of centre, i.e. a total span of
                            #         duration/2 (the flat bottom only). Half-width, not full width.
SHAPE_WINDOW_FRACTION = 2   # int   — asymmetry and snr split at duration/2 from centre, i.e. a total span
                            #         of one full duration (so ingress and egress are included).
                            #         NOT used by measure_width: width deliberately scans the whole local
                            #         view so an unusually long grazing eclipse is still measured in full.
                            #         See CLAUDE.md section 5 (width).
ODD_EVEN_N_BINS = 25        # int   — coarse on purpose: halved data, and only one number is needed
SECONDARY_BINS_PER_DURATION = 4    # int — bin width for the global view = duration / this
SECONDARY_BLANK_DURATIONS = 2      # int — blank out +/- this many durations around the primary
SECONDARY_THRESHOLD = 3            # int — a dip must sit this many noise-widths below baseline
SECONDARY_MIN_RUN_BINS = 2         # int — minimum run length for the "deepest_min2" variant. A run of one
                                   #       bin has its own extreme value as its median, so it can beat a
                                   #       real secondary; at 4 bins per duration a genuine secondary spans
                                   #       roughly 4 bins, so requiring 2 kills single-bin noise without
                                   #       biasing toward wide contaminants the way longest-run does.
SECONDARY_MAX_BLANK_FRACTION = 0.8 # float — if blanking +/- SECONDARY_BLANK_DURATIONS covers this much of
                                   #       the orbit there is no "outside the primary" left to search, so
                                   #       the question cannot be asked. 217 of the 7587 CONFIRMED/FALSE
                                   #       POSITIVE rows trip this. It guards the feature, not the row:
                                   #       the other five features are degraded there, not meaningless.

OUTLIER_SIGMA_LOWER = 20    # int — lenient downward: protects real deep transits from being clipped
OUTLIER_SIGMA_UPPER = 5     # int — strict upward: removes cosmic rays and instrument glitches

FLATTEN_WINDOW_DURATIONS = 3  # int — detrending window spans this many transit durations
MIN_FLATTEN_WINDOW = 5        # int — Savitzky-Golay needs a window wider than its polynomial order

MAD_TO_SIGMA = 1.4826       # float — rescales a median-absolute-deviation onto a std-dev scale

KEPLER_LONG_CADENCE_SECONDS = 1800  # int — pin the cadence. search_lightcurve(author='Kepler') returns
                                    #       short-cadence (60 s) products alongside long-cadence ones for
                                    #       some targets, and download_all().stitch() silently mixes them:
                                    #       the same quarter appears twice at two cadences, the stitched
                                    #       time array is not monotonic, and the median diff that sizes the
                                    #       flatten window is dominated by whichever cadence has more
                                    #       points. See CLAUDE.md section 10.1.
EXPECTED_FLUX_ORIGIN = 'pdcsap_flux'  # str — what lightkurve should default to. Verified, not assumed:
                                      #       stitch() uses metadata_conflicts='silent', so post-stitch
                                      #       metadata comes from whichever product happened to be first.
SIBLING_BLANK_DURATIONS = 1.5      # float — blank +/- this many durations around each SIBLING KOI's
                                   #       transits, matching the 1.5 used for the primary. A star with
                                   #       several KOIs has its other planets' transits sitting at
                                   #       arbitrary phase once you fold on this planet's period, and
                                   #       they form runs deep enough to be misreported as a secondary.
                                   #       Measured in the 100-star run: 0 of 6 detected "secondaries"
                                   #       on multi-KOI stars sat near phase +/-0.5, against 29 of 69 on
                                   #       single-KOI stars. See CLAUDE.md 10.14.

MAST_SEARCH_DELAY_SECONDS = 1.5    # float — minimum gap between search_lightcurve calls. search hits the
                                   #       network even when the FITS files are already cached, which is
                                   #       what rate-limited pass 3 of the 100-star run (60 consecutive
                                   #       ConnectionErrors). At ~1000 stars this adds ~25 minutes and is
                                   #       cheap insurance against being throttled for hours.
MAST_MAX_ATTEMPTS = 3              # int   — attempts per star before giving up and letting resume have it
MAST_BACKOFF_SECONDS = 5.0         # float — first backoff; doubles each attempt

MIN_UNMASKED_FRACTION_FOR_DETREND = 0.25  # float — the trend fit needs out-of-transit points to fit
                                    #       TO. Masking primary +/-1.5 durations AND secondary +/-1.5
                                    #       durations covers the whole orbit once duration/period >= 1/6
                                    #       (507 of the 7587 rows); the primary alone does so at >= 1/3
                                    #       (96 rows). Past this floor flatten gets an empty array and
                                    #       raises "cannot reshape array of size 0". Hence the fallback
                                    #       chain in extract_features: secondary -> primary-only -> none.
MIN_POINTS_AFTER_CLEANING = 50      # int — sanity floor, NOT a tuned number. Below this the cadence
                                    #       estimate and flatten's internal interpolation stop being
                                    #       meaningful. Recorded as a deterministic rejection.

DEFAULT_DOWNLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mast_cache')
DEFAULT_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'features_checkpoint.csv')


# ---------------------------------------------------------------------------
# Column groups
#
# FEATURE_NAMES is the model matrix. Everything else is identifiers, ground-truth
# checks or diagnostics and must be dropped before training (CLAUDE.md section 3).
# ---------------------------------------------------------------------------

FEATURE_NAMES = (
    'depth',
    'width',
    'asymmetry',
    'snr',
    'odd_even_diff',
    'secondary_depth',
)

IDENTIFIER_NAMES = (
    'kepid',          # the star — also the grouping key for the train/test split
    'kepoi_name',     # the KOI, e.g. K00752.01 — stable unique row key for checkpointing
    'disposition',    # the label
)

# Carried through only as ground-truth checks on our own measurements. koi_depth in
# particular comes from the same vetting process that produced the labels, so it is
# leakage and must never enter X.
REFERENCE_NAMES = (
    'koi_depth_ppm',
    'koi_duration_hours',
    'koi_period_days',
    'koi_num_transits',
)

# Measurements about the run, recorded so section 7.5's open questions can be
# answered with a groupby instead of a re-download. Never features.
DIAGNOSTIC_NAMES = (
    # --- what actually came back from MAST (section 10.1) ---
    'n_products_all',          # products the search returned before the cadence filter
    'n_products_used',         # products left after pinning long cadence
    'cadences_seen',           # str, e.g. "60|1800" — quantifies how widespread the mixing was
    'flux_origin',             # str — verified, not assumed
    'n_raw_points',
    'n_duplicate_times',       # exact-duplicate timestamps dropped after the sort
    'cadence_minutes',
    'flatten_window_points',   # compare against n_raw_points: where the window exceeds the points in a
                               # segment, lightkurve silently substitutes a median for the Savitzky-Golay
                               # fit rather than raising
    # --- geometry, for the continuum in section 10.5 ---
    'duration_over_period',
    'n_koi_for_star',
    'detrend_masked_fraction',  # share of points withheld from the trend fit
    'detrend_fallback',         # str — which rung of the fallback chain fired:
                                # '' (primary+secondary+siblings) | 'secondary_only'
                                # | 'primary_only' | 'none'   (sections 10.12, 10.14)
    'n_siblings',               # other KOI rows on this same star
    'sibling_masked_fraction',  # share of points sitting inside a sibling KOI's transit
    # --- bin occupancy, to replace the estimate in section 10.6 with measurement ---
    'n_local_points',
    'n_local_bins',
    'n_local_bins_filled',
    'local_points_per_bin_median',
    # --- the three secondary variants (sections 10.2, 10.9), each with its own phase.
    #     min2 is now the FEATURE; the other two stay as diagnostics so the comparison
    #     can be re-checked at 1000 stars rather than taken on trust from 100. ---
    'secondary_noise',
    'secondary_depth_longest',                 # the old default: longest qualifying run
    'secondary_phase_fraction_longest',
    'secondary_depth_deepest',                 # deepest of any length — the fragile control
    'secondary_phase_fraction_deepest',
    'secondary_depth_deepest_min2',            # == the secondary_depth feature
    'secondary_phase_fraction_deepest_min2',
    'secondary_exclusion_fraction',            # share of global-view points dropped as sibling transits
    'secondary_guard',         # str — why the secondary search was skipped, when it was
)

ERROR_NAMES = (
    'error',        # str — "TypeName: message", empty on success
    'error_kind',   # str — '' | 'precheck' | 'nodata' | 'transient' | 'unknown'
)

OUTPUT_COLUMNS = (
    IDENTIFIER_NAMES + REFERENCE_NAMES + FEATURE_NAMES + DIAGNOSTIC_NAMES + ERROR_NAMES
)

TEXT_DIAGNOSTICS = frozenset({'cadences_seen', 'flux_origin', 'secondary_guard',
                              'detrend_fallback'})


# ---------------------------------------------------------------------------
# Errors and their classification
#
# The broad catch in process_koi is deliberate and TEMPORARY: at ~1000 stars we
# do not yet know what breaks, so this finds out empirically and the catch can be
# narrowed once the first full run reports. Classifying the result is what makes
# resume useful — a MAST timeout should be retried on the next run, a target with
# no data at this cadence should not.
# ---------------------------------------------------------------------------

class NoDataError(RuntimeError):
    """No usable light curve exists for this target at the pinned cadence."""


# Substrings that mark a failure as worth retrying. Matched against the message
# because the exception types raised through astroquery/requests/urllib vary.
TRANSIENT_MESSAGE_MARKERS = (
    'timed out', 'timeout', 'connection', 'broken pipe', 'max retries',
    'temporarily unavailable', 'temporary failure', 'remote end closed',
    'incompleteread', 'service unavailable', 'bad gateway', 'gateway timeout',
    'ssl', 'eof occurred', ' 429', ' 500', ' 502', ' 503', ' 504',
)

TRANSIENT_EXCEPTION_NAMES = frozenset({
    'TimeoutError', 'ConnectionError', 'ConnectionResetError', 'ConnectionAbortedError',
    'BrokenPipeError', 'ChunkedEncodingError', 'ReadTimeout', 'ConnectTimeout',
    'RemoteDisconnected', 'IncompleteRead', 'SSLError', 'ProtocolError',
    'HTTPError', 'URLError', 'timeout',
})


def classify_error(exc):
    """Decide whether a failure is worth retrying on the next run.

    In:  exc — the caught exception
    Out: str — 'nodata'      deterministic, never retry
               'transient'   network or MAST hiccup, retry
               'diskfull'    the download cache ran out of room; retry once space
                             exists. Its own kind rather than 'transient' because
                             the fix is on this machine, not at MAST.
               'corruptcache' a cached FITS file is truncated or unreadable, which
                             is what a disk filling mid-download leaves behind.
                             Retrying only helps after the bad file is evicted, so
                             process_koi evicts it before giving up.
               'unknown'     not retried by default, because a bug that fails the
                             same way every time would otherwise loop forever.

    The 1000-star run produced 37 'unknown' failures that were all one cause: the
    disk filled at row ~1100, 27 rows raised ENOSPC directly, and the next 10 read
    back the half-written FITS. Both now classify, so the next run says what
    happened instead of burying it in 'unknown'.
    """
    name = type(exc).__name__
    message = str(exc).lower()

    if 'no space left on device' in message or getattr(exc, 'errno', None) == 28:
        return 'diskfull'
    # Two distinct signatures for the same thing, both seen in the 1000-star run:
    # a file truncated badly enough to lose its type, and one that still looks like
    # a FITS but will not parse. A connection reset mid-download produces either.
    if ('not recognized as a supported data product' in message
            or 'error in reading data product' in message):
        return 'corruptcache'
    if name in TRANSIENT_EXCEPTION_NAMES:
        return 'transient'
    if any(marker in message for marker in TRANSIENT_MESSAGE_MARKERS):
        return 'transient'
    if isinstance(exc, NoDataError) or name == 'SearchError':
        return 'nodata'
    return 'unknown'


def evict_star_cache(kepid, download_dir):
    """Delete one star's cached FITS files so the next attempt re-downloads them.

    Only ever called on a star whose cache was found unreadable. The cache is
    regenerable by definition, so this is safe; leaving a truncated file in place
    is not, because lightkurve will keep reading it back forever.

    In:  kepid        — int
         download_dir — str or None
    Out: int, number of directories removed
    """
    if not download_dir:
        return 0

    kepler_dir = os.path.join(download_dir, 'mastDownload', 'Kepler')
    if not os.path.isdir(kepler_dir):
        return 0

    prefix = f"kplr{int(kepid):09d}_"
    removed = 0
    for entry in os.listdir(kepler_dir):
        if entry.startswith(prefix):
            shutil.rmtree(os.path.join(kepler_dir, entry), ignore_errors=True)
            removed += 1
    return removed


def precheck_koi(row):
    """Reject rows that cannot possibly work, before spending a download on them.

    Only catalogue-level impossibilities belong here. A condition that breaks one
    feature but leaves the others usable is guarded inside that feature instead —
    see measure_secondary_depth and measure_odd_even_diff.

    In:  row — pandas Series, one KOI catalogue row
    Out: str reason, or None if the row should be attempted
    """
    period = row.get('koi_period', np.nan)
    epoch_time = row.get('koi_time0bk', np.nan)
    duration = row.get('koi_duration', np.nan)
    num_transits = row.get('koi_num_transits', np.nan)

    if pd.isna(period) or period <= 0:
        return 'koi_period missing or non-positive'
    # A NaN epoch produces a garbage fold rather than an exception, so it has to be
    # caught here rather than relying on the try/except. In the current cumulative
    # table this never fires — period, epoch and duration are all complete — but it
    # costs nothing and the failure it prevents would be silent.
    if pd.isna(epoch_time):
        return 'koi_time0bk missing'
    if pd.isna(duration) or duration <= 0:
        return 'koi_duration missing or non-positive'
    # 208 rows in the CONFIRMED/FALSE POSITIVE subset. Nothing was observed, so there
    # is no transit to measure. NaN passes through: unknown is not the same as zero.
    if not pd.isna(num_transits) and num_transits < 1:
        return 'koi_num_transits is 0'

    return None


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def find_runs(mask):
    """Find every unbroken run of True values in a boolean array.

    In:  mask — ndarray of bool
    Out: (starts, ends) as ndarrays of int. starts inclusive, ends exclusive, so
         mask[starts[i]:ends[i]] is exactly the i-th run.
    """
    # Padding with False at both ends guarantees every run has an opening +1 and a
    # closing -1. The leading pad shifts indices right by one, np.diff shifts them
    # back left by one, so the returned indices land correctly on the original mask.
    padded = np.concatenate(([False], np.asarray(mask, dtype=bool), [False]))
    changes = np.diff(padded.astype(int))               # ndarray of int, len(mask) + 1

    starts = np.where(changes == 1)[0]    # ndarray of int — index where each run begins
    ends = np.where(changes == -1)[0]     # ndarray of int — index just past where each run ends
    return starts, ends


def find_longest_true(mask):
    """Find the longest unbroken run of True values in a boolean array.

    A real dip is contiguous; a noise spike is a single isolated bin. Taking the
    longest run therefore ignores strays without needing to restrict where we look.

    In:  mask  — ndarray of bool
    Out: (start, end) as ints, where start is inclusive and end is exclusive,
         so mask[start:end] is exactly the run. Returns (None, None) if no True.
    """
    starts, ends = find_runs(mask)

    if len(starts) == 0:
        return None, None

    lengths = ends - starts               # ndarray of int — length of each run
    longest = int(np.argmax(lengths))     # int — position of the winning run
    return int(starts[longest]), int(ends[longest])


def robust_noise(values):
    """Estimate the spread of an array, ignoring outliers.

    Uses the median absolute deviation rather than a standard deviation: one bad
    value, squared, dominates a std-dev, whereas a MAD barely moves. Scaling by
    1.4826 makes the result directly comparable to a std-dev for clean data.

    In:  values — ndarray of float (may contain NaN)
    Out: float. Returns np.nan if the spread can't be measured.
    """
    if len(values) == 0 or not np.any(np.isfinite(values)):
        return np.nan

    centre = np.nanmedian(values)                              # float
    spread = np.nanmedian(np.abs(values - centre))             # float
    noise = spread * MAD_TO_SIGMA                              # float

    if noise == 0 or np.isnan(noise):
        return np.nan
    return noise


def circular_mean_phase_fraction(phase_slice, period):
    """Mean phase of a set of bins, in units of period, respecting the wrap.

    A plain mean is wrong for any run that straddles the +/-period/2 boundary:
    phase values of [+2.45, +2.48, -2.48, -2.45] days average to ~0 when the run
    actually sits at +/-0.5. Averaging the angles instead gives the right answer
    there and the same answer everywhere else.

    In:  phase_slice — ndarray of float, days from transit centre
         period      — float, days
    Out: float in [-0.5, 0.5]
    """
    angles = 2 * np.pi * np.asarray(phase_slice, dtype=float) / period
    mean_angle = np.arctan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
    return mean_angle / (2 * np.pi)


def nan_features():
    """A full set of features, all NaN, so a failed row has the same columns.

    Out: dict of feature name -> np.nan
    """
    return {name: np.nan for name in FEATURE_NAMES}


def blank_diagnostics():
    """A full set of diagnostics, blank, for rows that never got far enough to measure.

    Out: dict of diagnostic name -> np.nan (or '' for the text ones)
    """
    return {
        name: ('' if name in TEXT_DIAGNOSTICS else np.nan)
        for name in DIAGNOSTIC_NAMES
    }


# ---------------------------------------------------------------------------
# Stage 1 — download and clean
# ---------------------------------------------------------------------------

_last_search_time = [0.0]   # module-level so the throttle spans every call in a run


def _search_with_throttle(kepid, max_attempts=MAST_MAX_ATTEMPTS):
    """Query MAST for one star, rate-limited and with bounded retry.

    search_lightcurve hits the network even when the FITS files are already cached,
    so the download cache does NOT protect against rate limiting. Pass 3 of the
    100-star run hit 60 consecutive ConnectionErrors for exactly this reason.

    The delay is a floor between calls, not a sleep after each one, so a slow star
    costs nothing extra. The retry is bounded: anything still failing after
    max_attempts is left to the resume path, which is what a multi-hour run needs so
    a single network blip does not require babysitting.

    In:  kepid        — int
         max_attempts — int
    Out: SearchResult
    """
    delay = MAST_SEARCH_DELAY_SECONDS
    backoff = MAST_BACKOFF_SECONDS

    for attempt in range(1, max_attempts + 1):
        waited = time.monotonic() - _last_search_time[0]
        if waited < delay:
            time.sleep(delay - waited)

        try:
            result = lk.search_lightcurve(f"KIC {int(kepid)}", author='Kepler')
            _last_search_time[0] = time.monotonic()
            return result
        except Exception as e:                                   # noqa: BLE001
            _last_search_time[0] = time.monotonic()
            if attempt == max_attempts or classify_error(e) != 'transient':
                raise
            print(f"  [retry {attempt}/{max_attempts - 1}] KIC {int(kepid)}: "
                  f"{type(e).__name__}; backing off {backoff:.0f}s")
            time.sleep(backoff)
            backoff *= 2


def load_star(kepid, download_dir=None, exptime=KEPLER_LONG_CADENCE_SECONDS):
    """Download every available Kepler quarter for one star at one cadence, and clean it.

    The cadence is pinned rather than taking whatever the search returns. Mixing
    60 s and 1800 s products means the same quarter arrives twice, the stitched
    time array runs backwards at the seam, and the median time difference that
    sizes the flatten window reflects whichever cadence contributed more points.
    See CLAUDE.md section 10.1.

    In:  kepid        — int, the Kepler Input Catalog ID
         download_dir — str or None, where lightkurve caches the FITS files
         exptime      — int, seconds; 1800 is Kepler long cadence
    Out: (LightCurve, dict of diagnostics)
    Raises: NoDataError when nothing usable exists at this cadence
    """
    search_result = _search_with_throttle(kepid)

    n_products_all = len(search_result)
    if n_products_all == 0:
        raise NoDataError(f"search returned no Kepler products for KIC {int(kepid)}")

    # One search, filtered locally: this records what cadences exist for the target
    # (the diagnostic that quantifies how widespread the mixing was) without paying
    # for a second round trip to MAST.
    exptimes = np.asarray(search_result.table['exptime'], dtype=float)
    cadences_seen = '|'.join(str(int(round(v))) for v in sorted(set(exptimes)))

    keep = np.isclose(exptimes, float(exptime))
    if not keep.any():
        raise NoDataError(
            f"no {int(exptime)}s products for KIC {int(kepid)}; available: {cadences_seen}"
        )

    collection = search_result[keep].download_all(download_dir=download_dir)
    if collection is None or len(collection) == 0:
        raise NoDataError(f"download_all returned nothing for KIC {int(kepid)}")

    star_lc = collection.stitch()                               # LightCurve

    # Verified rather than trusted: stitch() uses metadata_conflicts='silent', so
    # this reflects whichever product came first in the collection.
    flux_origin = str(star_lc.meta.get('FLUX_ORIGIN', ''))
    if flux_origin.lower() != EXPECTED_FLUX_ORIGIN:
        print(f"  [warn] KIC {int(kepid)}: FLUX_ORIGIN is {flux_origin!r}, "
              f"expected {EXPECTED_FLUX_ORIGIN!r}")

    # stitch() is a plain vstack in collection order — it does not sort. Everything
    # downstream that walks the array in index order (the cadence estimate, and
    # Savitzky-Golay inside flatten) needs monotonic time. Insurance even with the
    # cadence pinned, because collection order is the search table's order.
    star_lc = star_lc[np.argsort(star_lc.time.value)]

    # Exact duplicate timestamps would put a zero into the diff used for the cadence
    # estimate, and a zero median cadence divides through the flatten window.
    times = star_lc.time.value
    keep_unique = np.concatenate(([True], np.diff(times) > 0))
    n_duplicate_times = int((~keep_unique).sum())
    if n_duplicate_times:
        star_lc = star_lc[keep_unique]

    # Asymmetric clipping: aggressive on upward spikes, forgiving downward so a
    # genuinely deep eclipse isn't mistaken for a glitch and thrown away.
    star_lc = star_lc.remove_outliers(
        sigma_lower=OUTLIER_SIGMA_LOWER,
        sigma_upper=OUTLIER_SIGMA_UPPER,
    )

    if len(star_lc) < MIN_POINTS_AFTER_CLEANING:
        raise NoDataError(
            f"only {len(star_lc)} points left after cleaning KIC {int(kepid)} "
            f"(floor is {MIN_POINTS_AFTER_CLEANING})"
        )

    info = {
        'n_products_all': n_products_all,
        'n_products_used': int(keep.sum()),
        'cadences_seen': cadences_seen,
        'flux_origin': flux_origin,
        'n_raw_points': len(star_lc),
        'n_duplicate_times': n_duplicate_times,
    }
    return star_lc, info


class StarCache:
    """Hold one star's cleaned light curve so repeat KOI rows don't re-download it.

    Multi-planet systems put the same kepid on several KOI rows (651 stars in the
    CONFIRMED/FALSE POSITIVE subset carry 2-7 rows each). build_feature_table sorts
    its batch by kepid, which makes those rows adjacent — so a cache of exactly one
    star captures all the reuse a full LRU would, with bounded memory and nothing to
    invalidate when a feature definition changes.

    Nothing downstream mutates the cached light curve: remove_outliers, flatten and
    fold all return new objects.
    """

    def __init__(self, download_dir=None):
        self.download_dir = download_dir
        self._kepid = None
        self._payload = None
        self.hits = 0
        self.misses = 0

    def get(self, kepid):
        """In: kepid — int.  Out: (LightCurve, dict of diagnostics)"""
        kepid = int(kepid)
        if self._kepid == kepid:
            self.hits += 1
            return self._payload

        self.misses += 1
        # Release the previous star BEFORE fetching the next. Without this, the old
        # and new stitched multi-quarter light curves are both alive for the duration
        # of load_star, so peak memory is two curves rather than one.
        #
        # Memory only: this cannot change any measured value. A cache miss recomputes
        # the star from scratch either way, so nothing downstream sees a difference.
        # The one behavioural nuance is that a failed load now leaves the cache empty
        # instead of holding the previous star -- which is the safer of the two, and
        # cannot matter here because rows are sorted by kepid.
        self._kepid = None
        self._payload = None
        self._payload = load_star(kepid, download_dir=self.download_dir)
        self._kepid = kepid
        return self._payload


# ---------------------------------------------------------------------------
# Stage 2 — flatten, with the transit masked out of the trend fit
# ---------------------------------------------------------------------------

def build_transit_mask(star_lc, period, epoch_time, duration, include_secondary=False):
    """Flag which raw data points fall inside a transit.

    Note this predicts transit times arithmetically from the catalogue's period
    and epoch — it never inspects the flux. Folding is used only as a convenient
    way to compute "distance from the nearest transit centre" for every point at
    once, and time_original translates the selection back to real timestamps.

    Two different masks are needed downstream and they are NOT interchangeable:

      include_secondary=False  the primary only. This is what measure_odd_even_diff
                               wants — folding secondary-eclipse points into an
                               odd-versus-even transit depth comparison would
                               corrupt it.
      include_secondary=True   primary plus the expected secondary region. This is
                               what flatten() wants: any dip the trend fit can see
                               is a dip it partly divides out (CLAUDE.md 4.3, 10.10).

    LIMITATION: the secondary region is assumed to sit at phase +/-0.5, which is
    true for a circular orbit and wrong for an eccentric one, where the secondary
    shifts off centre. So this is a partial fix. secondary_depth is still recorded
    as measured, and secondary_phase_fraction reports where the dip actually was,
    which is what tells us how often the assumption holds.

    In:  star_lc     — LightCurve (unfolded)
         period      — float, days
         epoch_time  — float, BKJD
         duration    — float, hours
         include_secondary — bool, also mask phase +/-0.5
    Out: ndarray of bool, aligned index-for-index with star_lc
    """
    # normalize_phase=True makes .phase a fraction of the period (-0.5 to 0.5)
    # rather than days, so it can be compared against fractional_duration directly.
    temp_fold = star_lc.fold(period=period, epoch_time=epoch_time, normalize_phase=True)
    folded_phase = temp_fold.phase.value

    fractional_duration = (duration / 24) / period              # float — transit length as a fraction of one orbit
    phase_mask = np.abs(folded_phase) < (fractional_duration * 1.5)   # ndarray of bool, in folded order

    if include_secondary:
        # abs(abs(phase) - 0.5) measures distance from the wrap point, so this picks up
        # the secondary from both ends of the phase axis at once.
        phase_mask = phase_mask | (
            np.abs(np.abs(folded_phase) - 0.5) < (fractional_duration * 1.5)
        )

    # np.isin is a membership test, so it doesn't care that folding reordered the
    # rows — the result comes back in star_lc's own order and length.
    return np.isin(star_lc.time.value, temp_fold.time_original.value[phase_mask])


def build_sibling_mask(star_lc, siblings, blank_durations=SIBLING_BLANK_DURATIONS):
    """Flag points sitting inside any OTHER KOI's transit on the same star.

    A multi-planet system puts several KOI rows on one star. Folding on this KOI's
    period scatters the siblings' transits across arbitrary phase, where they form
    contiguous runs deep enough to cross the 3-sigma threshold and be reported as a
    secondary eclipse. The 100-star run measured the signature: 0 of 6 detected
    "secondaries" on multi-KOI stars sat near phase +/-0.5, against 29 of 69 on
    single-KOI stars. The siblings' own period and epoch are in the catalogue, so
    this is predictable arithmetic, not a caveat.

    Phase is computed directly by modulo rather than via fold() + time_original.
    That is the same arithmetic fold() does — ((t - epoch + P/2) mod P) - P/2 — but
    it needs no round-trip back to real time, because we never leave time order.

    In:  star_lc        — LightCurve (unfolded, time-sorted)
         siblings       — iterable of (period days, epoch BKJD, duration hours)
         blank_durations — float, half-width of the blanked region in durations
    Out: ndarray of bool aligned with star_lc. All False when there are no siblings.
    """
    mask = np.zeros(len(star_lc), dtype=bool)
    times = star_lc.time.value

    for period, epoch_time, duration in siblings or ():
        if not np.isfinite([period, epoch_time, duration]).all() or period <= 0 or duration <= 0:
            continue
        phase = ((times - epoch_time + period / 2) % period) - period / 2   # ndarray of float, days
        mask |= np.abs(phase) < (blank_durations * (duration / 24))

    return mask


def flatten_star(star_lc, transit_mask, duration):
    """Remove slow brightness drift without eating into the transit itself.

    The detrending window is sized in data points, so it has to be derived from
    this star's own cadence: too narrow and the fit follows the dip, too wide and
    it can't track the star's real variability.

    In:  star_lc      — LightCurve (unfolded, time-sorted)
         transit_mask — ndarray of bool, or None to skip detrending entirely (the
                        last rung of the fallback chain: a curve where the transit
                        covers so much of the orbit that no baseline is left to fit
                        a trend to. stitch() has already normalised the flux, so
                        this is a degradation rather than a catastrophe.)
         duration     — float, hours
    Out: (LightCurve flattened, dict with cadence_minutes and flatten_window_points)
    Raises: NoDataError if the cadence can't be measured
    """
    cadence_minutes = float(np.median(np.diff(star_lc.time.value)) * 24 * 60)   # float — gap between consecutive points

    if not np.isfinite(cadence_minutes) or cadence_minutes <= 0:
        raise NoDataError(f"cadence estimate came out as {cadence_minutes}")

    if transit_mask is None:
        return star_lc, {'cadence_minutes': cadence_minutes,
                         'flatten_window_points': np.nan}

    window_length = int((FLATTEN_WINDOW_DURATIONS * 60 * duration) / cadence_minutes)   # int — window in data points
    window_length = max(window_length, MIN_FLATTEN_WINDOW)
    window_length += 1 if window_length % 2 == 0 else 0    # Savitzky-Golay windows must be odd

    # mask= tells the trend fit to ignore in-transit points, so the fit has no
    # reason to dip toward the transit (or overshoot recovering from it).
    #
    # Note lightkurve splits the curve into segments at large time gaps and, where a
    # segment is shorter than window_length, substitutes the segment median for the
    # Savitzky-Golay fit rather than raising. So an oversized window degrades to "no
    # detrending" silently — which is why flatten_window_points is recorded.
    flat_lc = star_lc.flatten(window_length=window_length, mask=transit_mask)

    return flat_lc, {'cadence_minutes': cadence_minutes, 'flatten_window_points': window_length}


# ---------------------------------------------------------------------------
# Stage 3 — fold, slice to the transit window, bin
# ---------------------------------------------------------------------------

def prepare_transit(lc, period, epoch_time, duration,
                    n_bins=LOCAL_N_BINS,
                    window_durations=LOCAL_WINDOW_DURATIONS,
                    return_info=False):
    """Fold, cut out the region around the transit, bin it, and measure depth.

    Everything downstream works from these outputs rather than recomputing the
    fold and bin, so all the shape features describe the same curve.

    In:  lc               — LightCurve (flattened, unfolded)
         period           — float, days
         epoch_time       — float, BKJD
         duration         — float, hours
         n_bins           — int, bins across the whole window
         window_durations — int, window width in transit durations
         return_info      — bool, append a fifth element of bin-occupancy diagnostics
    Out: (depth, flux, phase, baseline)  — or the same plus info when return_info
         depth    — float, fractional dip below baseline. NaN if unmeasurable.
         flux     — ndarray of float, one value per bin
         phase    — ndarray of float, days from transit centre, one per bin
         baseline — float, the out-of-transit flux level
    """
    info = {
        'n_local_points': 0,
        'n_local_bins': 0,
        'n_local_bins_filled': 0,
        'local_points_per_bin_median': np.nan,
    }

    folded = lc.fold(period=period, epoch_time=epoch_time)      # FoldedLightCurve, phase in days

    half_window_days = (duration * window_durations / 2) / 24   # float — half the window, hours -> days
    window_mask = np.abs(folded.phase.value) < half_window_days  # ndarray of bool
    local_view = folded[window_mask]                             # FoldedLightCurve, just the transit region

    if len(local_view) == 0:
        empty = np.array([], dtype=float)
        result = (np.nan, empty, empty, np.nan)
        return result + (info,) if return_info else result

    bin_width_hours = duration * window_durations / n_bins       # float
    binned = local_view.bin(time_bin_size=bin_width_hours * u.hour)   # FoldedLightCurve, one row per bin

    # np.asarray is load-bearing, not tidying. binned.flux.value is an astropy
    # MaskedNDArray, and np.nanmedian over a fully-masked slice returns a *masked*
    # value that float() then coerces to 0.0 rather than NaN. That turned "every
    # in-transit bin is empty" into depth = baseline - 0 = 1.0, a reported 100%
    # dimming, on 3 of 104 rows in the first 100-star run.
    #
    # Precisely: asarray strips the MASK and keeps the data underneath. It works here
    # because bin() writes NaN under the mask for empty bins — it is not a general
    # masked-to-NaN conversion. The isfinite guards below carry the real weight.
    # See CLAUDE.md section 10.13.
    flux = np.asarray(binned.flux.value, dtype=float)     # ndarray of float
    phase = np.asarray(binned.phase.value, dtype=float)   # ndarray of float, days

    if return_info:
        info['n_local_points'] = len(local_view)
        info['n_local_bins'] = len(flux)
        info['n_local_bins_filled'] = int(np.isfinite(flux).sum())
        if len(phase) > 0:
            # Bins are uniform and contiguous for a fixed time_bin_size, so the edges
            # follow from the centres. This measures points per bin directly instead of
            # estimating it from koi_num_transits — see CLAUDE.md section 10.6.
            half_bin_days = (bin_width_hours / 24) / 2
            edges = np.append(phase - half_bin_days, phase[-1] + half_bin_days)
            counts, _ = np.histogram(local_view.phase.value, bins=edges)
            info['local_points_per_bin_median'] = float(np.median(counts))

    # duration/4 keeps to the flat bottom of the dip. Wider drags the median up
    # toward baseline (the sloped edges) and undermeasures the depth.
    in_transit = np.abs(phase) < (duration / 24) / DEPTH_WINDOW_FRACTION   # ndarray of bool

    # The test is whether either region contains DATA, not merely whether bins fall
    # in it. A long-period KOI can put 25 bins inside the transit window and have
    # every one of them empty — the window exists, the measurement does not.
    if not np.isfinite(flux[in_transit]).any() or not np.isfinite(flux[~in_transit]).any():
        result = (np.nan, flux, phase, np.nan)
        return result + (info,) if return_info else result

    baseline = float(np.nanmedian(flux[~in_transit]))            # float
    depth = baseline - float(np.nanmedian(flux[in_transit]))     # float

    result = (depth, flux, phase, baseline)
    return result + (info,) if return_info else result


# ---------------------------------------------------------------------------
# Stage 4 — the six features
# ---------------------------------------------------------------------------

def measure_width(flux, phase, baseline, depth):
    """How long the dip lasts, measured where it crosses half its depth.

    Uses the longest contiguous run rather than counting all below-threshold bins
    (a stray inflates the count) or spanning first-to-last (a stray at the edge
    stretches it enormously). Deliberately unrestricted in phase, so an unusually
    long grazing eclipse is still measured in full.

    In:  flux, phase — ndarrays of float from prepare_transit
         baseline, depth — floats from prepare_transit
    Out: float, hours. np.nan if nothing crosses the threshold.
    """
    if not np.isfinite(depth) or not np.isfinite(baseline) or len(flux) == 0:
        return np.nan

    # Note the threshold inverts if depth is negative (no dip found, flux sitting
    # above baseline): "below half depth" then selects most of the window and the
    # longest run becomes meaningless. Left as-is because changing it would move the
    # validated numbers; watch for it in the scaled run's depth distribution.
    below_half_depth = flux < (baseline - depth / 2)             # ndarray of bool
    start, end = find_longest_true(below_half_depth)

    if start is None:
        return np.nan

    # end is exclusive, so the last bin in the run is at end - 1.
    return float((phase[end - 1] - phase[start]) * 24)           # float, days -> hours


def measure_asymmetry(flux, phase, baseline, duration):
    """Is the dip deeper on the way in than on the way out?

    A real planet transit is symmetric, since the geometry going in mirrors the
    geometry coming out. Signed rather than absolute: a tree can ignore the sign
    by splitting at zero, but can't recover it if it's been thrown away.

    In:  flux, phase — ndarrays of float from prepare_transit
         baseline    — float from prepare_transit
         duration    — float, hours
    Out: float. Positive means ingress (left) is deeper than egress (right).
    """
    if not np.isfinite(baseline) or len(flux) == 0:
        return np.nan

    # duration/2 here, not /4 — asymmetry lives in the sloped edges, which the
    # narrower depth window deliberately excludes.
    in_transit = np.abs(phase) < (duration / 24) / SHAPE_WINDOW_FRACTION   # ndarray of bool

    ingress = in_transit & (phase < 0)                           # ndarray of bool
    egress = in_transit & (phase > 0)                            # ndarray of bool

    if not ingress.any() or not egress.any():
        return np.nan

    ingress_depth = baseline - float(np.nanmedian(flux[ingress]))   # float
    egress_depth = baseline - float(np.nanmedian(flux[egress]))     # float

    return ingress_depth - egress_depth


def measure_snr(flux, phase, depth, duration):
    """Depth relative to how much the flux wobbles when nothing is happening.

    A 200 ppm dip on a quiet star is convincing; the same dip on a noisy star is
    meaningless. Depth alone can't tell those apart.

    In:  flux, phase — ndarrays of float from prepare_transit
         depth       — float from prepare_transit
         duration    — float, hours
    Out: float. np.nan if the noise level can't be measured.
    """
    if not np.isfinite(depth) or len(flux) == 0:
        return np.nan

    # duration/2 so the "noise" sample sits clear of the ingress/egress slopes,
    # which are real signal and would inflate the estimate.
    out_of_transit = np.abs(phase) > (duration / 24) / SHAPE_WINDOW_FRACTION   # ndarray of bool

    if not out_of_transit.any():
        return np.nan

    noise = robust_noise(flux[out_of_transit])                   # float

    if np.isnan(noise):
        return np.nan
    return depth / noise


def measure_odd_even_diff(flat_star_lc, transit_mask, period, epoch_time, duration,
                          n_bins=ODD_EVEN_N_BINS):
    """Compare the depth of odd-numbered transits against even-numbered ones.

    A real planet's transits are identical whichever one you look at. A blended
    eclipsing binary often alternates between a deeper and shallower dip.

    Splitting has to happen before folding: folding stacks every transit onto one
    shared phase axis, which erases which transit a point came from.

    In:  flat_star_lc — LightCurve (flattened, unfolded)
         transit_mask — ndarray of bool, aligned with flat_star_lc
         period, epoch_time, duration — floats
         n_bins       — int, coarse since each group holds only half the data
    Out: float, absolute difference between the two depths. np.nan when a star has
         too few transits to fill both parities, so one bad sub-feature NaNs this
         column rather than the whole row.
    """
    # Dividing elapsed time by the period gives two things at once: the whole
    # number is which transit, the fractional leftover is the phase. Here we want
    # the whole number.
    transit_number = np.round(
        (flat_star_lc.time.value - epoch_time) / period
    ).astype(int)                                                # ndarray of int, one per data point

    # Both conditions must hold: in-transit at all, and the right parity.
    even_filter = transit_mask & (transit_number % 2 == 0)       # ndarray of bool
    odd_filter = transit_mask & (transit_number % 2 == 1)        # ndarray of bool

    # A star observed for one or two transits can leave a parity empty. Folding and
    # binning an empty light curve is not a meaningful question.
    if not even_filter.any() or not odd_filter.any():
        return np.nan

    try:
        even_depth, _, _, _ = prepare_transit(
            flat_star_lc[even_filter], period, epoch_time, duration, n_bins=n_bins
        )
        odd_depth, _, _, _ = prepare_transit(
            flat_star_lc[odd_filter], period, epoch_time, duration, n_bins=n_bins
        )
    except Exception:
        # Per-feature guard, same intent as the emptiness check above: a sparse
        # parity subset that trips something inside fold/bin should cost this one
        # column, not the other five features for this star.
        return np.nan

    return abs(even_depth - odd_depth)


def _best_run_by_depth(flux, phase, baseline, period, starts, ends, min_bins=1):
    """Pick the run whose median flux sits deepest below baseline.

    In:  flux, phase   — ndarrays of float, the binned global view
         baseline      — float
         period        — float, days
         starts, ends  — ndarrays of int from find_runs
         min_bins      — int, ignore runs shorter than this
    Out: (depth, phase_fraction). (0.0, np.nan) when no run qualifies — "no
         secondary" is a real answer, not a failure.
    """
    best_depth = None
    best_slice = None

    for start, end in zip(starts, ends):
        if (end - start) < min_bins:
            continue
        candidate = baseline - float(np.nanmedian(flux[start:end]))
        if not np.isfinite(candidate):
            continue
        if best_depth is None or candidate > best_depth:
            best_depth = candidate
            best_slice = (start, end)

    if best_depth is None:
        return 0.0, np.nan

    start, end = best_slice
    return best_depth, circular_mean_phase_fraction(phase[start:end], period)


def measure_secondary_depth(lc, period, epoch_time, duration,
                            bins_per_duration=SECONDARY_BINS_PER_DURATION,
                            threshold=SECONDARY_THRESHOLD,
                            exclude_mask=None):
    """Find a dip elsewhere in the orbit, other than the primary transit.

    An eclipsing binary eclipses twice per orbit; a planet effectively once. A
    detectable secondary is therefore strong evidence of two stars.

    This needs the whole folded period, not the local view — a secondary sits
    roughly half a period away, far outside the local window. Eccentric orbits
    shift it off-centre, so we search everywhere rather than checking phase 0.5.

    Three selection rules are reported side by side. The 100-star run settled which
    one to feed the model — `secondary_depth_deepest_min2` — but all three stay
    recorded so the comparison can be re-checked at 1000 stars instead of resting on
    38 detections (CLAUDE.md 10.2, 10.15):

        secondary_depth_longest       longest qualifying run (the old default)
        secondary_depth_deepest       deepest run of any length — the fragile control
        secondary_depth_deepest_min2  deepest run of >= SECONDARY_MIN_RUN_BINS  <-- FEATURE

    Measured at 100 stars: longest-run fired on 66% of CONFIRMED against 83% of
    FALSE POSITIVE (1.26x), i.e. mostly finding noise. min2 fired on 23% against 48%
    (2.08x) and on dips 2.5x deeper.

    A phase fraction near +/-0.5 means a genuine secondary; an arbitrary value
    means the run is contamination — another planet in the system, or stellar
    variability that happened to cross the threshold.

    Two measured effects, both now fixed (CLAUDE.md 10.9 and 10.10):

    1. A secondary near phase +/-0.5 used to straddle the ends of the phase axis
       and arrive as TWO runs of half the length. np.roll below moves the wrap
       point to the array centre so it arrives as one run.
    2. flatten() used to mask only the PRIMARY, so the trend fit partly followed
       the secondary and divided it out. build_transit_mask(include_secondary=True)
       now covers the expected secondary region as well — see the limitation noted
       there for eccentric orbits.

    In:  lc — LightCurve (flattened, unfolded)
         period, epoch_time, duration — floats
         bins_per_duration — int, sets bin width so a secondary spans several bins
         threshold         — int, how many noise-widths below baseline counts as a dip
    Out: dict — depths are fractional; np.nan means the question could not be asked,
         0.0 means it was asked and the answer was no.
    """
    blank = {
        'secondary_depth_longest': np.nan,
        'secondary_phase_fraction_longest': np.nan,
        'secondary_depth_deepest': np.nan,
        'secondary_phase_fraction_deepest': np.nan,
        'secondary_depth_deepest_min2': np.nan,
        'secondary_phase_fraction_deepest_min2': np.nan,
        'secondary_noise': np.nan,
        'secondary_exclusion_fraction': 0.0,
        'secondary_guard': '',
    }

    # Drop sibling-KOI transits before folding. Removing the points entirely is
    # simpler and safer than blanking a phase region, because a sibling on a
    # different period does not land at a fixed phase here — it smears.
    if exclude_mask is not None and np.any(exclude_mask):
        blank['secondary_exclusion_fraction'] = float(np.mean(exclude_mask))
        lc = lc[~np.asarray(exclude_mask, dtype=bool)]
        if len(lc) < MIN_POINTS_AFTER_CLEANING:
            blank['secondary_guard'] = 'too few points left after excluding sibling transits'
            return blank

    # Blanking +/- SECONDARY_BLANK_DURATIONS around phase 0 removes this much of the
    # orbit. Past SECONDARY_MAX_BLANK_FRACTION there is nothing left to search, and
    # the old code returned 0.0 here — which reads as "no secondary found", a
    # confident answer to a question that was never asked. NaN is the honest value,
    # and HistGradientBoostingClassifier consumes it natively.
    blanked_fraction = (2 * SECONDARY_BLANK_DURATIONS * (duration / 24)) / period
    if blanked_fraction >= SECONDARY_MAX_BLANK_FRACTION:
        blank['secondary_guard'] = f'blanking covers {blanked_fraction:.2f} of the orbit'
        return blank

    folded = lc.fold(period=period, epoch_time=epoch_time)       # FoldedLightCurve

    # Bin width comes from the physics, not a fixed count: the array collapses to
    # one number, so equal-length output across stars doesn't matter here.
    bin_width = (duration / bins_per_duration) * u.hour          # Quantity
    binned = folded.bin(time_bin_size=bin_width)                 # FoldedLightCurve

    # See prepare_transit: binned.flux.value is a MaskedNDArray, and masked entries
    # must become real NaN before any nan-aware function touches them.
    flux = np.asarray(binned.flux.value, dtype=float)     # ndarray of float
    phase = np.asarray(binned.phase.value, dtype=float)   # ndarray of float, days

    # Phase -period/2 and +period/2 are the SAME physical point, but they sit at
    # opposite ends of the array, so find_runs treats a secondary centred there as
    # two separate half-length runs. Rolling by half the length puts that wrap point
    # at the array centre, where the run-finder sees it as contiguous.
    #
    # The cost is that the primary is now split across the array edges instead —
    # which is free, because outside_primary blanks it either way. After the roll
    # index 0 sits at phase ~0, so no run can begin or end at an edge.
    #
    # This is not cosmetic: a 4-bin secondary split into two 2-bin runs still passed
    # SECONDARY_MIN_RUN_BINS, but a 2-bin secondary split into two 1-bin runs was
    # discarded as noise — so the min2 variant could not be evaluated fairly at all
    # until this was fixed.
    roll_by = len(flux) // 2
    flux = np.roll(flux, roll_by)
    phase = np.roll(phase, roll_by)

    # Blank generously: the primary's sloped edges are deep enough to win any
    # "find the deepest dip" search and be misreported as a secondary.
    outside_primary = np.abs(phase) > (duration / 24) * SECONDARY_BLANK_DURATIONS   # ndarray of bool

    if not np.isfinite(flux[outside_primary]).any():
        blank['secondary_guard'] = 'no data outside the blanked primary'
        return blank

    baseline = float(np.nanmedian(flux[outside_primary]))        # float
    noise = robust_noise(flux[outside_primary])                  # float

    if not np.isfinite(baseline) or np.isnan(noise):
        # Same reasoning as the geometry guard: without a noise estimate the
        # threshold test cannot run, so "no secondary" is not a justified answer.
        blank['secondary_guard'] = 'baseline or noise unmeasurable'
        return blank

    # The & with outside_primary is what stops the run-finder latching onto the
    # primary transit, which is far below any threshold we'd set.
    below_threshold = outside_primary & (flux < (baseline - threshold * noise))   # ndarray of bool
    starts, ends = find_runs(below_threshold)

    results = {'secondary_noise': noise, 'secondary_guard': '',
               'secondary_exclusion_fraction': blank['secondary_exclusion_fraction']}

    # Variant 1 — longest run. Current behaviour, kept so the validated numbers stay
    # comparable. Biased toward wide features, which is the wrong bias for a short
    # deep secondary and the right one for a broad contaminant.
    if len(starts) == 0:
        results['secondary_depth_longest'] = 0.0    # a real answer ("no secondary"), not a failure
        results['secondary_phase_fraction_longest'] = np.nan
    else:
        lengths = ends - starts
        longest = int(np.argmax(lengths))
        start, end = int(starts[longest]), int(ends[longest])
        results['secondary_depth_longest'] = baseline - float(np.nanmedian(flux[start:end]))
        results['secondary_phase_fraction_longest'] = circular_mean_phase_fraction(
            phase[start:end], period
        )

    # Variant 2 — deepest run of any length. Fragile on purpose, as the control: a
    # single bin just past threshold is its own median, so it can outscore a real
    # secondary. Recorded to show how much that actually costs.
    depth_any, phase_any = _best_run_by_depth(flux, phase, baseline, period, starts, ends, min_bins=1)
    results['secondary_depth_deepest'] = depth_any
    results['secondary_phase_fraction_deepest'] = phase_any

    # Variant 3 — deepest run of at least SECONDARY_MIN_RUN_BINS. The contiguity
    # protection longest-run was providing, without the width bias.
    depth_min2, phase_min2 = _best_run_by_depth(
        flux, phase, baseline, period, starts, ends, min_bins=SECONDARY_MIN_RUN_BINS
    )
    results['secondary_depth_deepest_min2'] = depth_min2
    results['secondary_phase_fraction_deepest_min2'] = phase_min2

    return results


# ---------------------------------------------------------------------------
# Stage 5 — one star, all six features
# ---------------------------------------------------------------------------

def extract_features(kepid, period, epoch_time, duration,
                     star_lc=None, star_info=None, download_dir=None, siblings=None):
    """Run the whole pipeline for one KOI and return its features and diagnostics.

    In:  kepid       — int
         period      — float, days      (koi_period)
         epoch_time  — float, BKJD      (koi_time0bk)
         duration    — float, hours     (koi_duration)
         star_lc     — LightCurve or None; pass a cached one to skip the download
         star_info   — dict or None, the diagnostics that came with star_lc
         download_dir — str or None
         siblings    — iterable of (period, epoch, duration) for the OTHER KOIs on
                       this star, or None. See build_sibling_mask.
    Out: (features dict, diagnostics dict)
    """
    if star_lc is None:
        star_lc, star_info = load_star(kepid, download_dir=download_dir)

    diagnostics = blank_diagnostics()
    diagnostics.update(star_info or {})
    diagnostics['duration_over_period'] = (duration / 24) / period

    # Two masks, deliberately different. transit_mask is the primary only and feeds
    # measure_odd_even_diff, which must not see secondary-eclipse points. detrend_mask
    # also covers the expected secondary, because anything the trend fit can see is
    # something it partly divides out. See build_transit_mask.
    transit_mask = build_transit_mask(star_lc, period, epoch_time, duration)    # ndarray of bool
    secondary_mask = build_transit_mask(star_lc, period, epoch_time, duration,
                                        include_secondary=True)                # ndarray of bool
    sibling_mask = build_sibling_mask(star_lc, siblings)                        # ndarray of bool

    diagnostics['n_siblings'] = len(siblings or ())
    diagnostics['sibling_masked_fraction'] = float(sibling_mask.mean())

    # Fallback chain, four rungs. Each masked region is one the trend fit must not
    # see, and each one makes "the mask covers the whole orbit" likelier — past that
    # point flatten has no baseline left to fit and raises. Step down rather than
    # lose the row: the other features are degraded without detrending, not
    # meaningless. detrend_fallback records which rung fired.
    for candidate, label in (
        (secondary_mask | sibling_mask, ''),
        (secondary_mask, 'secondary_only'),
        (transit_mask, 'primary_only'),
        (None, 'none'),
    ):
        if candidate is None or (~candidate).mean() >= MIN_UNMASKED_FRACTION_FOR_DETREND:
            detrend_mask, fallback = candidate, label
            break

    diagnostics['detrend_fallback'] = fallback
    diagnostics['detrend_masked_fraction'] = (
        1.0 if detrend_mask is None else float(detrend_mask.mean())
    )

    flat_star_lc, flatten_info = flatten_star(star_lc, detrend_mask, duration)  # LightCurve
    diagnostics.update(flatten_info)

    depth, flux, phase, baseline, local_info = prepare_transit(
        flat_star_lc, period, epoch_time, duration, return_info=True
    )
    diagnostics.update(local_info)

    secondary = measure_secondary_depth(flat_star_lc, period, epoch_time, duration,
                                        exclude_mask=sibling_mask)
    diagnostics.update(secondary)

    features = {
        'depth': depth,
        'width': measure_width(flux, phase, baseline, depth),
        'asymmetry': measure_asymmetry(flux, phase, baseline, duration),
        'snr': measure_snr(flux, phase, depth, duration),
        'odd_even_diff': measure_odd_even_diff(
            flat_star_lc, transit_mask, period, epoch_time, duration
        ),
        # The min2 variant, not longest-run. See measure_secondary_depth for the
        # numbers that settled it.
        'secondary_depth': secondary['secondary_depth_deepest_min2'],
    }

    return features, diagnostics


def siblings_for(row, sibling_table):
    """Find the other KOI rows belonging to the same star.

    In:  row            — pandas Series, the KOI being processed
         sibling_table  — DataFrame with kepid, kepoi_name, koi_period, koi_time0bk,
                          koi_duration; or None
    Out: list of (period days, epoch BKJD, duration hours)
    """
    if sibling_table is None:
        return []

    same_star = sibling_table[
        (sibling_table['kepid'] == row['kepid'])
        & (sibling_table['kepoi_name'] != row.get('kepoi_name', ''))
    ]
    return [
        (s['koi_period'], s['koi_time0bk'], s['koi_duration'])
        for _, s in same_star.iterrows()
    ]


def process_koi(row, star_cache=None, download_dir=None, verbose=True, sibling_table=None):
    """Turn one catalogue row into one output row, whatever happens.

    A failed star keeps its row with NaN features rather than disappearing, so
    failures can be counted and inspected afterwards instead of being inferred
    from a missing index.

    In:  row         — pandas Series, one KOI catalogue row
         star_cache  — StarCache or None
         download_dir — str or None
         verbose     — bool, print failures as they happen
    Out: dict with exactly OUTPUT_COLUMNS as keys
    """
    base = {
        'kepid': row['kepid'],
        'kepoi_name': row.get('kepoi_name', ''),
        'disposition': row.get('koi_disposition', ''),
        'koi_depth_ppm': row.get('koi_depth', np.nan),
        'koi_duration_hours': row.get('koi_duration', np.nan),
        'koi_period_days': row.get('koi_period', np.nan),
        'koi_num_transits': row.get('koi_num_transits', np.nan),
    }

    reason = precheck_koi(row)
    if reason is not None:
        if verbose:
            print(f"  [precheck] {base['kepoi_name']} (KIC {int(row['kepid'])}): {reason}")
        return {**base, **nan_features(), **blank_diagnostics(),
                'error': reason, 'error_kind': 'precheck'}

    try:
        if star_cache is not None:
            star_lc, star_info = star_cache.get(row['kepid'])
        else:
            star_lc, star_info = load_star(row['kepid'], download_dir=download_dir)

        features, diagnostics = extract_features(
            kepid=row['kepid'],
            period=row['koi_period'],
            epoch_time=row['koi_time0bk'],
            duration=row['koi_duration'],
            star_lc=star_lc,
            star_info=star_info,
            download_dir=download_dir,
            siblings=siblings_for(row, sibling_table),
        )
        diagnostics['n_koi_for_star'] = row.get('n_koi_for_star', np.nan)

        return {**base, **features, **diagnostics, 'error': '', 'error_kind': ''}

    # TEMPORARY, and deliberate: at ~1000 stars we do not yet know what breaks, so
    # this catches everything and records what it was. Once the first full run
    # reports its error_kind histogram, narrow this to the types that actually
    # occurred. It is not here because the failure modes are unknowable — it is here
    # to find out what they are.
    except Exception as e:                                       # noqa: BLE001
        kind = classify_error(e)
        message = f"{type(e).__name__}: {e}"[:300]

        # A truncated cache file will be read back identically forever, so evict it
        # here rather than leaving the next run to hit the same corpse.
        if kind == 'corruptcache':
            removed = evict_star_cache(row['kepid'], download_dir)
            message = f"{message} [evicted {removed} cached dir(s)]"[:300]

        if verbose:
            print(f"  [{kind}] {base['kepoi_name']} (KIC {int(row['kepid'])}): {message}")
        return {**base, **nan_features(), **blank_diagnostics(),
                'error': message, 'error_kind': kind}


# ---------------------------------------------------------------------------
# Stage 6 — sampling and running across a batch
# ---------------------------------------------------------------------------

def sample_kois(koi_table, n_stars=100, balance='natural', random_state=42):
    """Draw a batch of KOI rows, sampling by star rather than by row.

    Sampling by star matters for two reasons: the grouped train/test split needs
    whole stars on one side, and a star's rows share a single download.

    CANDIDATE rows are excluded here and nowhere else. They are neither confirmed
    planets nor known false positives, and `y = (disposition == 'CONFIRMED')` would
    silently train all 1977 of them as false positives.

    In:  koi_table    — DataFrame, the full cumulative table
         n_stars      — int, how many distinct stars to draw
         balance      — 'natural' keeps the catalogue's own 1.76:1 FALSE POSITIVE to
                        CONFIRMED ratio, which is what deployment looks like and what
                        keeps precision and recall meaningful. 'balanced' draws equal
                        numbers of each *star*. Natural is the default: at 1.76:1 the
                        imbalance is mild, and class_weight='balanced' is available at
                        training time, which is the reversible place to decide.

                        Note 'balanced' does not give balanced *rows*, and overshoots:
                        50/50 stars at random_state=42 yields 78 CONFIRMED to 51 FALSE
                        POSITIVE rows, because confirmed multi-planet systems carry
                        more KOI rows per star. Balancing rows and grouping by star are
                        not simultaneously satisfiable here.
         random_state — int
    Out: DataFrame with a clean index, sorted by kepid, plus an n_koi_for_star column
    """
    if balance not in ('natural', 'balanced'):
        raise ValueError(f"balance must be 'natural' or 'balanced', got {balance!r}")

    labelled = koi_table[koi_table['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
    labelled['n_koi_for_star'] = labelled.groupby('kepid')['kepid'].transform('size')

    # A star's stratum, for 'balanced'. 47 stars carry both a CONFIRMED and a FALSE
    # POSITIVE row; calling those CONFIRMED is a choice, and it only affects which
    # pool they are drawn from, not their row labels.
    star_label = (
        labelled.groupby('kepid')['koi_disposition']
        .apply(lambda s: 'CONFIRMED' if (s == 'CONFIRMED').any() else 'FALSE POSITIVE')
    )

    rng = np.random.default_rng(random_state)

    if balance == 'natural':
        pool = star_label.index.to_numpy()
        chosen = rng.choice(pool, size=min(n_stars, len(pool)), replace=False)
    else:
        per_class = n_stars // 2
        chosen = []
        for label in ('CONFIRMED', 'FALSE POSITIVE'):
            pool = star_label[star_label == label].index.to_numpy()
            chosen.append(rng.choice(pool, size=min(per_class, len(pool)), replace=False))
        chosen = np.concatenate(chosen)

    batch = labelled[labelled['kepid'].isin(chosen)].copy()
    # Sorting by kepid is what makes StarCache's single slot sufficient.
    batch = batch.sort_values(['kepid', 'kepoi_name']).reset_index(drop=True)
    return batch


def load_checkpoint(checkpoint_path, retry_kinds=('transient',)):
    """Read a checkpoint and decide which KOIs still need work.

    Resume distinguishes failure types: a MAST timeout should be retried on the
    next run, a precheck rejection or a target with no data at this cadence should
    not. 'unknown' is not retried by default — see classify_error.

    In:  checkpoint_path — str or None
         retry_kinds     — tuple of str, error kinds to attempt again
    Out: dict of kepoi_name -> stored row dict, for rows that should NOT be redone
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return {}

    stored = pd.read_csv(checkpoint_path)
    if len(stored) == 0:
        return {}

    # A retried KOI appears more than once; the last write is the current answer.
    stored = stored.drop_duplicates(subset='kepoi_name', keep='last')
    stored['error_kind'] = stored['error_kind'].fillna('')

    done = stored[~stored['error_kind'].isin(retry_kinds)]
    return {r['kepoi_name']: r.to_dict() for _, r in done.iterrows()}


def build_feature_table(koi_batch, checkpoint_path=DEFAULT_CHECKPOINT,
                        download_dir=DEFAULT_DOWNLOAD_DIR,
                        retry_kinds=('transient',), verbose=True,
                        sibling_table=None):
    """Extract features for every row of a KOI table, with caching and checkpointing.

    Rows are written to the checkpoint as they complete, so a crash at row 800 does
    not lose the first 799. Runtime is dominated by downloads, not computation, and
    this runs serially on purpose: establishing the failure taxonomy on a serial run
    is easier than debugging a thread pool and a broad except at the same time.

    In:  koi_batch       — DataFrame with the KOI columns; index is not relied on
         checkpoint_path — str or None to disable
         download_dir    — str, where lightkurve caches FITS files
         retry_kinds     — tuple of str, which failures to attempt again on resume
         verbose         — bool
    Out: DataFrame, one row per KOI row, columns exactly OUTPUT_COLUMNS
    """
    if download_dir:
        os.makedirs(download_dir, exist_ok=True)

    # Default the sibling lookup to the batch itself. sample_kois returns every KOI
    # row for each star it picks, so the batch already holds the siblings; pass an
    # explicit table when feeding build_feature_table a partial selection.
    if sibling_table is None and 'kepoi_name' in koi_batch.columns:
        sibling_table = koi_batch

    batch = koi_batch.sort_values(['kepid', 'kepoi_name']) if 'kepoi_name' in koi_batch.columns \
        else koi_batch.sort_values('kepid')

    already_done = load_checkpoint(checkpoint_path, retry_kinds=retry_kinds)
    if already_done and verbose:
        print(f"Resuming: {len(already_done)} KOIs already complete in {checkpoint_path}")

    cache = StarCache(download_dir=download_dir)
    rows = []
    header_written = bool(checkpoint_path) and os.path.exists(checkpoint_path)

    for position, (_, row) in enumerate(batch.iterrows(), start=1):
        koi_name = row.get('kepoi_name', '')

        if koi_name in already_done:
            rows.append(already_done[koi_name])
            continue

        if verbose:
            print(f"[{position}/{len(batch)}] {koi_name} (KIC {int(row['kepid'])})")

        result = process_koi(row, star_cache=cache, download_dir=download_dir,
                             verbose=verbose, sibling_table=sibling_table)
        rows.append(result)

        if checkpoint_path:
            pd.DataFrame([result], columns=list(OUTPUT_COLUMNS)).to_csv(
                checkpoint_path, mode='a', header=not header_written, index=False
            )
            header_written = True

    if verbose:
        print(f"\nStar loads attempted: {cache.misses} (some may have failed); "
              f"{cache.hits} rows served from cache")

    return pd.DataFrame(rows, columns=list(OUTPUT_COLUMNS))


# ---------------------------------------------------------------------------
# Optional — plot one star's local view, for eyeballing
# ---------------------------------------------------------------------------

def plot_local_view(kepid, period, epoch_time, duration, label=None, download_dir=None):
    """Download, clean, flatten, fold and bin one star, then plot it.

    Fixed bin count is correct here (unlike in the feature functions): consistent
    resolution across stars is what makes plots comparable by eye.

    In:  kepid, period, epoch_time, duration — as above
         label — str or None, legend text
    Out: matplotlib Axes
    """
    star_lc, _ = load_star(kepid, download_dir=download_dir)
    detrend_mask = build_transit_mask(star_lc, period, epoch_time, duration,
                                      include_secondary=True)
    flat_star_lc, _ = flatten_star(star_lc, detrend_mask, duration)

    folded = flat_star_lc.fold(period=period, epoch_time=epoch_time)

    half_window_days = (duration * LOCAL_WINDOW_DURATIONS / 2) / 24
    local_view = folded[np.abs(folded.phase.value) < half_window_days]

    bin_width = (duration * LOCAL_WINDOW_DURATIONS / LOCAL_N_BINS) * u.hour
    binned = local_view.bin(time_bin_size=bin_width)

    return binned.plot(label=label or f"KIC {int(kepid)}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def describe_batch(batch):
    """Print what a sampled batch contains, without downloading anything.

    Reports stars and rows separately: sampling is by star, so "100 stars" is not
    "100 rows".
    """
    n_stars = batch['kepid'].nunique()
    counts = batch['koi_disposition'].value_counts()

    print(f"Batch: {n_stars} stars, {len(batch)} KOI rows")
    for label, count in counts.items():
        print(f"  {label:<16} {count:>5} rows ({count / len(batch):.1%})")

    multi = batch.groupby('kepid').size()
    print(f"  stars with >1 KOI row: {int((multi > 1).sum())} "
          f"(max {int(multi.max())} rows on one star)")

    reasons = [precheck_koi(row) for _, row in batch.iterrows()]
    rejected = [r for r in reasons if r is not None]
    print(f"  precheck rejections:   {len(rejected)} of {len(batch)} rows")
    for reason in sorted(set(rejected)):
        print(f"    {reason}: {reasons.count(reason)}")


if __name__ == '__main__':
    koi_table = pd.read_csv("MyProject_sync.csv", low_memory=False)   # DataFrame

    test_batch = sample_kois(koi_table, n_stars=100, balance='natural', random_state=42)
    describe_batch(test_batch)

    if '--run' not in sys.argv:
        print("\nDry run — nothing downloaded. Re-run with --run to extract features.")
        sys.exit(0)

    feature_table = build_feature_table(test_batch)
    print(feature_table[list(IDENTIFIER_NAMES) + list(FEATURE_NAMES)].to_string())
