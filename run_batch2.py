"""Extract the second ~1000-star batch and append it to the existing feature table.

Adds NO pipeline changes. It calls build_feature_table exactly as the first run did,
so the combined table stays homogeneous. Everything here is batching, guards and
instrumentation -- none of it touches how a row is measured.

WHY THIS RUNS IN SHORT INVOCATIONS
    The first attempt ran all 40 chunks in one process and was killed at chunk 12
    with exit code 4 and no traceback -- the signature of an external kill, not a
    Python exception. The machine has 7.8 GB of RAM and was at 92% load with 0.6 GB
    free. Disk was never the problem (21.8 GB free, 1.7 GB cache).

    So each invocation now does a few chunks and exits, and a driver re-launches it.
    Process death is the only truly reliable way to return memory that a long-lived
    Python process is holding through C extensions. build_feature_table's existing
    checkpoint/resume means restarting costs nothing but interpreter start-up.

    Exit codes: 0 = batch complete, 10 = more work remains (driver re-launches),
    20 = disk floor (fatal, needs a human), 21 = RAM floor (transient, driver waits
    and retries -- free RAM here swings with other activity on the machine).

Chunking is by STAR, never by row, so every KOI row of a star is processed inside one
build_feature_table call -- which is what lets the single-slot StarCache serve the
repeat rows instead of re-downloading.

    python run_batch2.py                        # dry run
    python run_batch2.py --run                  # all remaining chunks in one process
    python run_batch2.py --run --max-chunks 4   # four chunks, then exit 10
"""

import ctypes
import gc
import os
import shutil
import sys
from ctypes import wintypes

import pandas as pd

import transitcheck_features as tf

EXISTING = 'features_1000.csv'
COMBINED = 'features_2000.csv'
CATALOGUE = 'MyProject_sync.csv'

N_NEW_STARS = 1000
CHUNK_STARS = 25
MIN_FREE_GB = 2.0          # disk floor
MIN_FREE_RAM_MB = 400      # memory floor -- pause cleanly rather than be killed
RANDOM_STATE = 43

EXIT_COMPLETE = 0
EXIT_MORE_WORK = 10
EXIT_DISK_FLOOR = 20   # fatal: disk does not free itself, a human must intervene
EXIT_RAM_FLOOR = 21    # retryable: observed free RAM swinging 441 -> 1652 MB during
                       # a single run, driven by other activity on the machine rather
                       # than by this process. Waiting is usually enough.


# --------------------------------------------------------------------------
# Resource probes
# --------------------------------------------------------------------------

class _MemoryStatusEx(ctypes.Structure):
    _fields_ = [('dwLength', ctypes.c_ulong), ('dwMemoryLoad', ctypes.c_ulong),
                ('ullTotalPhys', ctypes.c_ulonglong), ('ullAvailPhys', ctypes.c_ulonglong),
                ('ullTotalPageFile', ctypes.c_ulonglong), ('ullAvailPageFile', ctypes.c_ulonglong),
                ('ullTotalVirtual', ctypes.c_ulonglong), ('ullAvailVirtual', ctypes.c_ulonglong),
                ('ullAvailExtendedVirtual', ctypes.c_ulonglong)]


class _ProcessMemoryCounters(ctypes.Structure):
    _fields_ = [('cb', wintypes.DWORD), ('PageFaultCount', wintypes.DWORD),
                ('PeakWorkingSetSize', ctypes.c_size_t), ('WorkingSetSize', ctypes.c_size_t),
                ('QuotaPeakPagedPoolUsage', ctypes.c_size_t), ('QuotaPagedPoolUsage', ctypes.c_size_t),
                ('QuotaPeakNonPagedPoolUsage', ctypes.c_size_t), ('QuotaNonPagedPoolUsage', ctypes.c_size_t),
                ('PagefileUsage', ctypes.c_size_t), ('PeakPagefileUsage', ctypes.c_size_t)]


def system_ram_mb():
    """Out: (available MB, load percent) for the whole machine."""
    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(status)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status))
    return status.ullAvailPhys / 1e6, status.dwMemoryLoad


# GetCurrentProcess returns a HANDLE, which is pointer-sized. ctypes defaults every
# return type to a 32-bit int, which truncates it on 64-bit Windows and makes the
# call fail silently -- returning 0 rather than raising. Declaring the signatures is
# what makes this work at all; without it the probe reported 0 MB forever.
_kernel32 = ctypes.windll.kernel32
_psapi = ctypes.windll.psapi
_kernel32.GetCurrentProcess.restype = wintypes.HANDLE
_psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE,
                                        ctypes.POINTER(_ProcessMemoryCounters),
                                        wintypes.DWORD]
_psapi.GetProcessMemoryInfo.restype = wintypes.BOOL


def process_rss_mb():
    """Out: (current working set MB, peak working set MB). (0, 0) if unavailable."""
    counters = _ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    if not _psapi.GetProcessMemoryInfo(_kernel32.GetCurrentProcess(),
                                       ctypes.byref(counters), counters.cb):
        return 0.0, 0.0
    return counters.WorkingSetSize / 1e6, counters.PeakWorkingSetSize / 1e6


def free_gb(path='.'):
    return shutil.disk_usage(os.path.abspath(path)).free / 1e9


def dir_gb(path):
    if not os.path.isdir(path):
        return 0.0
    total = 0
    for root, _, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total / 1e9


# --------------------------------------------------------------------------

def build_new_batch(cat, existing_kepids):
    """Sample N_NEW_STARS stars not already extracted. Raises if any overlap."""
    labelled = cat[cat['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])]
    available = labelled[~labelled['kepid'].isin(existing_kepids)]

    batch = tf.sample_kois(available, n_stars=N_NEW_STARS,
                           balance='natural', random_state=RANDOM_STATE)

    overlap = set(batch['kepid']) & set(existing_kepids)
    if overlap:
        raise RuntimeError(f"batch overlaps the existing table on {len(overlap)} kepids")
    return batch


def already_done(path):
    """Out: set of kepoi_name already recorded, whatever the outcome."""
    if not os.path.exists(path):
        return set()
    stored = pd.read_csv(path, usecols=['kepoi_name', 'error_kind'])
    stored['error_kind'] = stored['error_kind'].fillna('')
    stored = stored.drop_duplicates(subset='kepoi_name', keep='last')
    keep = stored[~stored['error_kind'].isin(tf_RETRY_KINDS)]
    return set(keep['kepoi_name'])


tf_RETRY_KINDS = ('transient', 'diskfull', 'corruptcache')


def main(argv):
    max_chunks = None
    if '--max-chunks' in argv:
        max_chunks = int(argv[argv.index('--max-chunks') + 1])

    cat = pd.read_csv(CATALOGUE, low_memory=False)
    original = pd.read_csv(EXISTING, usecols=['kepid', 'kepoi_name'])
    batch = build_new_batch(cat, set(original['kepid']))

    if not os.path.exists(COMBINED) and '--run' in argv:
        shutil.copy(EXISTING, COMBINED)
        print(f"seeded {COMBINED} from {EXISTING}")

    done = already_done(COMBINED)
    remaining = batch[~batch['kepoi_name'].isin(done)]

    print(f"new batch      : {len(batch)} rows / {batch['kepid'].nunique()} stars")
    print(f"already done   : {len(batch) - len(remaining)} rows")
    print(f"remaining      : {len(remaining)} rows / {remaining['kepid'].nunique()} stars")

    if '--run' not in argv:
        tf.describe_batch(batch)
        print(f"\nfree disk {free_gb():.1f} GB   free RAM {system_ram_mb()[0]:.0f} MB")
        print("\nDry run. Re-run with --run.")
        return EXIT_COMPLETE

    if len(remaining) == 0:
        print("\nnothing left to do")
        return EXIT_COMPLETE

    star_ids = sorted(remaining['kepid'].unique())
    chunks = [star_ids[i:i + CHUNK_STARS] for i in range(0, len(star_ids), CHUNK_STARS)]
    limit = len(chunks) if max_chunks is None else min(max_chunks, len(chunks))
    print(f"{len(chunks)} chunks remain; this invocation will do {limit}\n")

    for i, chunk in enumerate(chunks[:limit], start=1):
        disk = free_gb()
        ram_mb, ram_load = system_ram_mb()
        rss, peak = process_rss_mb()
        cache = dir_gb(tf.DEFAULT_DOWNLOAD_DIR)

        print(f"--- chunk {i}/{limit}  ({len(chunk)} stars)  "
              f"disk {disk:.1f} GB  cache {cache:.1f} GB  "
              f"RSS {rss:.0f} MB (peak {peak:.0f})  "
              f"sysRAM free {ram_mb:.0f} MB / load {ram_load}% ---")

        if disk < MIN_FREE_GB:
            print(f"\n=== STOPPING (disk): {disk:.2f} GB free, below {MIN_FREE_GB} GB ===")
            return EXIT_DISK_FLOOR
        if ram_mb < MIN_FREE_RAM_MB:
            print(f"\n=== PAUSING (RAM): {ram_mb:.0f} MB free, below {MIN_FREE_RAM_MB} MB ===")
            print("Checkpoint intact. The driver will wait and retry -- free RAM on this")
            print("machine swings with other activity, so this is usually temporary.")
            return EXIT_RAM_FLOOR

        sub = remaining[remaining['kepid'].isin(chunk)]
        tf.build_feature_table(
            sub,
            checkpoint_path=COMBINED,
            sibling_table=cat,       # UNFILTERED, so CANDIDATE siblings are masked
            verbose=True,
        )

        del sub
        gc.collect()

    rss, peak = process_rss_mb()
    print(f"\ninvocation done. RSS {rss:.0f} MB, peak {peak:.0f} MB, "
          f"cache {dir_gb(tf.DEFAULT_DOWNLOAD_DIR):.2f} GB, disk {free_gb():.1f} GB")

    return EXIT_COMPLETE if limit >= len(chunks) else EXIT_MORE_WORK


if __name__ == '__main__':
    sys.exit(main(sys.argv))
