"""Launch (or resume) the 1000-star extraction.

Separate entry point so the 100-star __main__ stays a fast smoke test.

    python run_1000.py                      # dry run: sample + precheck only
    python run_1000.py --run                # run, retrying transient failures
    python run_1000.py --run --retry KINDS  # widen what a resume re-attempts,
                                            # e.g. --retry transient,unknown,diskfull

--retry matters after an environmental failure. The 1000-star run filled the disk
at row ~1100; 37 rows were recorded as 'unknown' because classify_error did not yet
know about ENOSPC or the truncated FITS it leaves behind. Those stored kinds do not
change retroactively, so re-attempting them needs the kind they were stored under.
"""
import sys

import pandas as pd

import transitcheck_features as tf

N_STARS = 1000
CHECKPOINT = 'features_1000.csv'


def parse_retry_kinds(argv):
    """In: argv list. Out: tuple of error kinds a resume should re-attempt."""
    if '--retry' not in argv:
        return ('transient', 'diskfull', 'corruptcache')
    value = argv[argv.index('--retry') + 1]
    return tuple(k.strip() for k in value.split(',') if k.strip())


if __name__ == '__main__':
    cat = pd.read_csv("MyProject_sync.csv", low_memory=False)
    batch = tf.sample_kois(cat, n_stars=N_STARS, balance='natural', random_state=42)
    tf.describe_batch(batch)

    if '--run' not in sys.argv:
        print("\nDry run. Re-run with --run.")
        sys.exit(0)

    retry_kinds = parse_retry_kinds(sys.argv)
    print(f"\nretrying stored kinds: {retry_kinds}")

    # Pass the UNFILTERED catalogue as the sibling table. The batch has already
    # been filtered to CONFIRMED/FALSE POSITIVE, so defaulting to it would leave
    # CANDIDATE siblings unmasked -- and a candidate planet contaminates the
    # secondary search exactly as much as a confirmed one. Measured in the
    # 1000-star run: rows on a star with a CANDIDATE sibling put only 25% of
    # detected dips near phase +/-0.5, against 51% for rows without one.
    table = tf.build_feature_table(batch, checkpoint_path=CHECKPOINT,
                                   retry_kinds=retry_kinds,
                                   sibling_table=cat)
    print(f"\nwrote {len(table)} rows to {CHECKPOINT}")
