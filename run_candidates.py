"""Launch (or resume) feature extraction for every CANDIDATE KOI.

Step 1 of the review app (Phase 4): the model can only score rows that have
features, and no CANDIDATE row has any yet -- sample_kois excludes them on purpose,
because their label is unknown (CLAUDE.md 10.7). That exclusion is about training.
Scoring them is the point of the whole project, so here they are taken directly.

    python run_candidates.py                      # dry run: describe + precheck only
    python run_candidates.py --run                # run, retrying transient failures
    python run_candidates.py --run --retry KINDS  # widen what a resume re-attempts

Same pipeline, caching, throttling and resume logic as run_1000.py.
"""
import sys

import pandas as pd

import transitcheck_features as tf

CHECKPOINT = 'features_candidates.csv'


def parse_retry_kinds(argv):
    """In: argv list. Out: tuple of error kinds a resume should re-attempt."""
    if '--retry' not in argv:
        return ('transient', 'diskfull', 'corruptcache')
    value = argv[argv.index('--retry') + 1]
    return tuple(k.strip() for k in value.split(',') if k.strip())


if __name__ == '__main__':
    cat = pd.read_csv("MyProject_sync.csv", low_memory=False)

    batch = cat[cat['koi_disposition'] == 'CANDIDATE'].copy()
    # A diagnostic column the training tables also carry. Counted over the whole
    # catalogue, so it means "KOIs on this star", whatever their disposition. (The
    # training tables counted labelled rows only, so the two are not identical.)
    batch['n_koi_for_star'] = batch['kepid'].map(cat.groupby('kepid').size())
    # Sorted by star so StarCache's single slot serves every KOI on a star from one
    # download, as sample_kois arranges for the training batches.
    batch = batch.sort_values(['kepid', 'kepoi_name']).reset_index(drop=True)
    tf.describe_batch(batch)

    if '--run' not in sys.argv:
        print("\nDry run. Re-run with --run.")
        sys.exit(0)

    retry_kinds = parse_retry_kinds(sys.argv)
    print(f"\nretrying stored kinds: {retry_kinds}")

    # The unfiltered catalogue as the sibling table, for the CLAUDE.md 10.18 reason:
    # every other KOI on the star -- confirmed, false positive or candidate -- has
    # transits that would otherwise contaminate the secondary search.
    table = tf.build_feature_table(batch, checkpoint_path=CHECKPOINT,
                                   retry_kinds=retry_kinds,
                                   sibling_table=cat)
    print(f"\nwrote {len(table)} rows to {CHECKPOINT}")
