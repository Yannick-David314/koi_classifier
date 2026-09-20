"""Feature-set ablation: what does each family of features actually buy?

The project's claim is that LIGHT-CURVE features carry the performance. That claim is
currently untested, because the top-ranked feature (duration_over_period) is computed
from catalogue columns, not from the light curve.

This separates the two families cleanly and measures each, paired over the same 25
star-grouped splits so the differences are comparable rather than anecdotal.

    LIGHT_CURVE   measured by this pipeline from Kepler photometry
    CATALOGUE     read from the KOI table's transit-model fit

    python ablate_features.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix, roc_auc_score

LIGHT_CURVE = ['depth', 'width', 'asymmetry', 'snr', 'odd_even_diff', 'secondary_depth']
CATALOGUE = ['duration_over_period', 'koi_impact']
THRESHOLD = 0.25
SEEDS = list(range(100, 125))

CONFIGS = [
    ('light-curve only (6)',            LIGHT_CURVE),
    ('light-curve + dur/period (7)',    LIGHT_CURVE + ['duration_over_period']),
    ('light-curve + both cat (8)',      LIGHT_CURVE + CATALOGUE),
    ('catalogue only (2)',              CATALOGUE),
    ('dur/period alone (1)',            ['duration_over_period']),
]


def load():
    df = pd.read_csv('features_2000.csv').drop_duplicates(subset='kepoi_name', keep='last')
    ok = df[df['error'].isna()].copy()
    cat = pd.read_csv('MyProject_sync.csv', low_memory=False)
    ok = ok.merge(cat[['kepoi_name', 'koi_impact']], on='kepoi_name', how='left')
    return ok


def run(ok, cols, seed):
    X = ok[cols]
    y = (ok['disposition'] == 'CONFIRMED').astype(int)
    g = ok['kepid']
    tr, te = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, g))
    m = HistGradientBoostingClassifier(random_state=42).fit(X.iloc[tr], y.iloc[tr])
    proba = m.predict_proba(X.iloc[te])[:, 1]
    pred = (proba >= THRESHOLD).astype(int)
    yt = y.iloc[te]
    tn, fp, fn, tp = confusion_matrix(yt, pred).ravel()
    return {
        'acc': (pred == yt).mean(),
        'auc': roc_auc_score(yt, proba),
        'prec': tp / (tp + fp) if (tp + fp) else 0.0,
        'rec': tp / (tp + fn) if (tp + fn) else 0.0,
        'queue_cut': (tn + fn) / len(yt),          # share auto-rejected
        'planets_lost': fn / (fn + tp) if (fn + tp) else 0.0,
    }


ok = load()
print(f"rows {len(ok)}   stars {ok['kepid'].nunique()}   "
      f"koi_impact missing on {int(ok['koi_impact'].isna().sum())} rows\n")

results = {name: [run(ok, cols, s) for s in SEEDS] for name, cols in CONFIGS}

print(f"{'configuration':<30} {'accuracy':>16} {'ROC AUC':>16} {'CONF P':>8} {'CONF R':>8}")
print('-' * 82)
for name, _ in CONFIGS:
    r = results[name]
    acc = np.array([x['acc'] for x in r])
    auc = np.array([x['auc'] for x in r])
    print(f"{name:<30} {acc.mean():>9.4f} ±{acc.std():.4f} "
          f"{auc.mean():>9.4f} ±{auc.std():.4f} "
          f"{np.mean([x['prec'] for x in r]):>8.3f} {np.mean([x['rec'] for x in r]):>8.3f}")

print(f"\n{'configuration':<30} {'queue cut':>12} {'planets lost':>14}")
print('-' * 58)
for name, _ in CONFIGS:
    r = results[name]
    print(f"{name:<30} {np.mean([x['queue_cut'] for x in r]):>11.1%} "
          f"{np.mean([x['planets_lost'] for x in r]):>13.1%}")


def paired(a, b, metric='acc'):
    d = np.array([x[metric] for x in results[a]]) - np.array([x[metric] for x in results[b]])
    se = d.std() / np.sqrt(len(d))
    return d.mean(), d.std(), int((d > 0).sum()), len(d), (d.mean() / se if se else float('nan'))

print("\npaired differences over the same 25 splits (accuracy):")
print('-' * 82)
for a, b, label in [
    ('light-curve + dur/period (7)', 'light-curve only (6)',
     'what duration_over_period adds to the light-curve model'),
    ('light-curve + both cat (8)', 'light-curve + dur/period (7)',
     'what koi_impact adds on top of that'),
    ('light-curve only (6)', 'catalogue only (2)',
     'light-curve features vs catalogue features, head to head'),
]:
    m, s, w, n, t = paired(a, b)
    verdict = 'REAL' if abs(t) > 2 else 'within noise'
    print(f"  {label}")
    print(f"    {m:+.4f} ± {s:.4f}   better in {w}/{n} splits   t = {t:+.1f}   -> {verdict}\n")
