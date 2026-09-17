"""Reproduce the notebook's training run and check the recorded results.

The notebook saves output for the headline metrics but not for three claims: the
grid search, the catalogue-only baseline, and the 1000-vs-2000 comparison (those
cells are commented out). This re-runs everything from the same CSV with the same
random states, so every number in the writeup is backed by something reproducible.

    python verify_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ['depth', 'width', 'asymmetry', 'snr',
            'odd_even_diff', 'secondary_depth', 'duration_over_period']
THRESHOLD = 0.25


def load(path):
    df = pd.read_csv(path).drop_duplicates(subset='kepoi_name', keep='last')
    ok = df[df['error'].isna()].copy()
    return (ok[FEATURES], (ok['disposition'] == 'CONFIRMED').astype(int), ok['kepid'])


def split(X, y, groups):
    outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(outer.split(X, y, groups))
    return train_idx, test_idx


def fit_predict(X, y, train_idx, test_idx, features=None, threshold=THRESHOLD):
    cols = features or FEATURES
    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X.iloc[train_idx][cols], y.iloc[train_idx])
    proba = model.predict_proba(X.iloc[test_idx][cols])[:, 1]
    return (proba >= threshold).astype(int)


def rule(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


X, y, groups = load('features_2000.csv')
train_idx, test_idx = split(X, y, groups)
y_test = y.iloc[test_idx]

rule("1. SPLIT — does it reproduce?")
print(f"successful rows       : {len(X)}        (notebook: 2224)")
print(f"test rows             : {len(test_idx)}         (recorded: 434)")
print(f"test CONFIRMED        : {int(y_test.sum())}         (recorded: 133)")
print(f"test FALSE POSITIVE   : {int((1 - y_test).sum())}         (recorded: 301)")
print(f"star-disjoint         : {not (set(groups.iloc[train_idx]) & set(groups.iloc[test_idx]))}")

rule("2. HEADLINE METRICS — do they reproduce?")
pred = fit_predict(X, y, train_idx, test_idx)
print(classification_report(y_test, pred,
                            target_names=['FALSE POSITIVE', 'CONFIRMED'], digits=2))
cm = confusion_matrix(y_test, pred)
print(cm)
print(f"\nrecorded: [[240 61] [16 117]]   reproduced: [[{cm[0,0]} {cm[0,1]}] [{cm[1,0]} {cm[1,1]}]]")
print(f"match: {cm.tolist() == [[240, 61], [16, 117]]}")

rule("3. BASE RATES — which one is the right comparison?")
overall_fp = float((y == 0).mean())
test_fp = float((y_test == 0).mean())
print(f"FALSE POSITIVE share, whole dataset : {overall_fp:.4f}")
print(f"FALSE POSITIVE share, TEST set      : {test_fp:.4f}   <- the majority-class")
print(f"                                                        baseline to beat")
print(f"model accuracy                      : {(pred == y_test).mean():.4f}")
print("\nStar-grouped splitting means the test set's class mix differs from the")
print("whole dataset's. A baseline must be quoted against the split it is scored on.")

rule("4. CATALOGUE-ONLY BASELINE — duration_over_period alone")
for cols, label in (
    (['duration_over_period'], 'duration_over_period only'),
    (FEATURES, 'all seven features'),
):
    p = fit_predict(X, y, train_idx, test_idx, features=cols)
    acc = (p == y_test).mean()
    tn, fp_, fn, tp = confusion_matrix(y_test, p).ravel()
    rec = tp / (tp + fn) if (tp + fn) else 0
    pre = tp / (tp + fp_) if (tp + fp_) else 0
    print(f"  {label:<28} accuracy {acc:.4f}   CONFIRMED precision {pre:.2f} recall {rec:.2f}")
print(f"  {'always say FALSE POSITIVE':<28} accuracy {test_fp:.4f}   CONFIRMED precision 0.00 recall 0.00")

rule("5. DOES DOUBLING THE DATA HELP? — 1000 vs 2000 stars")
X1, y1, g1 = load('features_1000.csv')
tr1, te1 = split(X1, y1, g1)
p1 = fit_predict(X1, y1, tr1, te1)
tn, fp_, fn, tp = confusion_matrix(y1.iloc[te1], p1).ravel()
print(f"  1000-star table: {len(X1)} rows, test {len(te1)}   "
      f"accuracy {(p1 == y1.iloc[te1]).mean():.4f}  "
      f"CONFIRMED precision {tp / (tp + fp_):.2f} recall {tp / (tp + fn):.2f}")
tn, fp_, fn, tp = cm.ravel()
print(f"  2000-star table: {len(X)} rows, test {len(test_idx)}   "
      f"accuracy {(pred == y_test).mean():.4f}  "
      f"CONFIRMED precision {tp / (tp + fp_):.2f} recall {tp / (tp + fn):.2f}")
print("\n  NOTE: different test sets, so this is indicative, not a controlled comparison.")

rule("6. DOES koi_impact HELP?")
cat = pd.read_csv('MyProject_sync.csv', low_memory=False)
full = pd.read_csv('features_2000.csv').drop_duplicates(subset='kepoi_name', keep='last')
full = full[full['error'].isna()].merge(
    cat[['kepoi_name', 'koi_impact']], on='kepoi_name', how='left')
Xi = full[FEATURES + ['koi_impact']]
yi = (full['disposition'] == 'CONFIRMED').astype(int)
gi = full['kepid']
tri, tei = split(Xi, yi, gi)
pi = fit_predict(Xi, yi, tri, tei, features=FEATURES + ['koi_impact'])
tn, fp_, fn, tp = confusion_matrix(yi.iloc[tei], pi).ravel()
print(f"  with koi_impact   : accuracy {(pi == yi.iloc[tei]).mean():.4f}  "
      f"CONFIRMED precision {tp / (tp + fp_):.2f} recall {tp / (tp + fn):.2f}  "
      f"({int(full['koi_impact'].isna().sum())} rows missing koi_impact)")
tn, fp_, fn, tp = cm.ravel()
print(f"  without           : accuracy {(pred == y_test).mean():.4f}  "
      f"CONFIRMED precision {tp / (tp + fp_):.2f} recall {tp / (tp + fn):.2f}")
print()
