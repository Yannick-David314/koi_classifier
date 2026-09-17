"""Chase down the three recorded numbers that did not reproduce.

  A. Confusion matrix off by 4 rows -- was the final model fit on the inner
     training split (tr) rather than the full outer training set (train_idx)?
  B. Catalogue-only baseline recorded as 0.63 -- reproduces at 0.70. Threshold?
  C. "Doubling the data" and "adding koi_impact" recorded as null -- both moved
     accuracy on a single split. Is that signal or noise? Repeat over many splits.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import confusion_matrix

FEATURES = ['depth', 'width', 'asymmetry', 'snr',
            'odd_even_diff', 'secondary_depth', 'duration_over_period']


def load(path, extra=None):
    df = pd.read_csv(path).drop_duplicates(subset='kepoi_name', keep='last')
    ok = df[df['error'].isna()].copy()
    if extra is not None:
        cat = pd.read_csv('MyProject_sync.csv', low_memory=False)
        ok = ok.merge(cat[['kepoi_name'] + extra], on='kepoi_name', how='left')
    return ok


def evaluate(ok, cols, threshold=0.25, seed=42):
    X, y, g = ok[cols], (ok['disposition'] == 'CONFIRMED').astype(int), ok['kepid']
    tr_i, te_i = next(GroupShuffleSplit(1, test_size=0.2, random_state=seed).split(X, y, g))
    m = HistGradientBoostingClassifier(random_state=42)
    m.fit(X.iloc[tr_i], y.iloc[tr_i])
    pred = (m.predict_proba(X.iloc[te_i])[:, 1] >= threshold).astype(int)
    return (pred == y.iloc[te_i]).mean(), confusion_matrix(y.iloc[te_i], pred)


def rule(t):
    print(f"\n{'=' * 68}\n{t}\n{'=' * 68}")


ok2 = load('features_2000.csv')
X = ok2[FEATURES]
y = (ok2['disposition'] == 'CONFIRMED').astype(int)
g = ok2['kepid']

rule("A. Was the final model fit on `tr` (60%) instead of `train_idx` (80%)?")
train_idx, test_idx = next(GroupShuffleSplit(1, test_size=0.2, random_state=42).split(X, y, g))
a, b = next(GroupShuffleSplit(1, test_size=0.25, random_state=7).split(
    X.iloc[train_idx], y.iloc[train_idx], g.iloc[train_idx]))
tr, va = train_idx[a], train_idx[b]

for name, idx in (('full outer train (train_idx)', train_idx), ('inner train only (tr)', tr)):
    m = HistGradientBoostingClassifier(random_state=42)
    m.fit(X.iloc[idx], y.iloc[idx])
    pred = (m.predict_proba(X.iloc[test_idx])[:, 1] >= 0.25).astype(int)
    cm = confusion_matrix(y.iloc[test_idx], pred)
    hit = cm.tolist() == [[240, 61], [16, 117]]
    print(f"  fit on {name:<30} n={len(idx):<5} -> [[{cm[0,0]} {cm[0,1]}] [{cm[1,0]} {cm[1,1]}]]"
          f"   {'<== MATCHES RECORDED' if hit else ''}")

rule("B. Catalogue-only baseline: which threshold gives 0.63?")
test_major = float((y.iloc[test_idx] == 0).mean())
print(f"  majority-class baseline on this test set: {test_major:.4f}\n")
print(f"  {'threshold':>10}  {'dur/period only':>16}  {'all 7 features':>15}")
for t in (0.25, 0.4, 0.5, 0.6):
    a1, _ = evaluate(ok2, ['duration_over_period'], threshold=t)
    a7, _ = evaluate(ok2, FEATURES, threshold=t)
    print(f"  {t:>10}  {a1:>16.4f}  {a7:>15.4f}")

rule("C. Are the 'null levers' actually null? — 25 random star-grouped splits")
ok1 = load('features_1000.csv')
oki = load('features_2000.csv', extra=['koi_impact'])

seeds = list(range(100, 125))
res = {'1000 stars': [], '2000 stars': [], '2000 + koi_impact': []}
for s in seeds:
    res['1000 stars'].append(evaluate(ok1, FEATURES, seed=s)[0])
    res['2000 stars'].append(evaluate(ok2, FEATURES, seed=s)[0])
    res['2000 + koi_impact'].append(evaluate(oki, FEATURES + ['koi_impact'], seed=s)[0])

print(f"  {'configuration':<22} {'mean acc':>9} {'std':>7} {'min':>7} {'max':>7}")
for k, v in res.items():
    v = np.array(v)
    print(f"  {k:<22} {v.mean():>9.4f} {v.std():>7.4f} {v.min():>7.4f} {v.max():>7.4f}")

d = np.array(res['2000 + koi_impact']) - np.array(res['2000 stars'])
print(f"\n  koi_impact effect, paired over the same 25 splits:")
print(f"    mean {d.mean():+.4f}   std {d.std():+.4f}   "
      f"helped in {int((d > 0).sum())}/{len(d)} splits")
se = d.std() / np.sqrt(len(d))
print(f"    mean/SE = {d.mean() / se:.2f}  (|t| > 2 would be a real effect)")

d2 = np.array(res['2000 stars']) - np.array(res['1000 stars'])
print(f"\n  1000 -> 2000 stars (UNPAIRED, different data and test sets):")
print(f"    mean difference {d2.mean():+.4f}")
print("    Not a controlled comparison -- the test sets differ in size and class mix.")
print()
