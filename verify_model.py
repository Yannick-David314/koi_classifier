"""Reproduce the notebook's training run and check the recorded results.

Retrains the six-feature model from the same CSV with the same random states and
checks the split, the headline metrics, the base rates, and the 1000-vs-2000
comparison. The light-curve vs catalogue comparison is not re-checked here: it is
reproduced over 25 star-grouped splits by ablate_features.py (CLAUDE.md section 11).

    python verify_model.py
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix

FEATURES = ['depth', 'width', 'asymmetry', 'snr',
            'odd_even_diff', 'secondary_depth']
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
print(f"\nrecorded: [[223 78] [27 106]]   reproduced: [[{cm[0,0]} {cm[0,1]}] [{cm[1,0]} {cm[1,1]}]]")
print(f"match: {cm.tolist() == [[223, 78], [27, 106]]}")

rule("3. BASE RATES — which one is the right comparison?")
overall_fp = float((y == 0).mean())
test_fp = float((y_test == 0).mean())
print(f"FALSE POSITIVE share, whole dataset : {overall_fp:.4f}")
print(f"FALSE POSITIVE share, TEST set      : {test_fp:.4f}   <- the majority-class")
print(f"                                                        baseline to beat")
print(f"model accuracy                      : {(pred == y_test).mean():.4f}")
print("\nStar-grouped splitting means the test set's class mix differs from the")
print("whole dataset's. A baseline must be quoted against the split it is scored on.")

rule("4. LIGHT-CURVE vs CATALOGUE — see ablate_features.py")
print("Not re-checked here. The earlier one-column baseline (duration_over_period")
print("alone) was a subset of the model it was compared against, so it could not")
print("fail (CLAUDE.md 11.6). The honest comparison runs over 25 star-grouped")
print("splits in ablate_features.py, which produces the README's numbers.")

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
