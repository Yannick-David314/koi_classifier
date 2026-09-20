"""Train and save the final TransitCheck model.

Feature set: LIGHT-CURVE ONLY. Nothing read from the catalogue's transit-model fit
enters the model. See CLAUDE.md section 11 for why, and ablate_features.py for what
that choice costs.

The catalogue's period, epoch and duration are still *inputs to the pipeline* -- you
cannot fold a light curve without them -- but no catalogue quantity is handed to the
classifier as a feature. That is what makes "light-curve features vs catalogue
features" a genuine comparison rather than a comparison against a subset of itself.

    python train_model.py
"""

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_auc_score, average_precision_score)
from sklearn.inspection import permutation_importance

FEATURES = ['depth', 'width', 'asymmetry', 'snr', 'odd_even_diff', 'secondary_depth']
THRESHOLD = 0.25
OUTER_SEED = 42
MODEL_PATH = 'transitcheck_model.joblib'

df = pd.read_csv('features_2000.csv').drop_duplicates(subset='kepoi_name', keep='last')
ok = df[df['error'].isna()].copy()
X, y, groups = ok[FEATURES], (ok['disposition'] == 'CONFIRMED').astype(int), ok['kepid']

train_idx, test_idx = next(
    GroupShuffleSplit(1, test_size=0.2, random_state=OUTER_SEED).split(X, y, groups))

assert not (set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])), "stars leaked"

model = HistGradientBoostingClassifier(random_state=42)
model.fit(X.iloc[train_idx], y.iloc[train_idx])

proba = model.predict_proba(X.iloc[test_idx])[:, 1]
pred = (proba >= THRESHOLD).astype(int)
y_test = y.iloc[test_idx]

print(f"rows {len(ok)}   stars {ok['kepid'].nunique()}   "
      f"train {len(train_idx)}   test {len(test_idx)}\n")
print(classification_report(y_test, pred,
                            target_names=['FALSE POSITIVE', 'CONFIRMED'], digits=2))
cm = confusion_matrix(y_test, pred)
print(cm)

tn, fp, fn, tp = cm.ravel()
print(f"\nROC AUC            {roc_auc_score(y_test, proba):.4f}")
print(f"average precision  {average_precision_score(y_test, proba):.4f}")
print(f"\nas a review filter:")
print(f"  queue                 {len(y_test)} KOIs")
print(f"  auto-rejected         {tn + fn} ({(tn + fn) / len(y_test):.1%})")
print(f"  real planets lost     {fn} of {fn + tp} ({fn / (fn + tp):.1%})")
print(f"  left for humans       {fp + tp}, of which {tp} are planets")

imp = permutation_importance(model, X.iloc[test_idx], y_test,
                             n_repeats=30, random_state=42, scoring='accuracy')
print("\npermutation importances (test set):")
for i in np.argsort(imp.importances_mean)[::-1]:
    print(f"  {FEATURES[i]:<18} {imp.importances_mean[i]:+.4f} ± {imp.importances_std[i]:.4f}")

joblib.dump({'model': model, 'features': FEATURES, 'threshold': THRESHOLD}, MODEL_PATH)
print(f"\nsaved {MODEL_PATH}  (model + feature list + threshold)")
