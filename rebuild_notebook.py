"""Rebuild transitCheck_training_ipynb.ipynb for the six-feature configuration
and execute it locally, writing real outputs back into the file.

Why this exists: the original notebook ran in Colab against the seven-feature set.
Its saved outputs now contradict the README, and three of its cells were commented
out so their claimed results could not be checked. Executing it in the same
environment as the rest of the repo fixes both problems at once.

nbclient is not installed, so cells are executed here in a shared namespace with
stdout captured. Every cell in this notebook is plain code with print output, so a
captured stream is a faithful representation.
"""

import io
import json
from contextlib import redirect_stdout

NOTEBOOK = 'transitCheck_training_ipynb.ipynb'

CELLS = [
    # 0 -- load. Replaces Colab's files.upload() with a direct local read.
    """import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.inspection import permutation_importance

# Reads the repo's own CSV. The Colab original began with files.upload(); that
# step is unnecessary outside Colab and is what made the notebook non-portable.
df = pd.read_csv('features_2000.csv')
print(len(df), 'raw')

df = df.drop_duplicates(subset='kepoi_name', keep='last')
print(len(df), 'after dedup')      # expect 2289

ok = df[df['error'].isna()].copy()
print(len(ok), 'successful rows')

# LIGHT-CURVE FEATURES ONLY.
# duration_over_period was removed and koi_impact was never added: both come from
# the catalogue's transit-model fit, and including either makes the project's
# central claim untestable. The ablation that settled this is in
# ablate_features.py; the reasoning is CLAUDE.md section 11.
FEATURES = ['depth', 'width', 'asymmetry', 'snr',
            'odd_even_diff', 'secondary_depth']

X = ok[FEATURES]
y = (ok['disposition'] == 'CONFIRMED').astype(int)
groups = ok['kepid']""",

    # 1 -- splits, unchanged from the original
    """outer = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(outer.split(X, y, groups))

inner = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=7)
a, b = next(inner.split(X.iloc[train_idx], y.iloc[train_idx], groups.iloc[train_idx]))
tr, va = train_idx[a], train_idx[b]

print(f"train {len(train_idx)}  (inner fit {len(tr)} / validation {len(va)})")
print(f"test  {len(test_idx)}")
print("star-disjoint:", not (set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])))""",

    # 2 -- threshold sweep on the validation split
    """# Untuned, matching the final model below. The threshold is chosen HERE, on a
# validation split carved out of the training stars -- the test set is not touched
# until the last cell. Grid search in the next cell; it was not adopted.

m = HistGradientBoostingClassifier(random_state=42).fit(X.iloc[tr], y.iloc[tr])
pv = m.predict_proba(X.iloc[va])[:, 1]

for t in [0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]:
    p = (pv >= t).astype(int)
    tp = ((p == 1) & (y.iloc[va] == 1)).sum()
    fp = ((p == 1) & (y.iloc[va] == 0)).sum()
    fn = ((p == 0) & (y.iloc[va] == 1)).sum()
    print(f"{t}: recall {tp/(tp+fn):.2f}, precision {tp/(tp+fp):.2f}")""",

    # 3 -- grid search, now RUN rather than commented out
    """# Previously commented out, so its result could not be checked. Now live, because
# an unverifiable claim in a notebook is worse than 50 seconds of compute.
from sklearn.model_selection import GridSearchCV, GroupKFold
import numpy as np

grid = {
    'learning_rate': [0.05, 0.1],
    'max_leaf_nodes': [15, 31],
    'min_samples_leaf': [20, 40],
    'l2_regularization': [0.0, 1.0],
    'max_iter': [100, 300],
}

search = GridSearchCV(
    HistGradientBoostingClassifier(random_state=42),
    grid,
    scoring='average_precision',
    cv=GroupKFold(n_splits=5),
    n_jobs=2,
)
search.fit(X.iloc[train_idx], y.iloc[train_idx], groups=groups.iloc[train_idx])

default_score = search.cv_results_['mean_test_score'][
    np.where((search.cv_results_['param_learning_rate'] == 0.1) &
             (search.cv_results_['param_max_leaf_nodes'] == 31) &
             (search.cv_results_['param_min_samples_leaf'] == 20) &
             (search.cv_results_['param_l2_regularization'] == 0.0) &
             (search.cv_results_['param_max_iter'] == 100))[0][0]
]

r = search.cv_results_
best_i = search.best_index_
default_i = int(np.where((r['param_learning_rate'] == 0.1) &
                         (r['param_max_leaf_nodes'] == 31) &
                         (r['param_min_samples_leaf'] == 20) &
                         (r['param_l2_regularization'] == 0.0) &
                         (r['param_max_iter'] == 100))[0][0])

gap = r['mean_test_score'][best_i] - r['mean_test_score'][default_i]
pooled = np.sqrt(r['std_test_score'][best_i]**2 + r['std_test_score'][default_i]**2)

print(f"{len(r['params'])} configurations")
print(f"best params:   {search.best_params_}")
print(f"best CV:       {r['mean_test_score'][best_i]:.4f} ± {r['std_test_score'][best_i]:.4f}")
print(f"default CV:    {r['mean_test_score'][default_i]:.4f} ± {r['std_test_score'][default_i]:.4f}")
print(f"gap:           {gap:+.4f}   pooled fold std {pooled:.4f}   ratio {gap/pooled:.2f}")
print()
print("Not adopted. The gap is a quarter of the fold-to-fold spread, so it is")
print("not distinguishable from which stars happened to land in which fold.")""",

    # 4 -- the final model
    """THRESHOLD = 0.25   # <- chosen in the threshold sweep two cells above

final = HistGradientBoostingClassifier(random_state=42).fit(X.iloc[train_idx], y.iloc[train_idx])
probs = final.predict_proba(X.iloc[test_idx])[:, 1]
preds = (probs >= THRESHOLD).astype(int)

print(classification_report(y.iloc[test_idx], preds,
                            target_names=['FALSE POSITIVE', 'CONFIRMED']))
print(confusion_matrix(y.iloc[test_idx], preds))

imp = permutation_importance(final, X.iloc[test_idx], y.iloc[test_idx],
                             n_repeats=30, random_state=42, scoring='accuracy')
for n, mn, sd in sorted(zip(FEATURES, imp.importances_mean, imp.importances_std),
                        key=lambda t: -t[1]):
    print(f"{n:22s} {mn:+.4f} ± {sd:.4f}")""",

    # 5 -- class balance sanity check, previously commented out
    """# Star-grouped splitting means the class mix differs between splits. Worth
# printing, because a baseline quoted against the wrong split is misleading --
# see CLAUDE.md 11.6.
print(f"CONFIRMED rate, whole dataset: {y.mean():.4f}")
print(f"CONFIRMED rate, validation:    {y.iloc[va].mean():.4f}")
print(f"CONFIRMED rate, test:          {y.iloc[test_idx].mean():.4f}")
print(f"majority-class accuracy on test: {1 - y.iloc[test_idx].mean():.4f}")""",

    # 6 -- save
    """import joblib

joblib.dump({
    'model': final,
    'features': FEATURES,
    'threshold': THRESHOLD,
}, 'transitcheck_model.joblib')

print('saved transitcheck_model.joblib')
print('  features :', FEATURES)
print('  threshold:', THRESHOLD)""",
]


def execute(sources):
    """Run each cell in a shared namespace, returning nbformat output lists."""
    namespace = {'__name__': '__main__'}
    outputs = []
    for i, src in enumerate(sources):
        buf = io.StringIO()
        with redirect_stdout(buf):
            exec(compile(src, f'<cell {i}>', 'exec'), namespace)
        text = buf.getvalue()
        outputs.append(
            [{'output_type': 'stream', 'name': 'stdout',
              'text': text.splitlines(keepends=True)}] if text else []
        )
        print(f"  cell {i}: ok ({len(text.splitlines())} output lines)")
    return outputs


print("executing cells...")
outs = execute(CELLS)

notebook = {
    'cells': [
        {'cell_type': 'code', 'execution_count': i + 1, 'metadata': {},
         'outputs': outs[i], 'source': src.splitlines(keepends=True)}
        for i, src in enumerate(CELLS)
    ],
    'metadata': {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python',
                       'name': 'python3'},
        'language_info': {'name': 'python'},
    },
    'nbformat': 4,
    'nbformat_minor': 0,
}

with io.open(NOTEBOOK, 'w', encoding='utf-8') as fh:
    json.dump(notebook, fh, indent=1, ensure_ascii=False)
    fh.write('\n')

print(f"\nwrote {NOTEBOOK}: {len(CELLS)} cells, all executed")
