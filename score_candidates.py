import joblib
import pandas as pd

bundle = joblib.load('transitcheck_model.joblib')
print(bundle.keys())
print(bundle['features'])
print(bundle['threshold'])

candidates = pd.read_csv('features_candidates.csv') # candidates is now a dataframe
candidates = candidates[candidates['error'].isna()] # keeping only the rows with no errors

# print(candidates)

cand_features = candidates[bundle['features']]
cand_results = bundle['model'].predict_proba(cand_features)

candidates['score'] = cand_results[:, 1]
candidates = candidates.sort_values('score', ascending=False)
candidates.to_csv('scored_candidates.csv', index=False)



