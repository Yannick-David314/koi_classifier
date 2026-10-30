from dotenv import load_dotenv
import numpy as np
import os
import pandas as pd
import psycopg


load_dotenv()
url = os.environ['DATABASE_URL']

candidates = pd.read_csv('scored_candidates.csv')
candidates = candidates.replace({np.nan: None})
value_list = []
for candidate in candidates.itertuples():
    value_tuple = (candidate.kepoi_name, candidate.kepid, candidate.depth, candidate.width, candidate.asymmetry, candidate.snr, candidate.odd_even_diff, candidate.secondary_depth, candidate.score)
    value_list.append(value_tuple)

sql = '''
INSERT INTO candidates (
    kepoi_name,
    kepid,
    depth,
    width,
    asymmetry,
    snr,
    odd_even_diff,
    secondary_depth,
    score
) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
ON CONFLICT (kepoi_name) DO UPDATE SET 
    kepid = EXCLUDED.kepid,
    depth = EXCLUDED.depth,
    width = EXCLUDED.width,
    asymmetry = EXCLUDED.asymmetry,
    snr = EXCLUDED.snr,
    odd_even_diff = EXCLUDED.odd_even_diff,
    secondary_depth = EXCLUDED.secondary_depth,
    score = EXCLUDED.score;
'''

with psycopg.connect(url) as connect:
    with connect.cursor() as cursor:
        loaded_table = cursor.executemany(sql, value_list)
# print(table)
