from dotenv import load_dotenv
import os
import psycopg

load_dotenv()
url = os.environ['DATABASE_URL']

sql = '''
CREATE TABLE IF NOT EXISTS candidates (
    kepoi_name TEXT PRIMARY KEY,
    kepid INTEGER NOT NULL,
    depth DOUBLE PRECISION,
    width DOUBLE PRECISION,
    asymmetry DOUBLE PRECISION,
    snr DOUBLE PRECISION,
    odd_even_diff DOUBLE PRECISION,
    secondary_depth DOUBLE PRECISION,
    score DOUBLE PRECISION NOT NULL
); 
'''

with psycopg.connect(url) as connect:
    table = connect.execute(sql)

sql = '''
CREATE TABLE IF NOT EXISTS verdicts (
    id INTEGER GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    kepoi_name TEXT NOT NULL REFERENCES candidates(kepoi_name),
    verdict TEXT NOT NULL CHECK (verdict IN ('CONFIRMED', 'FALSE POSITIVE')),
    remarks TEXT,
    reviewer TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
)
'''

with psycopg.connect(url) as connect:
    table = connect.execute(sql)


print(table)
