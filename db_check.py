from dotenv import load_dotenv
import os
import psycopg

load_dotenv()
url = os.environ['DATABASE_URL']

with psycopg.connect(url) as connect:
    isConnected = connect.execute('SELECT version()').fetchone()

print(isConnected)