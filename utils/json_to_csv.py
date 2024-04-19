import json
import pandas as pd

with open('data/tweets_main.json', 'r') as f:
    data = json.load(f)

df = pd.DataFrame(data)

df.to_csv('data/tweets_main.csv', index=False)