import pandas as pd
from transformers import pipeline
from tqdm import tqdm

df = pd.read_csv('data/tweets_main.csv')

nlp = pipeline('sentiment-analysis')

tqdm.pandas(desc="Processing:")
df['sentiment'] = df['content'].progress_apply(lambda x: nlp(x)[0]['label'])

df.to_csv('data/tweets_main_sentiment.csv', index=False)