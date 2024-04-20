import pandas as pd
from transformers import pipeline, AutoTokenizer
from tqdm import tqdm
import os
import re
from langdetect import detect

os.environ['TRANSFORMERS_CACHE'] = './models'
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "10.3.0"

df = pd.read_csv('data/tweets_main.csv')

device = 0  # GPU 0

nlp = pipeline('sentiment-analysis', model="cardiffnlp/twitter-roberta-base-sentiment", device=device)

tokenizer = AutoTokenizer.from_pretrained("cardiffnlp/twitter-roberta-base-sentiment")

label_to_sentiment = {'LABEL_0': 'negative', 'LABEL_1': 'neutral', 'LABEL_2': 'positive'}

def remove_urls(text):
    return re.sub(r'https?://\S+|www\.\S+', '', text)

def is_english(text):
    try:
        return detect(text) == 'en'
    except:
        return False

def analyze_sentiment(text, nlp, tokenizer, max_length=500):
    tokens = tokenizer.tokenize(text)
    chunks = [tokens[i:i+max_length] for i in range(0, len(tokens), max_length)]
    sentiments = []
    for chunk in chunks:
        chunk_text = tokenizer.convert_tokens_to_string(chunk)
        label = nlp(chunk_text)[0]['label']
        sentiment = label_to_sentiment[label]
        sentiments.append(sentiment)
    if sentiments:
        return max(set(sentiments), key=sentiments.count)
    else:
        return 'neutral'  # default value

# Filter out non-English entries
df = df[df['content'].apply(is_english)]

tqdm.pandas(desc="Processing:")
df['content'] = df['content'].apply(remove_urls)
df['sentiment'] = df['content'].progress_apply(lambda x: analyze_sentiment(x, nlp, tokenizer))

df.to_csv('data/tweets_main_sentiment.csv', index=False)