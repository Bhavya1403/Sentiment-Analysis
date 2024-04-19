import asyncio
import json
from twscrape import API, gather
from twscrape.logger import set_log_level
from tqdm.asyncio import tqdm
import os

async def main():
    api = API()
    
    # q = "Israel Palestine war since:2023-01-01 until:2023-05-31"
    q = "Global Warming since:2023-01-01 until:2023-05-31"
    
    # Load existing data if file exists
    if os.path.exists('data/tweets_main.json'):
        with open('data/tweets_main.json', 'r') as f:
            try:
                results = json.load(f)
            except json.JSONDecodeError:
                results = []
    else:
        results = []

    async for tweet in tqdm(api.search(q, limit=5000)):
        results.append({
            'id': tweet.id,
            'username': tweet.user.username,
            'content': tweet.rawContent
        })

    with open('data/tweets_main.json', 'w') as f:
        json.dump(results, f)

if __name__ == "__main__":
    asyncio.run(main())