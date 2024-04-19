import tweepy
import os
import dotenv

dotenv.load_dotenv()


client = tweepy.Client(
    bearer_token=os.getenv("BEARER_TOKEN"),
)

response = client.search_recent_tweets(query="context:12.706083845846597632")

for tweet in response.data:
    print(tweet.id)
