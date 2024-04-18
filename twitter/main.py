import tweepy
import os
import dotenv

dotenv.load_dotenv()


client = tweepy.Client(
    bearer_token="AAAAAAAAAAAAAAAAAAAAAKagswEAAAAABA1xjIhzaC9Jae8orxIoEhUZZ50%3DDq8JAXOELa0WAbfMcMeSanrZ3uTnaOb2Jhxhw2wgoaPiXXmh2l"
)

response = client.search_recent_tweets(query="context:12.706083845846597632")

for tweet in response.data:
    print(tweet.id)
