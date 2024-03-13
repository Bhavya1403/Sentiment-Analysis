import tweepy
import configparser
import pandas as pd

#read configs
config = configparser.ConfigParser()
config.read('config.ini')

api_key = config['twitter']['api_key']
api_key_secret = config['twitter']['api_key_secret']
access_token = config['twitter']['access_token']
access_token_secret = config['twitter']['access_token_secret']

#authenticate 
auth = tweepy.OAuthHandler(api_key, api_key_secret)
auth.set_access_token(access_token,access_token_secret)

api = tweepy.API(auth)

# public_tweets = api.home_timeline()


#for tweet in public_tweets:
 #   print(tweet.text)
 
keywords = 'oscars'
limit = 1
 

columns = ['Time', 'User', 'Tweet']
data = []

tweets = tweepy.Cursor(api.search_tweets, q = keywords, count = limit, tweet_mode='extended').items(limit)

for tweet in tweets:
    data.append([tweet.created_at, tweet.user.screen_name, tweet.text])

df = pd.DataFrame(data, columns = columns)
print(df)

df.to_csv('tweets.csv')