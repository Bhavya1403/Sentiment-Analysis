import scrapy
import pandas as pd

class TwitterSpider(scrapy.Spider):
    name = "twitter_spider"
    start_urls = [
        "https://twitter.com/search?q=%23Oscars&src=typeahead_click&f=top"
    ]

    def start_requests(self):
        for url in self.start_urls:
            yield scrapy.Request(url, callback=self.parse)

    def parse(self, response):
        # Extract the tweets from the page
        tweets = response.css('.tweet-text::text').getall()
        
        # Print the tweets
        for tweet in tweets:
            print(tweet)
        
        # Find the URL of the next page of search results
        next_page = response.css('.next-page::attr(href)').get()
        
        # Check if there is a next page
        if next_page:
            # Send a request to the next page
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse_page)
    
    
    def parse_page(self, response):
        columns = ['Time', 'User', 'Tweet']
        dataList = []
        # Extract the tweets from the page
        tweets = response.css('.tweet-text::text').getall()
        
        # Print the tweets
        for tweet in tweets:
            #print(tweet)
            dataList.append([tweet.created_at, tweet.user.screen_name, tweet.text])
        
        # Find the URL of the next page of search results
        next_page = response.css('.next-page::attr(href)').get()
        
        # Check if there is a next page
        if next_page:
            # Send a request to the next page
            yield scrapy.Request(response.urljoin(next_page), callback=self.parse_page)
        else:
            df = pd.DataFrame(dataList, columns = columns)
            print(df)
            df.to_csv('tweets.csv')
            
            
    