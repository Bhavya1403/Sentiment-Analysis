from pygooglenews import GoogleNews
import bs4

# Define your search topic
topic = "Israel Palestine War"

# Instantiate the GoogleNews object
gn = GoogleNews(lang="en")  # Set language to English (default)

# Search for news articles matching the topic
news_results = gn.search(topic)
articles = news_results["entries"]
descriptions = []

for article in articles:
    summary = article["summary"]
    soup = bs4.BeautifulSoup(summary, "html.parser")
    anchor = soup.find("a")
    text = anchor.text.strip()
    descriptions.append(text)

print(descriptions)
