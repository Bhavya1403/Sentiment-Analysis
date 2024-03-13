from requests_html import HTMLSession
import pandas as pd

url = 'https://news.google.com/rss/search?q=israel%20palestine%20war&hl=en-IN&gl=IN&ceid=IN%3Aen'

s = HTMLSession()
r = s.get(url)

data =[]

for title in r.html.find('title'):
    data.append(title.text)

    
df = pd.DataFrame(data)
df.to_csv('googlenews.csv')
