# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Write functions for data visualization
def visualize_data(data):
    # Sentiment distribution 
    sentiment_counts = data['sentiment'].value_counts()
    plt.figure(figsize=(10,6))
    sns.barplot(x=sentiment_counts.index,y=sentiment_counts.values, alpha=0.8)
    plt.title('Sentiment Distribution')
    plt.ylabel('Number of Occurrences', fontsize=12)
    plt.xlabel('Sentiment', fontsize=12)
    plt.show()
    all_text = ' '.join(data['content'])

    # Wordcloud 
    wordcloud = WordCloud(width=800, height=500, random_state=21, max_font_size=110).generate(all_text)
    plt.figure(figsize=(10, 7))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis('off')
    plt.show()
    
    # Sentiment length distribution
    data['text_length'] = data['content'].apply(len)
    plt.figure(figsize=(10,6))
    sns.histplot(data['text_length'], bins=50)
    plt.title('Text Length Distribution')
    plt.xlabel('Text Length')
    plt.ylabel('Frequency')
    plt.show()
    
    # Text length per sentiment
    plt.figure(figsize=(10,6))
    sns.boxplot(x='sentiment', y='text_length', data=data)
    plt.title('Text Length per Sentiment')
    plt.xlabel('Sentiment')
    plt.ylabel('Text Length')
    plt.show()
    
    # Word count distribution
    data['word_count'] = data['content'].apply(lambda x: len(str(x).split()))
    plt.figure(figsize=(10,6))
    sns.histplot(data['word_count'], bins=50)
    plt.title('Word Count Distribution')
    plt.xlabel('Word Count')
    plt.ylabel('Frequency')
    plt.show()
    
    # Top users posting
    user_counts = data['username'].value_counts().head(10)
    plt.figure(figsize=(10,6))
    sns.barplot(x=user_counts.index, y=user_counts.values, alpha=0.8)
    plt.title('Top Users Posting')
    plt.ylabel('Number of Posts', fontsize=12)
    plt.xlabel('User', fontsize=12)
    plt.xticks(rotation=45)
    plt.show()
    
    # Sentiment distribution per user
    top_users_data = data[data['username'].isin(user_counts.index)]
    plt.figure(figsize=(10,6))
    sns.countplot(x='username', hue='sentiment', data=top_users_data)
    plt.title('Sentiment Distribution per User')
    plt.xlabel('User')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()
    
    pass

# Main function to execute visualization
if __name__ == "__main__":
    # Load data
    filename = "../data/tweets_main_sentiment.csv"
    data = pd.read_csv(filename)

    # Call visualization function
    visualize_data(data)
