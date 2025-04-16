import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import re

nltk.download("vader_lexicon")

def clean_tweet(text):
    text = re.sub(r"http\S+|www\S+|@\S+|#\S+", "", text)
    text = re.sub(r"[^\w\s]", "", text)
    return text.lower()

def analyze_sentiment(df, text_column="text"):
    sid = SentimentIntensityAnalyzer()
    df["cleaned"] = df[text_column].astype(str).apply(clean_tweet)
    df["sentiment_score"] = df["cleaned"].apply(lambda x: sid.polarity_scores(x)["compound"])
    df["sentiment_label"] = df["sentiment_score"].apply(lambda score:
        "positive" if score > 0.05 else "negative" if score < -0.05 else "neutral")
    return df

if __name__ == "__main__":
    df = pd.read_csv("data/2020-09-20 till 2020-10-13.csv")

    df["team"] = df["partition_1"]  # Add team column

    df = analyze_sentiment(df, text_column="text")
    df[["team", "sentiment_label"]].to_csv("data/team_sentiment_labeled.csv", index=False)

    print(df[["team", "sentiment_label"]].value_counts().head(10))
