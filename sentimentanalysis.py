import re
import nltk
import pandas as pd

nltk.download('twitter_samples', quiet=True)
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

from nltk.corpus import twitter_samples, stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

stop_words = set(stopwords.words('english'))


def clean_tweet(text):
    text = str(text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#', '', text)
    text = re.sub(r'\bRT\b', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = text.lower()
    tokens = word_tokenize(text)
    tokens = [w for w in tokens if w not in stop_words and len(w) > 1]
    return " ".join(tokens)


positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

data = pd.DataFrame({
    "text": positive_tweets + negative_tweets,
    "label": ["Positive"] * len(positive_tweets) + ["Negative"] * len(negative_tweets)
})

data["cleaned"] = data["text"].apply(clean_tweet)

X_train, X_test, y_train, y_test = train_test_split(
    data["cleaned"], data["label"], test_size=0.2, random_state=42, stratify=data["label"]
)

vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

model = LogisticRegression(max_iter=1000)
model.fit(X_train_vec, y_train)

y_pred = model.predict(X_test_vec)

print("=" * 80)
print("MODEL EVALUATION")
print("=" * 80)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))


def predict_sentiment(text, neutral_threshold=0.6):
    cleaned = clean_tweet(text)
    vec = vectorizer.transform([cleaned])
    probs = model.predict_proba(vec)[0]
    classes = model.classes_
    max_prob = max(probs)
    pred_label = classes[probs.argmax()]

    if max_prob < neutral_threshold:
        return "Neutral", round(max_prob, 4)
    return pred_label, round(max_prob, 4)


if __name__ == "__main__":
    sample_tweets = [
        "I absolutely love the new update! Great job team",
        "This is the worst service I have ever experienced. Never again.",
        "The event starts at 5 PM today.",
        "Can't believe how bad the customer support is... so frustrating!!",
        "Thanks for the quick response, really appreciate it",
        "I'm not sure how I feel about this new policy.",
        "Best purchase I've made all year! Highly recommend",
        "Stock market dips slightly amid uncertainty.",
    ]

    results = []
    for tweet in sample_tweets:
        label, confidence = predict_sentiment(tweet)
        results.append({"tweet": tweet, "sentiment": label, "confidence": confidence})

    results_df = pd.DataFrame(results)
    print("\n" + "=" * 80)
    print("SAMPLE PREDICTIONS")
    print("=" * 80)
    pd.set_option('display.max_colwidth', None)
    print(results_df.to_string(index=False))

    results_df.to_csv("tweet_sentiment_predictions.csv", index=False)
    print("\nResults saved to 'tweet_sentiment_predictions.csv'")
