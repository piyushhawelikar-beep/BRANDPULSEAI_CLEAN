import streamlit as st
import joblib
import pickle
import re
import random
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import nltk
nltk.download("stopwords")
nltk.download("wordnet")
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# -------------------------------
# NLTK setup
# -------------------------------
nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#\w+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    words = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(words)

# -------------------------------
# Page config
# -------------------------------
st.set_page_config(page_title="BrandPulse AI", layout="wide")
st.title("BrandPulse AI — Tweet Sentiment Dashboard")

# -------------------------------
# Load models (cached)
# -------------------------------
@st.cache_resource
def load_models():
    lr_model = joblib.load("lr_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    lstm_model = load_model("lstm_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    with open("label_encoder.pkl", "rb") as f:
        label_encoder = pickle.load(f)
    return lr_model, vectorizer, lstm_model, tokenizer, label_encoder

@st.cache_data
def load_sample_tweets():
    df = pd.read_csv("data/cleaned_tweets.csv")
    return df.dropna(subset=['clean_text'])

lr_model, vectorizer, lstm_model, tokenizer, label_encoder = load_models()
sample_df = load_sample_tweets()

# -------------------------------
# Prediction functions
# -------------------------------
def predict_lr(clean_text):
    vec = vectorizer.transform([clean_text])
    pred = lr_model.predict(vec)[0]
    return label_encoder.inverse_transform([pred])[0]

def predict_lstm(clean_text):
    seq = tokenizer.texts_to_sequences([clean_text])
    padded = pad_sequences(seq, maxlen=50, padding='post', truncating='post')
    probs = lstm_model.predict(padded, verbose=0)
    return label_encoder.inverse_transform([np.argmax(probs)])[0]

# -------------------------------
# Section 1: Manual Check
# -------------------------------
st.header("Check a single tweet")
text_input = st.text_area("Enter a tweet:", placeholder="The service was amazing!")

if st.button("Analyze Sentiment"):
    if text_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        cleaned = clean_tweet(text_input)
        lr_pred = predict_lr(cleaned)
        lstm_pred = predict_lstm(cleaned)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Logistic Regression")
            st.write(f"**{lr_pred.capitalize()}**")
        with col2:
            st.subheader("LSTM")
            st.write(f"**{lstm_pred.capitalize()}**")

st.divider()

# -------------------------------
# Section 2: Simulated Live Feed
# -------------------------------
st.header("Simulated Live Tweet Feed")

if "feed" not in st.session_state:
    st.session_state.feed = []

if st.button("Stream next tweet"):
    row = sample_df.sample(1).iloc[0]
    pred = predict_lr(row['clean_text'])
    timestamp = datetime.now() - timedelta(minutes=random.randint(0, 1440))
    st.session_state.feed.append({
        "time": timestamp,
        "text": row['text'],
        "predicted_sentiment": pred
    })

if st.session_state.feed:
    feed_df = pd.DataFrame(st.session_state.feed).sort_values("time", ascending=False)
    st.dataframe(feed_df[["time", "text", "predicted_sentiment"]], use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Sentiment Distribution")
        counts = feed_df["predicted_sentiment"].value_counts()
        fig, ax = plt.subplots()
        ax.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
        ax.axis('equal')
        st.pyplot(fig)
        plt.close(fig)          # important to avoid the old bug

    with col2:
        st.subheader("Sentiment Trend (last 24h)")
        trend_df = feed_df.copy()
        trend_df["hour"] = trend_df["time"].dt.floor("h")
        trend_counts = trend_df.groupby(["hour", "predicted_sentiment"]).size().unstack(fill_value=0)
        st.line_chart(trend_counts)
else:
    st.info("Click 'Stream next tweet' to simulate incoming tweets.")
