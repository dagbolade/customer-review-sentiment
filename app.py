import random

import pandas as pd
import streamlit as st


# Dummy data for sentiment analysis
def get_dummy_sentiment(text):
    sentiments = ["Positive", "Negative", "Neutral"]
    return random.choice(sentiments)


# Function to get sentiment color
def get_sentiment_color(sentiment):
    if sentiment == "Positive":
        return "#4CAF50"  # Green
    elif sentiment == "Negative":
        return "#F44336"  # Red
    else:
        return "#808080"  # Gray


# Streamlit UI
st.set_page_config(page_title="Customer Review Sentiment Analysis", layout="wide")
st.title("📊 Customer Review Sentiment Analysis")

st.markdown(
    """
    Welcome to the Customer Review Sentiment Analysis tool! 
    Upload a CSV file containing customer reviews, or enter text manually to analyze sentiment.
    """
)

# Sidebar for file upload
st.sidebar.header("Upload Reviews CSV")
file = st.sidebar.file_uploader("Upload a CSV file", type=["csv"])

if file:
    df = pd.read_csv(file)
    df["Sentiment"] = df["Review"].apply(get_dummy_sentiment)
    st.write("### Analyzed Customer Reviews")
    st.dataframe(df)
else:
    st.write("Upload a CSV file to analyze customer reviews.")

# Manual text input for sentiment analysis
st.write("### Manual Sentiment Check")
text_input = st.text_area("Enter a customer review:")
if st.button("Check Sentiment"):
    if text_input:
        sentiment = get_dummy_sentiment(text_input)
        color = get_sentiment_color(sentiment)
        st.markdown(
            f"<p style='color: {color}; font-size: 20px; font-weight: bold;'>Predicted Sentiment: {sentiment}</p>",
            unsafe_allow_html=True,
        )
    else:
        st.warning("Please enter a review to analyze.")

# Footer
st.markdown("---")
st.markdown("Developed for sentiment analysis of customer reviews.")
