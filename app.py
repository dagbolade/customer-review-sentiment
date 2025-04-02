import torch

torch._C._disable_torch_function_subclass = True

import pandas as pd
import streamlit as st
import torch

# Ensure class is available for joblib deserialization
from preprocessor import (
    TextPreprocessor,
    make_prediction,
)  # This is needed for the pipeline to deserialize

# pipeline = joblib.load("linear_svc_model.pkl")
# label_encoder = joblib.load("label_encoder.pkl")


# Get sentiment analysis
def get_sentiment(text):
    # Make predictions with the model and return the sentiment
    # prediction = pipeline.predict([text])
    # sentiment = label_encoder.inverse_transform(prediction)[0]

    sentiment = make_prediction(text)
    return sentiment


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
    df["Sentiment"] = df["Review"].apply(get_sentiment)
    st.write("### Analyzed Customer Reviews")
    st.dataframe(df)
else:
    st.write("Upload a CSV file to analyze customer reviews.")

# Manual text input for sentiment analysis
st.write("### Manual Sentiment Check")
text_input = st.text_area("Enter a customer review:")
if st.button("Check Sentiment"):
    if text_input:
        sentiment = get_sentiment(text_input)
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
