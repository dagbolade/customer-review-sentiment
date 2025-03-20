from fastapi import FastAPI
import tensorflow as tf
import numpy as np
import pickle
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ✅ Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API", description="Classifies customer reviews as Positive or Negative",
              version="1.0")

# ✅ Load trained LSTM model
model = tf.keras.models.load_model("lstm_sentiment_model.keras")

# ✅ Load tokenizer
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Define max sequence length (should match training)
MAX_LEN = 100


# ✅ Define request body format
class ReviewInput(BaseModel):
    text: str


# ✅ Define sentiment prediction function
def predict_sentiment(review: str):
    # Preprocess input (tokenize + pad)
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Make prediction (model outputs probability)
    prediction = model.predict(padded)[0][0]

    # Return sentiment based on threshold
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return {"review": review, "sentiment": sentiment, "confidence": float(prediction)}


# ✅ API Route - Home
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API!"}


# ✅ API Route - Predict Sentiment
@app.post("/predict")
def classify_review(review: ReviewInput):
    return predict_sentiment(review.text)

