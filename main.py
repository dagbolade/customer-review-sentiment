from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import pickle
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
from transformers import TFBertForSequenceClassification, BertTokenizer
import json

from fastapi.middleware.cors import CORSMiddleware

import logging
logging.basicConfig(level=logging.DEBUG)
logging.debug("Loading LSTM model...")




# ✅ Initialize FastAPI app
app = FastAPI(title="Sentiment Analysis API", description="Classifies customer reviews as Positive or Negative",
              version="1.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from the Flask frontend
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all headers
)

# ✅ Load trained LSTM model
lstm_model = tf.keras.models.load_model("lstm_sentiment_model.keras")

# ✅ Load tokenizer for LSTM
with open("tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

# ✅ Load BERT model and tokenizer
BERT_MODEL_PATH = "./saved_new_bert_model"  # If the model is in the same directory as main.py
bert_model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)


# Load label mapping for BERT
with open(f"{BERT_MODEL_PATH}/label_map.json", "r") as f:
    label_map = json.load(f)

# ✅ Define max sequence length for LSTM (should match training)
MAX_LEN = 100

# ✅ Define request body format
class ReviewInput(BaseModel):
    text: str

# ✅ Define sentiment prediction function for LSTM
def predict_sentiment_lstm(review: str):
    # Preprocess input (tokenize + pad)
    seq = tokenizer.texts_to_sequences([review])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")

    # Make prediction (model outputs probability)
    prediction = lstm_model.predict(padded)[0][0]

    # Return sentiment based on threshold
    sentiment = "Positive" if prediction >= 0.5 else "Negative"
    return {"review": review, "sentiment": sentiment, "confidence": float(prediction)}

# ✅ Define sentiment prediction function for BERT
def predict_sentiment_bert(review: str):
    # Tokenize input
    inputs = bert_tokenizer(review, return_tensors="tf", truncation=True, padding=True, max_length=512)

    # Make prediction
    outputs = bert_model(inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1)
    predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
    confidence = tf.reduce_max(probs, axis=-1).numpy()[0]

    # Map predicted class to label
    sentiment = label_map[str(predicted_class)]
    return {"review": review, "sentiment": sentiment, "confidence": float(confidence)}

# ✅ API Route - Home
@app.get("/")
def home():
    return {"message": "Welcome to Sentiment Analysis API!"}

import logging

logging.basicConfig(level=logging.DEBUG)

@app.post("/predict_mock")
def predict_mock(review: ReviewInput):
    # Return a mock response without using the model
    return {
        "review": review.text,
        "sentiment": "Positive",  # Mock sentiment
        "confidence": 0.95        # Mock confidence
    }

@app.get("/test_connection")
def test_connection():
    return {"status": "success", "message": "Backend is working!"}

@app.post("/test_prediction")
def test_prediction(review: ReviewInput):
    try:
        # Just return the text without any prediction
        return {"text": review.text, "status": "received"}
    except Exception as e:
        logging.error(f"Error in test prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/test")
def test_endpoint(review: ReviewInput):
    return {"message": "Test successful", "received": review.text}
@app.post("/predict")
def classify_review_lstm(review: ReviewInput):
    try:
        # First, just echo back the input to test if basic routing works
        logging.debug(f"Received review: {review.text}")
        
        # Try loading the tokenizer separately to see if that's the issue
        try:
            seq = tokenizer.texts_to_sequences([review.text])
            logging.debug(f"Tokenizer converted text to sequence: {seq}")
        except Exception as e:
            logging.error(f"Error in tokenization: {e}")
            raise HTTPException(status_code=500, detail=f"Tokenization error: {str(e)}")
        
        # Try padding separately
        try:
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
            logging.debug(f"Padded sequence: {padded}")
        except Exception as e:
            logging.error(f"Error in padding: {e}")
            raise HTTPException(status_code=500, detail=f"Padding error: {str(e)}")
        
        # Try prediction separately
        try:
            prediction = lstm_model.predict(padded)[0][0]
            logging.debug(f"Prediction result: {prediction}")
        except Exception as e:
            logging.error(f"Error in prediction: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
        
        # Return the result
        sentiment = "Positive" if prediction >= 0.5 else "Negative"
        return {"review": review.text, "sentiment": sentiment, "confidence": float(prediction)}
    except Exception as e:
        logging.error(f"Overall error in classify_review_lstm: {e}")
        # Print a stack trace for more details
        import traceback
        logging.error(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_bert")
def classify_review_bert(review: ReviewInput):
    try:
        return predict_sentiment_bert(review.text)
    except Exception as e:
        logging.error(f"Error in classify_review_bert: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)