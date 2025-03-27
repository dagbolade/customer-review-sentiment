from fastapi import FastAPI, HTTPException
import tensorflow as tf
import numpy as np
import pickle
from pydantic import BaseModel
from tensorflow.keras.preprocessing.sequence import pad_sequences
from fastapi.middleware.cors import CORSMiddleware
from transformers import TFBertForSequenceClassification, BertTokenizer
import json
import logging

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies customer reviews as Positive, Neutral or Negative",
    version="1.0"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load models
try:
    lstm_model = tf.keras.models.load_model("lstm_sentiment_model.keras")
    with open("tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    
    BERT_MODEL_PATH = "./saved_new_bert_model"
    bert_model = TFBertForSequenceClassification.from_pretrained(BERT_MODEL_PATH)
    bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_PATH)
    
    with open(f"{BERT_MODEL_PATH}/label_map.json", "r") as f:
        label_map = json.load(f)
except Exception as e:
    logging.error(f"Error loading models: {e}")
    raise RuntimeError("Failed to load models")

MAX_LEN = 100  # For LSTM model

class ReviewInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API"}

@app.get("/test_connection")
def test_connection():
    return {"status": "success", "message": "API is working"}

@app.post("/predict")
def classify_review_lstm(review: ReviewInput):
    try:
        seq = tokenizer.texts_to_sequences([review.text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prediction = lstm_model.predict(padded)[0][0]
        
        # Updated sentiment classification with neutral range
        if prediction >= 0.6:
            sentiment = "Positive"
        elif prediction <= 0.4:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {"sentiment": sentiment, "confidence": float(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_bert")
def classify_review_bert(review: ReviewInput):
    try:
        inputs = bert_tokenizer(review.text, return_tensors="tf", truncation=True, padding=True, max_length=512)
        outputs = bert_model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
        confidence = tf.reduce_max(probs, axis=-1).numpy()[0]
        sentiment = label_map[str(predicted_class)]
        return {"sentiment": sentiment, "confidence": float(confidence)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)