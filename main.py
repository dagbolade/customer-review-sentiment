import sys

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
import os
import requests
import hashlib
from typing import Optional
import zipfile
import io
import gdown

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies customer reviews as Positive, Neutral or Negative",
    version="1.0"
)

# Constants
LABEL_MAP = {
    "0": "NEGATIVE",
    "1": "NEUTRAL", 
    "2": "POSITIVE"
}
MAX_LEN = 100
# BERT_DIR = "bert_model"
# GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1ksq2NSoHj38Ak6DiV39RKzbA6pqn-L8y?usp=drive_link"
# MODEL_FOLDER_ID = "1ksq2NSoHj38Ak6DiV39RKzbA6pqn-L8y"

BERT_DIR = "bert_model"
DOWNLOADS = {
    f"{BERT_DIR}/config.json": "10hkabfCS6yTsfueGr6UhxTnGh54L-7l4",
    f"{BERT_DIR}/model.safetensors": "1kvGD4Qc4wOdYi3-w-izUeTju0erURX8N",
    f"{BERT_DIR}/special_tokens_map.json": "1mPKbDkRKL2mm1e29TyQfJ0ADaYHxgL22",
    f"{BERT_DIR}/tokenizer_config.json": "10tlVYpPtSaVm_svtEAQ9ADcL_K78j8HW",
    f"{BERT_DIR}/training_args.bin": "1YsWx1OpCLU97U1pa83k6QjOlKUnk5Cky",
    f"{BERT_DIR}/vocab.txt": "1LsqA0l3XooH16LVl6qmZ-cBdsP0fka-S",
}


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
lstm_model: Optional[tf.keras.Model] = None
tokenizer: Optional[object] = None
bert_model: Optional[TFBertForSequenceClassification] = None
bert_tokenizer: Optional[BertTokenizer] = None
label_map: Optional[dict] = None

# https://drive.google.com/file/d/10hkabfCS6yTsfueGr6UhxTnGh54L-7l4/view?usp=sharing
# https://drive.google.com/file/d/1kvGD4Qc4wOdYi3-w-izUeTju0erURX8N/view?usp=sharing
# https://drive.google.com/file/d/1mPKbDkRKL2mm1e29TyQfJ0ADaYHxgL22/view?usp=sharing
# https://drive.google.com/file/d/10tlVYpPtSaVm_svtEAQ9ADcL_K78j8HW/view?usp=sharing
# https://drive.google.com/file/d/1YsWx1OpCLU97U1pa83k6QjOlKUnk5Cky/view?usp=sharing
# https://drive.google.com/file/d/1LsqA0l3XooH16LVl6qmZ-cBdsP0fka-S/view?usp=sharing
class ReviewInput(BaseModel):
    text: str

def download_file_from_drive(file_id: str, dest_path: str):
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    try:
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(1024):
                    f.write(chunk)
            logger.info(f"Downloaded: {dest_path}")
        else:
            raise Exception(f"HTTP {response.status_code}")
    except Exception as e:
        logger.error(f"Download failed for {dest_path}: {e}")


def validate_model_files():
    """Validate all required model files exist"""
    required_files = {
        "lstm": ["lstm_sentiment_model.h5", "tokenizer.pkl"],
        "bert": [
            f"{BERT_DIR}/config.json",
            f"{BERT_DIR}/model.safetensors",
            f"{BERT_DIR}/tokenizer_config.json",
            f"{BERT_DIR}/vocab.txt",
            f"{BERT_DIR}/special_tokens_map.json",
            f"{BERT_DIR}/training_args.bin"
        ]

    }

    missing_files = []
    for model_type, files in required_files.items():
        for file in files:
            if not os.path.exists(file):
                missing_files.append(file)
                logger.warning(f"Missing: {file}")

    # Attempt download for each missing file
    for file in missing_files[:]:  # [:] to avoid modifying list during iteration
        file_id = DOWNLOADS.get(file)
        if file_id:
            download_file_from_drive(file_id, file)
            if not os.path.exists(file):
                logger.error(f"Still missing after download: {file}")


def load_models():
    """Load all ML models and components"""
    global lstm_model, tokenizer, bert_model, bert_tokenizer, label_map
    
    try:
        validate_model_files()
        
        # Load LSTM model and tokenizer
        try:
            lstm_model = tf.keras.models.load_model("lstm_sentiment_model.h5")
            with open("tokenizer.pkl", "rb") as f:
                tokenizer = pickle.load(f)
            logger.info("LSTM model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {str(e)}")
            raise

        # Load BERT model and tokenizer
        try:
            bert_model = TFBertForSequenceClassification.from_pretrained(BERT_DIR)
            bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
            label_map = LABEL_MAP
            logger.info("BERT model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load BERT model: {str(e)}")
            raise

    except Exception as e:
        logger.critical(f"Critical error loading models: {str(e)}")
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    """Load models when application starts"""
    logger.info("Starting up... Loading ML models")
    try:
        load_models()
        logger.info("All models loaded successfully")
    except Exception as e:
        logger.critical(f"Failed to load models during startup: {str(e)}")
        sys.exit(1)

@app.get("/")
def home():
    return {
        "message": "Sentiment Analysis API",
        "status": "running",
        "models_loaded": {
            "lstm": lstm_model is not None,
            "bert": bert_model is not None
        }
    }

@app.get("/health")
def health_check():
    return {
        "status": "OK" if lstm_model and bert_model else "ERROR",
        "lstm_loaded": lstm_model is not None,
        "bert_loaded": bert_model is not None,
        "ready": lstm_model is not None and bert_model is not None
    }

@app.get("/model_status")
def get_model_status():
    bert_config = {}
    config_path = f"{BERT_DIR}/config.json"
    
    if os.path.exists(config_path):
        try:
            with open(config_path, "r") as f:
                bert_config = json.load(f)
        except Exception as e:
            logger.error(f"Error reading BERT config: {str(e)}")
    
    return {
        "lstm_loaded": lstm_model is not None,
        "bert_loaded": bert_model is not None,
        "bert_config": bert_config,
        "label_map": label_map
    }

@app.post("/predict")
def classify_review_lstm(review: ReviewInput):
    if lstm_model is None or tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="LSTM model not loaded. Service unavailable."
        )
    
    try:
        seq = tokenizer.texts_to_sequences([review.text])
        padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
        prediction = lstm_model.predict(padded, verbose=0)[0][0]
        
        if prediction >= 0.6:
            sentiment = "Positive"
        elif prediction <= 0.4:
            sentiment = "Negative"
        else:
            sentiment = "Neutral"
            
        return {
            "text": review.text[:100] + "..." if len(review.text) > 100 else review.text,
            "sentiment": sentiment,
            "confidence": float(prediction),
            "model": "LSTM"
        }
    except Exception as e:
        logger.error(f"LSTM prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

@app.post("/predict_bert")
def classify_review_bert(review: ReviewInput):
    if bert_model is None or bert_tokenizer is None:
        raise HTTPException(
            status_code=503,
            detail="BERT model not loaded. Service unavailable."
        )
    
    try:
        inputs = bert_tokenizer(
            review.text,
            return_tensors="tf",
            truncation=True,
            padding=True,
            max_length=512
        )
        
        outputs = bert_model(inputs)
        probs = tf.nn.softmax(outputs.logits, axis=-1)
        predicted_class = tf.argmax(probs, axis=-1).numpy()[0]
        confidence = tf.reduce_max(probs, axis=-1).numpy()[0]
        
        sentiment = label_map.get(str(predicted_class), "Unknown")
        
        return {
            "text": review.text[:100] + "..." if len(review.text) > 100 else review.text,
            "sentiment": sentiment,
            "confidence": float(confidence),
            "model": "BERT"
        }
    except Exception as e:
        logger.error(f"BERT prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"BERT prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")