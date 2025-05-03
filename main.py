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
BERT_DIR = "bert_model"
GOOGLE_DRIVE_URL = "https://drive.google.com/drive/folders/1ksq2NSoHj38Ak6DiV39RKzbA6pqn-L8y?usp=drive_link"
MODEL_FOLDER_ID = "1ksq2NSoHj38Ak6DiV39RKzbA6pqn-L8y"

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

class ReviewInput(BaseModel):
    text: str

def download_bert_from_drive():
    """Download BERT model from Google Drive"""
    try:
        os.makedirs(BERT_DIR, exist_ok=True)
        
        # Using gdown to download the folder
        gdown.download_folder(
            id=MODEL_FOLDER_ID,
            output=BERT_DIR,
            quiet=False,
            use_cookies=False
        )
        
        logger.info("Successfully downloaded BERT model from Google Drive")
        return True
    except Exception as e:
        logger.error(f"Failed to download BERT model from Google Drive: {str(e)}")
        return False

def validate_model_files():
    """Validate all required model files exist"""
    required_files = {
        "lstm": ["lstm_sentiment_model.keras", "tokenizer.pkl"],
        "bert": [
            f"{BERT_DIR}/config.json",
            f"{BERT_DIR}/model.safetensors",
            f"{BERT_DIR}/tokenizer_config.json",
            f"{BERT_DIR}/vocab.txt"
        ]
    }
    
    missing_files = []
    for model_type, files in required_files.items():
        for file in files:
            if not os.path.exists(file):
                missing_files.append(file)
                logger.error(f"Missing file: {file}")
    
    if missing_files:
        # Try to download missing BERT files
        if any(f.startswith(BERT_DIR) for f in missing_files):
            logger.info("Attempting to download missing BERT model files...")
            if not download_bert_from_drive():
                raise FileNotFoundError(f"Missing required model files: {missing_files}")
            
            # Recheck after download attempt
            missing_files = []
            for file in required_files["bert"]:
                if not os.path.exists(file):
                    missing_files.append(file)
            
            if missing_files:
                raise FileNotFoundError(f"Still missing files after download attempt: {missing_files}")
        else:
            raise FileNotFoundError(f"Missing required model files: {missing_files}")

def load_models():
    """Load all ML models and components"""
    global lstm_model, tokenizer, bert_model, bert_tokenizer, label_map
    
    try:
        validate_model_files()
        
        # Load LSTM model and tokenizer
        try:
            lstm_model = tf.keras.models.load_model("lstm_sentiment_model.keras")
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
        # You might want to exit here if models are critical
        # import sys; sys.exit(1)

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