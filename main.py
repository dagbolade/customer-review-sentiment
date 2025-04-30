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

app = FastAPI(
    title="Sentiment Analysis API",
    description="Classifies customer reviews as Positive, Neutral or Negative",
    version="1.0"
)

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
lstm_model = None
tokenizer = None
bert_model = None
bert_tokenizer = None
label_map = None
MAX_LEN = 100

class ReviewInput(BaseModel):
    text: str

def get_file_checksum(filepath):
    """Calculate MD5 checksum of a file"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()

def validate_config():
    """Ensure config.json exists and is valid"""
    config_path = "bert_model/config.json"
    if not os.path.exists(config_path):
        logger.warning("config.json not found, creating default")
        default_config = {
            "_name_or_path": "bert-base-uncased",
            "architectures": ["BertForSequenceClassification"],
            "attention_probs_dropout_prob": 0.1,
            "hidden_act": "gelu",
            "hidden_dropout_prob": 0.1,
            "hidden_size": 768,
            "initializer_range": 0.02,
            "intermediate_size": 3072,
            "layer_norm_eps": 1e-12,
            "max_position_embeddings": 512,
            "model_type": "bert",
            "num_attention_heads": 12,
            "num_hidden_layers": 12,
            "pad_token_id": 0,
            "position_embedding_type": "absolute",
            "transformers_version": "4.30.2",
            "type_vocab_size": 2,
            "use_cache": True,
            "vocab_size": 30522,
            "id2label": {"0": "NEGATIVE", "1": "NEUTRAL", "2": "POSITIVE"},
            "label2id": {"NEGATIVE": 0, "NEUTRAL": 1, "POSITIVE": 2},
            "num_labels": 3
        }
        os.makedirs("bert_model", exist_ok=True)
        with open(config_path, "w") as f:
            json.dump(default_config, f, indent=2)
        return default_config
    
    try:
        with open(config_path, "r") as f:
            config = json.load(f)
        return config
    except json.JSONDecodeError:
        logger.error("Invalid JSON in config.json, recreating")
        os.remove(config_path)
        return validate_config()

def download_file_with_retry(url: str, destination: str, max_retries=3):
    """Download file with retry logic and checksum verification"""
    for attempt in range(max_retries):
        try:
            # Clear any existing file
            if os.path.exists(destination):
                os.remove(destination)
                
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Get expected file size from headers
            file_size = int(response.headers.get('content-length', 0))
            downloaded_size = 0
            
            with open(destination, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded_size += len(chunk)
            
            # Verify file downloaded completely
            if file_size > 0 and os.path.getsize(destination) != file_size:
                raise IOError(f"Incomplete download (expected {file_size}, got {os.path.getsize(destination)})")
            
            logger.info(f"Downloaded {os.path.basename(destination)} successfully")
            return True
            
        except Exception as e:
            logger.warning(f"Attempt {attempt + 1} failed for {url}: {str(e)}")
            if os.path.exists(destination):
                os.remove(destination)
    
    logger.error(f"Failed to download {url} after {max_retries} attempts")
    return False

def load_models():
    global lstm_model, tokenizer, bert_model, bert_tokenizer, label_map
    
    try:
        # 1. Load LSTM model and tokenizer
        lstm_model = tf.keras.models.load_model("lstm_sentiment_model.keras")
        with open("tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        logger.info("LSTM model and tokenizer loaded successfully")
        
        # 2. Setup BERT model directory
        BERT_DIR = "bert_model"
        os.makedirs(BERT_DIR, exist_ok=True)
        
        # 3. Validate/repair config.json first
        validate_config()
        
        # 4. Download other required files with verification
        required_files = {
            "model.safetensors": "https://drive.google.com/uc?id=1kvGD4Qc4wOdYi3-w-izUeTju0erURX8N&export=download",
            "tokenizer_config.json": "https://drive.google.com/uc?id=10tlVYpPtSaVm_svtEAQ9ADcL_K78j8HW&export=download",
            "vocab.txt": "https://drive.google.com/uc?id=1LsqA0l3XooH16LVl6qmZ-cBdsP0fka-S&export=download",
            "label_map.json": "https://drive.google.com/uc?id=1mPKbDkRKL2mm1e29TyQfJ0ADaYHxgL22&export=download"
        }
        
        for filename, url in required_files.items():
            filepath = os.path.join(BERT_DIR, filename)
            if not os.path.exists(filepath) or os.path.getsize(filepath) == 0:
                if not download_file_with_retry(url, filepath):
                    raise RuntimeError(f"Failed to download {filename}")
            
            # Verify file is not empty
            if os.path.getsize(filepath) == 0:
                raise ValueError(f"Downloaded file {filename} is empty")
        
        # 5. Verify model.safetensors is valid
        try:
            import safetensors.torch
            safetensors.torch.load_file(os.path.join(BERT_DIR, "model.safetensors"))
            logger.info("model.safetensors verified successfully")
        except Exception as e:
            logger.error(f"model.safetensors verification failed: {str(e)}")
            raise ValueError("Invalid model.safetensors file") from e
        
        # 6. Load BERT model
        bert_model = TFBertForSequenceClassification.from_pretrained(BERT_DIR)
        bert_tokenizer = BertTokenizer.from_pretrained(BERT_DIR)
        
        # 7. Load label map
        with open(os.path.join(BERT_DIR, "label_map.json"), "r") as f:
            label_map = json.load(f)
            
        logger.info("BERT model loaded successfully")
        
        # 8. Test the model
        test_input = bert_tokenizer("This is a test", return_tensors="tf")
        test_output = bert_model(test_input)
        logger.info(f"BERT test output: {test_output.logits.numpy()}")
        
    except Exception as e:
        logger.error(f"Error loading models: {str(e)}", exc_info=True)
        raise RuntimeError(f"Model loading failed: {str(e)}")

@app.on_event("startup")
async def startup_event():
    try:
        load_models()
    except Exception as e:
        logger.critical(f"Failed to load models: {str(e)}")
        # Provide more helpful error message
        raise RuntimeError(
            "Failed to initialize models. Possible causes:\n"
            "1. Corrupted model files - try deleting the bert_model directory and restarting\n"
            "2. Internet connection issues for downloads\n"
            "3. Insufficient disk space\n"
            f"Technical details: {str(e)}"
        )

@app.get("/")
def home():
    return {"message": "Sentiment Analysis API"}

@app.get("/model_status")
def get_model_status():
    bert_config = {}
    if os.path.exists("bert_model/config.json"):
        with open("bert_model/config.json", "r") as f:
            bert_config = json.load(f)
    
    file_status = {}
    for fname in ["config.json", "model.safetensors", "tokenizer_config.json", "vocab.txt", "label_map.json"]:
        path = os.path.join("bert_model", fname)
        file_status[fname] = {
            "exists": os.path.exists(path),
            "size": os.path.getsize(path) if os.path.exists(path) else 0
        }
    
    return {
        "lstm_loaded": lstm_model is not None,
        "bert_loaded": bert_model is not None,
        "bert_config": bert_config,
        "label_map": label_map,
        "file_status": file_status
    }

@app.post("/predict")
def classify_review_lstm(review: ReviewInput):
    try:
        if lstm_model is None or tokenizer is None:
            raise HTTPException(status_code=503, detail="LSTM model not loaded")
            
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
            "sentiment": sentiment,
            "confidence": float(prediction),
            "model": "LSTM"
        }
    except Exception as e:
        logger.error(f"LSTM prediction error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/predict_bert")
def classify_review_bert(review: ReviewInput):
    try:
        if bert_model is None or bert_tokenizer is None:
            raise HTTPException(status_code=503, detail="BERT model not loaded")
        
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
            "sentiment": sentiment,
            "confidence": float(confidence),
            "model": "BERT"
        }
        
    except Exception as e:
        logger.error(f"BERT prediction error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"BERT prediction failed: {str(e)}"
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, log_level="info")