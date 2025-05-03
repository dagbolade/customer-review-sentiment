from flask import Flask, render_template, request, jsonify, send_from_directory, make_response
import requests
import pandas as pd
import os
import logging
import time
from flask_cors import CORS
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

# Initialize history as a global variable
history = []

# Configuration
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Backend API URLs
MODELS = {
    "LSTM": "http://127.0.0.1:8000/predict",
    "BERT": "http://127.0.0.1:8000/predict_bert"
}

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def detect_sentiment_columns(df):
    """Detect columns likely containing review text."""
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    possible_matches = [
        "review", "reviews", "comment", "comments", "text", 
        "feedback", "content", "opinion", "user_review"
    ]
    return [col for col in df.columns if any(match in col for match in possible_matches)]

def predict_sentiment_from_api(text, model):
    """Call the appropriate sentiment analysis API."""
    try:
        response = requests.post(MODELS[model], json={"text": text}, timeout=10)
        if response.status_code == 200:
            return response.json()
        logger.error(f"API request failed with status {response.status_code}: {response.text}")
        return {"sentiment": "Error", "confidence": 0.0, "error": response.text}
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to {model} API failed: {str(e)}")
        return {"sentiment": "Error", "confidence": 0.0, "error": str(e)}

@app.route("/")
def home():
     sentiment_colors = {
        "Positive": "green",
        "Negative": "red",
        "Neutral": "yellow",
        "Error": "gray"  
    }
     return render_template("index.html", models=MODELS.keys())

@app.route("/upload", methods=["POST"])
def handle_upload():
    if 'file' not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files['file']
    if not file or file.filename == '':
        return jsonify({"error": "No file selected"}), 400
    if not file.filename.lower().endswith('.csv'):
        return jsonify({"error": "Only CSV files supported"}), 400

    try:
        # Read CSV with multiple encoding attempts
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.stream.seek(0)
            df = pd.read_csv(file, encoding='latin1')

        sentiment_cols = detect_sentiment_columns(df)
        if not sentiment_cols:
            return jsonify({"error": "No review columns detected"}), 400

        model = request.form.get("model", "LSTM")
        for col in sentiment_cols:
            df[f"{col}_sentiment"] = df[col].apply(
                lambda x: predict_sentiment_from_api(str(x), model)["sentiment"]
            )
            df[f"{col}_confidence"] = df[col].apply(
                lambda x: f"{predict_sentiment_from_api(str(x), model)['confidence']*100:.2f}%"
            )

        filename = f"processed_{int(time.time())}_{secure_filename(file.filename)}"
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        df.to_csv(filepath, index=False, encoding='utf-8')

        return jsonify({
            "filename": filename,
            "download_url": f"/download/{filename}"
        })

    except Exception as e:
        logger.error(f"Error processing file: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download/<filename>")
def download_file(filename):
    try:
        return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        model = data.get("model", "LSTM")
        review_text = data.get("text", "")
        
        if not review_text.strip():
            return jsonify({"error": "Empty review text"}), 400

        result = predict_sentiment_from_api(review_text, model)
        
        # Add to history
        history.append({
            "review": review_text,
            "sentiment": result["sentiment"],
            "confidence": result["confidence"],
            "model": model
        })
        
        return jsonify(result)

    except Exception as e:
        logger.error(f"Error in analyze: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route("/download_history")
def download_history():
    if not history:
        return "No data to download!", 404
    
    # Convert history to CSV
    df = pd.DataFrame(history)
    csv_data = df.to_csv(index=False)
    
    # Create a response with CSV data
    response = make_response(csv_data)
    response.headers["Content-Disposition"] = "attachment; filename=sentiment_history.csv"
    response.headers["Content-Type"] = "text/csv"
    
    return response

@app.route("/cleanup", methods=["POST"])
def cleanup():
    try:
        for filename in os.listdir(UPLOAD_FOLDER):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)