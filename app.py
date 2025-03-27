from flask import Flask, render_template, request, jsonify, send_file
import requests
import pandas as pd
import os
from flask_cors import CORS
from io import StringIO

from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)
import logging


import time
import os

# At the start of your app
if not os.path.exists("uploads"):
    os.makedirs("uploads")


logging.basicConfig(level=logging.DEBUG)
logging.debug("Flask app initialized.")

@app.route("/cleanup", methods=["POST"])
def cleanup():
    try:
        for filename in os.listdir("uploads"):
            file_path = os.path.join("uploads", filename)
            if os.path.isfile(file_path):
                os.unlink(file_path)
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# FastAPI backend URLs
MODELS = {
    "LSTM": "http://127.0.0.1:8000/predict",
    "BERT": "http://127.0.0.1:8000/predict_bert"
}

# Store results for visualization
history = []

def detect_sentiment_columns(df):
    # More flexible column name detection
    df.columns = [str(col).strip().lower().replace(' ', '_') for col in df.columns]
    
    possible_matches = [
        "review", "reviews", "comment", "comments", "text", 
        "feedback", "content", "opinion", "user_review", 
        "customer_feedback", "message", "response",
        "assessment", "evaluation", "remarks", "notes"
    ]
    
    
    matched_columns = [
        col for col in df.columns
        if any(match in col for match in possible_matches)
    ]
    
    # Enhanced debug output
    print(f"\n[DEBUG] CSV Analysis:")
    print(f"Original Columns: {list(df.columns)}")
    print(f"Normalized Columns: {list(df.columns)}")
    print(f"Possible Matches: {possible_matches}")
    print(f"Matched Columns: {matched_columns}\n")
    
    return matched_columns

def predict_sentiment_from_api(text, model):
    response = requests.post(MODELS[model], json={"text": text})
    if response.status_code == 200:
        return response.json()["sentiment"]
    return "Error"

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "GET":
        return render_template("index.html", 
                            prediction=None, 
                            history=history, 
                            models=MODELS.keys(),
                            selected_model="LSTM")

    selected_model = request.form.get("model", "LSTM")

    # Handle file upload
    if "file" in request.files:
        file = request.files["file"]
        if file.filename == '':
            return jsonify({"error": "No file selected"}), 400
            
        if not file.filename.endswith('.csv'):
            return jsonify({"error": "Only CSV files supported"}), 400

        try:
            df = pd.read_csv(file)
            sentiment_cols = detect_sentiment_columns(df)
            if not sentiment_cols:
                return jsonify({"error": "No review columns found"}), 400

            for col in sentiment_cols:
            # Create temporary lists to store results
                sentiments = []
                confidences = []
                
                for text in df[col]:
                    # Get full prediction results
                    prediction = predict_sentiment_from_api(str(text), selected_model)
                    sentiments.append(prediction["sentiment"])
                    confidences.append(prediction["confidence"])
                
                # Add columns to DataFrame
                df[f"{col}_sentiment"] = sentiments
                df[f"{col}_confidence"] = confidences

            processed_name = f"processed_{int(time.time())}_{file.filename}"
            processed_path = os.path.join("uploads", processed_name)
            df.to_csv(processed_path, index=False)

            return jsonify({
                "status": "success",
                "download_url": f"/download/{processed_name}",
                "filename": processed_name
            })

        except Exception as e:
            return jsonify({"error": f"Processing failed: {str(e)}"}), 500

    # Handle text review
    elif "review" in request.form:
        review = request.form["review"]
        # ... (keep your existing text review handling code)

    return render_template("index.html",
                         prediction=None,
                         history=history,
                         models=MODELS.keys(),
                         selected_model=selected_model)
@app.route("/upload", methods=["POST"])
def handle_upload():
    # 1. Validate file existence
    if 'file' not in request.files:
        return jsonify({
            "status": "error",
            "message": "No file uploaded",
            "solution": "Please select a file before uploading"
        }), 400

    file = request.files['file']
    
    # 2. Validate file selection
    if file.filename == '':
        return jsonify({
            "status": "error",
            "message": "No file selected",
            "solution": "Please choose a CSV file to upload"
        }), 400

    # 3. Validate file extension
    if not file.filename.lower().endswith('.csv'):
        return jsonify({
            "status": "error",
            "message": "Invalid file type",
            "solution": "Only CSV files are supported",
            "received_file": file.filename
        }), 400

    try:
        # 4. Read CSV with multiple encoding attempts
        try:
            df = pd.read_csv(file, encoding='utf-8')
        except UnicodeDecodeError:
            file.stream.seek(0)
            df = pd.read_csv(file, encoding='latin1')

        # Get the original column names before normalization
        original_columns = list(df.columns)
        sentiment_cols = detect_sentiment_columns(df)
        
        if not sentiment_cols:
            return jsonify({
                "status": "error",
                "message": "No review columns detected",
                "details": {
                    "original_columns": original_columns,
                    "normalized_columns": list(df.columns),
                    "expected_columns": [
                        "review", "reviews", "comment", 
                        "text", "feedback", "opinion"
                    ],
                    "advice": "Rename your column to include 'review' or 'comment'"
                }
            }), 400

        # 5. Debug output
        print("\n=== CSV DEBUG INFORMATION ===")
        print(f"Columns detected: {list(df.columns)}")
        print("First 2 rows:")
        print(df.head(2).to_string())
        print("=======================\n")

        # 6. Detect sentiment columns with flexible matching
        df.columns = [str(col).strip().lower() for col in df.columns]
        sentiment_cols = [
            col for col in df.columns 
            if any(keyword in col for keyword in [
                'review', 'comment', 'text', 
                'feedback', 'opinion', 'response'
            ])
        ]

        if not sentiment_cols:
            return jsonify({
                "status": "error",
                "message": "No review columns detected",
                "details": {
                    "received_columns": original_columns,
                    "normalized_columns": list(df.columns),
                    "expected_patterns": possible_matches,
                    "suggestion": "Rename one column to include 'review', 'comment', or 'feedback'"
                }
            }), 400

        # 7. Process the CSV
        selected_model = request.form.get("model", "LSTM")
        
        # Process each sentiment column
        for col in sentiment_cols:
            # Create lists to store results
            sentiments = []
            confidences = []
            
            # Process each row
            for text in df[col]:
                prediction = predict_sentiment_from_api(str(text), selected_model)
                sentiments.append(prediction["sentiment"])
                confidences.append(prediction["confidence"])
            
            # Add to DataFrame
            df[f"{col}_sentiment"] = sentiments
            df[f"{col}_confidence"] = [f"{x*100:.2f}%" for x in confidences]  # Convert to percentage
            
            # Optional: Add numeric confidence for sorting/filtering
            df[f"{col}_confidence_value"] = confidences

        # 8. Save processed file
        # timestamp = int(time.time())
        # safe_filename = secure_filename(file.filename)
        processed_filename = f"processed_{int(time.time())}_{secure_filename(file.filename)}"
        processed_path = os.path.join("uploads", processed_filename)
        
        # Ensure the uploads directory exists
        os.makedirs("uploads", exist_ok=True)
        
        # Save the file - use 'w' mode and specify UTF-8 encoding
        # Save with explicit UTF-8 encoding
        try:
            df.to_csv(processed_path, index=False, encoding='utf-8')
        except Exception as e:
            return jsonify({
                "status": "error",
                "message": "Failed to save processed file",
                "error": str(e)
            }), 500
        
        # Return the response with proper headers
        return jsonify({
            "status": "success",
            "message": "File processed successfully",
            "filename": processed_filename,
            "download_url": f"/download/{processed_filename}",
            "processed_columns": [f"{col}_sentiment" for col in sentiment_cols]
        })

    except Exception as e:
        logging.error(f"Error in upload: {str(e)}", exc_info=True)
        return jsonify({
            "status": "error",
            "message": "Failed to save processed file",
            "error": str(e)
        }), 500

    except pd.errors.EmptyDataError:
        return jsonify({
            "status": "error",
            "message": "The CSV file is empty",
            "solution": "Upload a CSV file with data rows"
        }), 400
        
    except pd.errors.ParserError:
        return jsonify({
            "status": "error",
            "message": "Invalid CSV format",
            "solution": "Ensure your file is properly formatted as CSV"
        }), 400
        
    except Exception as e:
        return jsonify({
            "status": "error",
            "message": "An unexpected error occurred",
            "technical_details": str(e),
            "solution": "Try again or contact support"
        }), 500

# 📌 Route: Download Processed CSV
from flask import send_from_directory

@app.route("/download/<filename>")
def download_file(filename):
    try:
        return send_from_directory(
            directory="uploads",
            path=filename,
            as_attachment=True
        )
    except FileNotFoundError:
        return jsonify({"error": "File not found"}), 404



from flask import make_response

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
@app.route("/test_predict", methods=["POST"])
def test_predict():
    try:
        # Test basic prediction functionality
        response = requests.post(
            "http://127.0.0.1:8000/test_prediction", 
            json={"text": "This is a test review"},
            timeout=5
        )
        
        return jsonify({
            "status": "success" if response.status_code == 200 else "error",
            "response": response.json() if response.status_code == 200 else response.text
        })
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})
    
@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        data = request.json
        model = data.get("model", "LSTM")
        review_text = data.get("text", "")
        
        logging.debug(f"Received request for model: {model}, text: {review_text}")
        
        # Print the URL we're about to call
        api_url = MODELS[model]
        logging.debug(f"Calling FastAPI at: {api_url}")
        
        # Try a simple request first to check connectivity
        try:
            test_response = requests.get("http://127.0.0.1:8000/", timeout=5)
            logging.debug(f"Test connection to FastAPI: {test_response.status_code}")
        except Exception as e:
            logging.error(f"Test connection to FastAPI failed: {str(e)}")
        
        # Now try the actual API call
        try:
            response = requests.post(
                api_url, 
                json={"text": review_text},
                timeout=10
            )
            
            logging.debug(f"FastAPI response status: {response.status_code}")
            logging.debug(f"FastAPI response body: {response.text}")
            
            # Check response status
            if response.status_code == 200:
                result = response.json()
                
                # Add to history
                history.append({
                    "review": review_text, 
                    "sentiment": result["sentiment"], 
                    "confidence": result["confidence"], 
                    "model": model
                })
                
                # Save updated history
                df = pd.DataFrame(history)
                df.to_csv("history.csv", index=False)
                
                return jsonify(result)
            else:
                error_message = f"API Error: {response.status_code} - {response.text}"
                logging.error(error_message)
                return jsonify({"error": error_message}), 500
                
        except requests.exceptions.RequestException as e:
            logging.error(f"Error making request to FastAPI: {str(e)}")
            return jsonify({"error": f"Error connecting to API: {str(e)}"}), 500
    
    except Exception as e:
        logging.error(f"Error in analyze route: {str(e)}")
        import traceback
        logging.error(traceback.format_exc())
        return jsonify({"error": str(e)}), 500
@app.route("/test_backend")
def test_backend():
    try:
        # Test FastAPI connection
        response = requests.get("http://127.0.0.1:8000/test_connection", timeout=5)
        
        if response.status_code == 200:
            result = response.json()
            return jsonify({
                "status": "success", 
                "message": "Successfully connected to backend",
                "backend_response": result
            })
        else:
            return jsonify({
                "status": "error", 
                "message": f"Backend returned status code {response.status_code}"
            }), 500
    
    except requests.exceptions.RequestException as e:
        return jsonify({
            "status": "error", 
            "message": f"Could not connect to backend: {str(e)}"
        }), 500
    
# 📌 Helper function to predict sentiment from API
def predict_sentiment_from_api(text, model):
    response = requests.post(MODELS[model], json={"text": text})
    if response.status_code == 200:
        result = response.json()
        return {
            "sentiment": result["sentiment"],
            "confidence": result.get("confidence", 0.0)  # Default to 0.0 if not provided
        }
    return {"sentiment": "Error", "confidence": 0.0}


if __name__ == "__main__":
    app.run(debug=True, port=5000)