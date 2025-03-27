# Sentiment Analysis Web Application

## Overview
This project is a web application for sentiment analysis that uses both LSTM and BERT models to classify text reviews as positive or negative.

## Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

## Project Structure
```
sentiment-analysis-project/
│
├── app.py                   # Flask backend
├── fastapi_app.py           # FastAPI sentiment analysis service
├── index.html               # Frontend HTML file
├── lstm_sentiment_model.keras  # LSTM model file
├── tokenizer.pkl            # LSTM tokenizer
├── saved_new_bert_model/    # BERT model directory
│   ├── model_files
│   └── label_map.json
├── requirements.txt         # Project dependencies
└── README.md                # This documentation
```

## Installation Steps

### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd sentiment-analysis-project
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

#### Requirements File (requirements.txt)
```
flask
flask-cors
pandas
requests
tensorflow
fastapi
uvicorn
pydantic
transformers
```

### 3. Set Up Environment Variables (Optional but Recommended)
Create a `.env` file in the project root:
```
FLASK_APP=app.py
FLASK_ENV=development
```

### 4. Running the Application

#### Method 1: Without Virtual Environment
Open two terminal windows:

1. Start the FastAPI Sentiment Analysis Service:
```bash
python fastapi_app.py
```

2. In the second terminal, start the Flask Web Application:
```bash
python app.py
```

#### Method 2: Using Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Run FastAPI service
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000

# In another terminal, run Flask app
python app.py
```

### 5. Access the Application
Open a web browser and navigate to:
- Web Interface: `http://localhost:5000`
- FastAPI Docs: `http://localhost:8000/docs`

## Troubleshooting

### Common Issues
1. **Port Conflicts**: Ensure ports 5000 and 8000 are not in use by other applications.
2. **Model Loading**: Verify that model files (`lstm_sentiment_model.keras`, `tokenizer.pkl`, `saved_new_bert_model/`) are in the correct directory.
3. **Dependencies**: Make sure all required libraries are installed via `requirements.txt`.

### Logging
- Check the console output for any error messages
- Detailed logs are available in the application's logging configuration

## Features
- Upload CSV files for bulk sentiment analysis
- Real-time single review sentiment prediction
- Support for LSTM and BERT models
- Download processed files and analysis history

## Contributing
1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
