# Customer Review Sentiment Analysis

## Project Overview
A web application for sentiment analysis of customer reviews using advanced machine learning models (LSTM and BERT) that allows:
- Single review sentiment prediction
- Bulk CSV file sentiment analysis
- Downloading analysis history
- Switching between models

## Key Features
- 🤖 Two Machine Learning Models:
  - LSTM Model
  - BERT Model
- 📊 Sentiment Classification (Positive/Negative)
- 📁 CSV File Upload and Analysis
- 💾 Downloadable Analysis History

## Recent Improvements
- Added CSV file upload functionality
- Implemented sentiment analysis history download
- Integrated BERT model for enhanced prediction accuracy

## Prerequisites
- Python 3.8+
- pip

## Setup and Installation

### 1. Clone the Repository
```bash
git clone https://github.com/dagbolade/customer-review-sentiment.git
cd customer-review-sentiment
```

### 2. Create Virtual Environment

#### On Windows
```bash
python -m venv venv
venv\Scripts\activate
```

#### On macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## Running the Application

### Start Services

#### Terminal 1: FastAPI Sentiment Analysis Service
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### Terminal 2: Flask Web Application
```bash
python app.py
```

### Access the Application
- Web Interface: `http://localhost:5000`
- FastAPI Docs: `http://localhost:8000/docs`

## How to Use
1. Open web interface at `http://localhost:5000`
2. Choose LSTM or BERT model
3. Enter a review or upload a CSV file
4. View sentiment analysis results
5. Download processed files or analysis history

## Technical Details
- **Backend**: Flask, FastAPI
- **ML Frameworks**: TensorFlow, Transformers
- **Models**: 
  - LSTM Sentiment Classifier
  - BERT Sentiment Classifier

## Project Structure
```
customer-review-sentiment/
├── app.py               # Flask web application
├── fastapi_app.py       # ML model serving API
├── index.html           # Frontend interface
├── models/
│   ├── lstm_model.keras
│   └── saved_bert_model/
└── uploads/             # Processed files
```

## Contributing
1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push and create a Pull Request

## License
MIT license
