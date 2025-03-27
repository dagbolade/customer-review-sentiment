# Sentiment Analysis Web Application

## Project Overview
A comprehensive sentiment analysis web application using LSTM and BERT models for analyzing customer reviews.

## Recent Improvements
- Added CSV file upload functionality
- Implemented history download feature
- Integrated BERT model for enhanced sentiment analysis

## Development Setup

### Prerequisites
- Python 3.8+
- pip
- Git

### Local Development

#### 1. Clone the Repository
```bash
git clone <your-repository-url>
cd sentiment-analysis-project
```

#### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### Development Workflow

#### Running the Application
1. Start FastAPI Service:
```bash
uvicorn fastapi_app:app --host 127.0.0.1 --port 8000 --reload
```

2. Start Flask Web Application:
```bash
python app.py
```

### Key Development Features

#### CSV Upload
- Supports bulk sentiment analysis of CSV files
- Automatically detects review columns
- Adds sentiment and confidence columns to processed file

#### Sentiment Analysis Models
- LSTM Model
- BERT Model
- Easy model switching through web interface

#### History Tracking
- Real-time sentiment analysis tracking
- Downloadable CSV of analysis history

### Project Structure
```
sentiment-analysis-project/
│
├── app.py                   # Flask backend
├── fastapi_app.py           # FastAPI sentiment analysis service
├── index.html               # Frontend interface
├── uploads/                 # Uploaded CSV files
├── models/
│   ├── lstm_sentiment_model.keras
│   └── saved_new_bert_model/
├── history.csv              # Analysis history
└── requirements.txt         # Project dependencies
```

### Development Tools
- Flask for web backend
- FastAPI for ML model serving
- TensorFlow for machine learning models
- Pandas for data manipulation

## Contributing

### Setup for Contributors
1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Make your changes
4. Run tests
5. Commit and push
6. Create a Pull Request

### Contribution Guidelines
- Follow PEP 8 style guidelines
- Add tests for new features
- Update documentation
- Ensure no breaking changes in existing functionality

## Roadmap
- [ ] Add more sentiment analysis models
- [ ] Improve error handling
- [ ] Create more comprehensive testing
- [ ] Add visualization for sentiment analysis results

## Troubleshooting
- Ensure all dependencies are installed
- Check model files are in the correct directories
- Verify port availability (5000 for Flask, 8000 for FastAPI)

## License
[Specify your project's license]

## Contact
[Your contact information or project maintainer details]
```

