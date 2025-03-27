# Sentiment Analysis Web Application

## Project Overview
A comprehensive sentiment analysis web application using LSTM and BERT models for analyzing customer reviews.

## Development Setup

### Prerequisites
- Python 3.8+
- pip
- venv (Python's built-in virtual environment tool)

### Local Development

#### 1. Clone the Repository
```bash
git clone https://github.com/dagbolade/customer-review-sentiment.git
cd customer-review-sentiment
```

#### 2. Create and Activate Virtual Environment

##### On Windows
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate
```

##### On macOS/Linux
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
# Ensure virtual environment is activated
pip install -r requirements.txt
```

### Running the Application

#### 1. Start FastAPI Service
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### 2. Start Flask Web Application (in another terminal)
```bash
# Ensure virtual environment is activated
python app.py
```

#### Deactivating Virtual Environment
When you're done working on the project:
```bash
deactivate
```

### Key Development Features
- LSTM and BERT sentiment analysis models
- CSV file upload functionality
- Sentiment analysis history tracking
- Downloadable analysis results

### Project Structure
```
customer-review-sentiment/
│
├── venv/                    # Virtual environment (git-ignored)
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

### Virtual Environment Best Practices
- Always activate the virtual environment before working on the project
- Install project-specific packages only within the virtual environment
- Use `requirements.txt` to track project dependencies
- Add `venv/` to your `.gitignore`

## Contributing
1. Fork the repository
2. Create a feature branch
   ```bash
   git checkout -b feature/your-feature-name
   ```
3. Activate your virtual environment
4. Make your changes
5. Run tests
6. Commit and push
7. Create a Pull Request

## Troubleshooting Virtual Environment
- Ensure you're using the correct Python version
- Verify the virtual environment is activated
- Reinstall dependencies if encountering package conflicts
- Use `pip freeze` to check installed packages

## Additional Notes
- Recommended Python version: 3.8+
- Always use a virtual environment for Python projects
- Keep your virtual environment and dependencies updated

## License
MIT License
