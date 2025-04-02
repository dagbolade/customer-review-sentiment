# 📝 Customer Review Sentiment Analysis

This project aims to build an AI-powered tool to analyse customer reviews, detect sentiment (Positive, Neutral, Negative), and visualise insights. It provides actionable feedback for businesses based on customer sentiment.

---

## 🚀 Project Overview

Key features include:

- 📊 **Sentiment Analysis:** Classifying reviews as positive, neutral, or negative.
- 🔑 **Key Insights:** Highlighting themes and trends from reviews.
- 📈 **Visualisation:** Interactive dashboard to view sentiment breakdown.
- 📧 **Feedback:** Providing actionable feedback for businesses.

---

## 🛠️ Tech Stack

| Component            | Tools Used                          |
| -------------------- | ----------------------------------- |
| **Data Analysis**    | Python, Pandas, NLTK, Excel,PowerBI |
| **Machine Learning** | Scikit-learn, Hugging Face          |
| **API Development**  | FastAPI, Flask                      |
| **Frontend**         | Streamlit,Flask                     |
| **Version Control**  | Git & GitHub                        |
| **Cloud Platform**   | AWS                                 |

## DATA ANALYSIS

This repository contains a Python script to clean and preprocess customer reviews from an Excel file ("cleaned_zappos_men.xlsx"). The script removes special characters, converts text to lowercase, removes stopwords, and applies stemming to prepare the data for sentiment analysis or machine learning tasks.

## EXCEL was also used for dropping off columns from the original dataset .

## **📌 Features**

- Converts text to **lowercase**
- Removes **punctuation and special characters**
- **Tokenizes** words using `nltk`
- Removes **stopwords**
- Applies **stemming** using `PorterStemmer`

---

## **🚀 Installation & Setup**

### ** Clone the Repository**

```bash
git clone https://github.com/SAMUELAY1/customer-review-sentiment.git
cd customer-review-sentiment
```

---

## Run Frontend Application

1. **Clone Your Fork:**

```bash
git clone https://github.com/your-username/customer-review-sentiment.git
cd customer-review-sentiment
```

2. **For Mac and Linux**

```bash
python3 -m venv venv
source ./venv/bin/activate
```

3. **For Windows**

```bash
python -m venv venv
venv\Scripts\activate
```

4. **Install Dependencies and Run App**

```bash
pip install -r requirements.txt
streamlit run app.py
```
