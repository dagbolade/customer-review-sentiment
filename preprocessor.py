import re

from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin


class TextPreprocessor(BaseEstimator, TransformerMixin):
    # Preprocessing function
    @staticmethod
    def preprocess_text(text):
        if not isinstance(text, str):
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove punctuation and special characters
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # Tokenization using the correct 'punkt' resource
        tokens = word_tokenize(text)

        # Remove stopwords
        stop_words = set(stopwords.words("english"))
        filtered_tokens = [word for word in tokens if word not in stop_words]

        # Stemming (Porter Stemmer)
        stemmer = PorterStemmer()
        stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]

        return " ".join(stemmed_tokens)  # Return cleaned text as a string

    # Fit function does nothing but must return a value for fit_fransform to work in the pipeline
    def fit(self, X, y=None):
        return self

    # Overwrite the transform function like in the sklearn documentation on TransformerMixin
    def transform(self, X):
        return [TextPreprocessor.preprocess_text(text) for text in X]
