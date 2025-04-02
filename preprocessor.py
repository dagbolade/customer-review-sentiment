import re
from collections import Counter

import torch
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import BaseEstimator, TransformerMixin
from torch import nn
from torch.nn import functional as F


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


class BPETokenizer:
    def __init__(self, vocab_size: int = 50, stop_word: str = "</w>"):
        self.vocab_size = vocab_size
        self.merges = {}  # Stores merge operations
        self.vocab = {"<PAD>": 0, "<UNK>": 1}  # Vocabulary with special tokens
        self.stop_word = stop_word

    def _get_pairs(self, word_list: list):
        """Gets the frequency pair from a wordlist"""
        pairs = Counter()
        for word in word_list:
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[(symbols[i], symbols[i + 1])] += 1
        return pairs

    def _merge_vocab(self, word_list: list, pair: tuple):
        """Merges the most frequent pair in the vocabulary"""
        new_word_list = []
        for word in word_list:
            i = 0
            new_word = []
            while i < len(word.split()):
                if (
                    i < len(word.split()) - 1
                    and word.split()[i] == pair[0]
                    and word.split()[i + 1] == pair[1]
                ):
                    new_word.append(pair[0] + pair[1])
                    i += 2
                else:
                    new_word.append(word.split()[i])
                    i += 1
            new_word_list.append(" ".join(new_word))
        return new_word_list

    def train(self, corpus: str):
        """Train BPE on a given corpus."""
        # Preprocess the corpus
        words = corpus.split()
        word_list = [" ".join(word) + " " + self.stop_word for word in words]

        # Initialize vocabulary with characters
        for word in word_list:
            for char in word.split():
                if char not in self.vocab:
                    self.vocab[char] = len(self.vocab)

        # Perform merges until we reach the desired vocab size
        while len(self.vocab) < self.vocab_size:
            pairs = self._get_pairs(word_list)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            self.merges[best_pair] = best_pair[0] + best_pair[1]

            # Add the new merged token to vocabulary
            if best_pair[0] + best_pair[1] not in self.vocab:
                self.vocab[best_pair[0] + best_pair[1]] = len(self.vocab)

            word_list = self._merge_vocab(word_list, best_pair)

        print("Final Vocabulary:", self.vocab)

    def encode(self, text: str, max_length: int = None):
        """Encodes text into subword tokens and returns the tensor indices"""
        # Split into words first
        words = text.split()
        all_tokens = []
        all_token_ids = []

        for word in words:
            # Initialize with characters for each word
            word_processed = " ".join(word) + " " + self.stop_word

            # Apply all possible merges
            while True:
                pairs = self._get_pairs([word_processed])
                if not pairs:
                    break
                # Find the merge that appears first in our merge list
                best_pair = None
                for pair in self.merges:
                    if pair in pairs:
                        best_pair = pair
                        break
                if not best_pair:
                    break
                word_processed = self._merge_vocab([word_processed], best_pair)[0]

            tokens = word_processed.split()
            token_ids = [self.vocab.get(token, self.vocab["<UNK>"]) for token in tokens]

            all_tokens.extend(tokens)
            all_token_ids.extend(token_ids)

        # Add padding if max_length is specified
        if max_length is not None:
            if len(all_token_ids) < max_length:
                all_token_ids += [self.vocab["<PAD>"]] * (
                    max_length - len(all_token_ids)
                )
            elif len(all_token_ids) > max_length:
                all_token_ids = all_token_ids[:max_length]

        return all_tokens, all_token_ids

    def decode(self, token_ids):
        """Decode indices back to text."""
        tokens = []
        for idx in token_ids:
            for token, token_id in self.vocab.items():
                if token_id == idx:
                    tokens.append(token)
                    break

        # Reconstruct words by splitting at stop_word tokens
        decoded_text = []
        current_word = []

        for token in tokens:
            if token.endswith(self.stop_word):
                current_word.append(token.replace(self.stop_word, ""))
                decoded_text.append("".join(current_word))
                current_word = []
            else:
                current_word.append(token)

        # Handle any remaining tokens (if no stop_word at end)
        if current_word:
            decoded_text.append("".join(current_word))

        return " ".join(decoded_text)

    def __len__(self):
        return len(self.vocab)

    def save(self, path: str):
        """Save the tokenizer to a JSON file"""
        import json

        save_data = {
            "vocab_size": self.vocab_size,
            "stop_word": self.stop_word,
            "merges": {f"{k[0]},{k[1]}": v for k, v in self.merges.items()},
            "vocab": self.vocab,
        }
        with open(path, "w") as f:
            json.dump(save_data, f, indent=2)

        print("Succesfully saved tokenzer")

    @classmethod
    def load(cls, path: str):
        """Load a tokenizer from a JSON file"""
        import json

        with open(path, "r") as f:
            data = json.load(f)

        tokenizer = cls(vocab_size=data["vocab_size"], stop_word=data["stop_word"])

        # Convert string keys back to tuples
        tokenizer.merges = {tuple(k.split(",")): v for k, v in data["merges"].items()}

        tokenizer.vocab = data["vocab"]
        return tokenizer


# Device agnostic code
device = "cuda" if torch.cuda.is_available() else "cpu"


class TextCNN(nn.Module):
    def __init__(
        self, vocab_size, embedding_dim, num_filters, filter_sizes, output_dim, dropout
    ):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # Multiple convolutional layers with different filter sizes
        self.convs = nn.ModuleList(
            [
                nn.Conv1d(
                    in_channels=embedding_dim, out_channels=num_filters, kernel_size=fs
                )
                for fs in filter_sizes
            ]
        )

        self.fc = nn.Linear(len(filter_sizes) * num_filters, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x = [batch size, seq len]
        embedded = self.embedding(x)
        # embedded = [batch size, seq len, emb dim]

        # Conv1d expects [batch size, channels, seq len]
        embedded = embedded.permute(0, 2, 1)
        # embedded = [batch size, emb dim, seq len]

        conved = [F.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, num_filters, seq len - filter_size + 1]

        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        # pooled_n = [batch size, num_filters]

        cat = self.dropout(torch.cat(pooled, dim=1))
        # cat = [batch size, num_filters * len(filter_sizes)]

        return self.fc(cat)


# Hyperparameters
MAX_LENGTH = 500
EMBEDDING_DIM = 500
NUM_FILTERS = 100
FILTER_SIZES = [3, 4, 5]
OUTPUT_DIM = 5  # Number of classes
DROPOUT = 0.5

# Tokenizer loading
tokenizer = BPETokenizer.load("tokenizer_config.json")

# Initialize model
model = TextCNN(
    vocab_size=len(tokenizer.vocab),
    embedding_dim=EMBEDDING_DIM,
    num_filters=NUM_FILTERS,
    filter_sizes=FILTER_SIZES,
    output_dim=OUTPUT_DIM,
    dropout=DROPOUT,
)

model.load_state_dict(torch.load("textcnn_model.pt"))


def make_prediction(text: str):
    _, encoded_text = tokenizer.encode(text, max_length=MAX_LENGTH)

    with torch.inference_mode():
        pred = model(torch.tensor(encoded_text).unsqueeze(dim=0))
        pred = pred.argmax(dim=1).cpu().item()

        if pred <= 1:
            return "Negative"
        elif pred == 2:
            return "Neutral"
        elif pred > 2:
            return "Positive"
        else:
            return "No prediction"
