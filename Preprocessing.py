import nltk
import re
import pickle
import pandas as pd
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

# Load saved TF-IDF vectorizer (ensure you save it in your notebook before using this)
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

def preprocess_text(text):
    """Preprocesses a given text by tokenizing, removing stopwords, punctuation, and lemmatizing."""
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    no_stop_words = [word for word in tokens if word.lower() not in stop_words]
    no_punctuation = [word for word in no_stop_words if word.isalnum()]
    lemmatizer = WordNetLemmatizer()
    lemmatized_tokens = [lemmatizer.lemmatize(word) for word in no_punctuation]
    return ' '.join(lemmatized_tokens)

def vectorize_text(text):
    """Converts preprocessed text into TF-IDF features using the trained vectorizer."""
    vectorizer = TfidfVectorizer()
    result = vectorizer.fit_transform(text)
    return result