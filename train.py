import pandas as pd
import re
import nltk
import joblib
import os

from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# Download stopwords (first time only)
nltk.download('stopwords')

# Load stopwords once
stop_words = set(stopwords.words('english'))

# Load dataset
print("Loading Dataset...")
df = pd.read_csv("IMDB Dataset.csv")

# Use only first 10000 reviews for faster training
df = df.head(10000)

print("Dataset Loaded Successfully")
print("Shape:", df.shape)
print("Columns:", df.columns)

# Text Cleaning Function
def clean_text(text):
    text = str(text).lower()

    # Remove special characters and numbers
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = text.split()

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    return " ".join(words)

# Clean reviews
print("Cleaning reviews...")
df["review"] = df["review"].apply(clean_text)

print("Cleaning Completed")

# Features and Labels
X = df["review"]
y = df["sentiment"]

# TF-IDF Vectorization
print("Creating TF-IDF vectors...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2)
)

X = vectorizer.fit_transform(X)

# Split Dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Model
print("Training Model...")

model = LogisticRegression(max_iter=1000)

model.fit(X_train, y_train)

# Prediction
pred = model.predict(X_test)

# Evaluation
accuracy = accuracy_score(y_test, pred)

print("\nAccuracy:", accuracy)
print("\nClassification Report:\n")
print(classification_report(y_test, pred))

# Create models folder
os.makedirs("models", exist_ok=True)

# Save model and vectorizer
joblib.dump(model, "models/sentiment_model.pkl")
joblib.dump(vectorizer, "models/vectorizer.pkl")

print("\nModel Saved Successfully!")
print("Files Created:")
print("models/sentiment_model.pkl")
print("models/vectorizer.pkl")