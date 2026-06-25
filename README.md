# 🎭 SentiScan – Real-Time Sentiment Analysis Dashboard

SentiScan is an end-to-end NLP (Natural Language Processing) project that analyzes movie reviews and predicts whether the sentiment is **Positive** or **Negative**. The project is trained on the IMDb 50K Movie Reviews dataset and deployed as a live Streamlit web application.

## 🚀 Live Demo

**Live App:**
https://ayanahmad2953-sentiscan-app-93waxq.streamlit.app/

## 📌 Project Overview

The goal of this project is to automatically classify text reviews using Machine Learning and Natural Language Processing techniques.

Users can enter any movie review, and the system instantly predicts the sentiment.

Example:

**Input:**

> This movie was absolutely amazing and entertaining.

**Output:**

> Positive 😊

---

## ✨ Features

* Real-time sentiment prediction
* NLP-based text preprocessing
* TF-IDF feature extraction
* Logistic Regression classifier
* Interactive Streamlit web interface
* Trained on IMDb 50K Reviews dataset
* Fast and lightweight deployment

---

## 🛠️ Tech Stack

### Programming Language

* Python

### Machine Learning & NLP

* Scikit-Learn
* NLTK

### Data Processing

* Pandas

### Model Serialization

* Joblib

### Deployment

* Streamlit

---

## 📂 Project Structure

```text
SentiScan/
│
├── app.py
├── train.py
├── requirements.txt
│
└── models/
    ├── sentiment_model.pkl
    └── vectorizer.pkl
```

---

## ⚙️ Machine Learning Pipeline

### 1. Data Collection

* IMDb 50K Movie Reviews Dataset

### 2. Text Preprocessing

* Lowercase conversion
* Removal of special characters
* Stopword removal

### 3. Feature Engineering

* TF-IDF Vectorization

### 4. Model Training

* Logistic Regression

### 5. Evaluation

* Accuracy Score
* Classification Report

### 6. Deployment

* Streamlit Cloud

---

## 📊 Dataset

Dataset Used:
IMDb 50K Movie Reviews Dataset

* Total Reviews: 50,000
* Positive Reviews: 25,000
* Negative Reviews: 25,000

---

## ▶️ Installation

Clone the repository:

```bash
git clone https://github.com/ayanahmad2953/SentiScan.git
```

Move into the project directory:

```bash
cd SentiScan
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the application:

```bash
streamlit run app.py
```

---

## 🎯 Future Improvements

* Sentiment confidence score
* Bulk CSV sentiment analysis
* Interactive charts and analytics
* Download prediction results
* Multi-class sentiment prediction (Positive, Negative, Neutral)
* Deep Learning (LSTM/BERT) implementation

---

## 👨‍💻 Author

**Ayan Ahmad**

B.Tech CSE Student | AI/ML Enthusiast

* GitHub: https://github.com/ayanahmad2953

---

## ⭐ If you found this project useful, please consider giving it a star.
