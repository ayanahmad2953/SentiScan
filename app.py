import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("models/sentiment_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

# Page Title
st.set_page_config(page_title="SentiScan")

st.title("🎭 SentiScan")
st.subheader("Real-Time Sentiment Analysis Dashboard")

# Text Input
review = st.text_area("Enter a Review")

# Predict Button
if st.button("Analyze Sentiment"):

    if review.strip() != "":

        review_vector = vectorizer.transform([review])

        prediction = model.predict(review_vector)[0]

        if prediction == "positive":
            st.success("😊 Positive Sentiment")

        elif prediction == "negative":
            st.error("😞 Negative Sentiment")

        else:
            st.warning("😐 Neutral Sentiment")

    else:
        st.warning("Please enter some text.")