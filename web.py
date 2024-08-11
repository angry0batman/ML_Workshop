import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# Load the saved model and vectorizer
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

def predict_news(text):
    # Preprocess and vectorize the input text
    text_vectorized = vectorizer.transform([text])
    # Predict using the loaded model
    prediction = model.predict(text_vectorized)
    return prediction[0]

# Streamlit app UI
st.title("Fake News Detection")

user_input = st.text_area("Enter news text here:")
if st.button("Predict"):
    if user_input:
        prediction = predict_news(user_input)
        if prediction == 1:
            st.write("The news is **FAKE**.")
        else:
            st.write("The news is **REAL**.")
    else:
        st.write("Please enter some text to predict.")

