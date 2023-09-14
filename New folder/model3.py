import re
import streamlit as st
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.sparse import csr_matrix

st.title("Email Classification")

# Add a file upload widget
uploaded_file = st.file_uploader("Upload an email file (CSV, Excel, Txt, etc.)", type=["csv", "xlsx", "txt"])
def cleantext(text):
    text = re.sub(r"\n", " ", text)  # remove newline characters
    text = re.sub(r"[\d-]", "", text)  # remove digits and hyphens
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # remove special characters
    text = text.lower()
    return text

# Load your trained machine learning model (you should replace 'load_ml' with the actual loading code)
def load_ml(model_name):
    # Replace this with your actual model loading code
    return joblib.load('D:\\New folder\\model_svm.joblib')

model_name = 'Support Vector Machine'  # Replace with your model name

if uploaded_file is not None:
    # Read the uploaded file
    with open(uploaded_file.name, 'r') as file:
        email_text = file.read()

    # Split the text into individual emails (assuming they are separated by a common delimiter)
    emails = email_text.split('\n\n')

    # Clean and preprocess each email
    cleaned_emails = [cleantext(email) for email in emails[:100]]

    # Remove empty emails
    cleaned_emails = [email for email in cleaned_emails if email.strip()]
    email_df = pd.DataFrame({'text': cleaned_emails})

    # Load your machine learning model
    ml_model = load_ml(model_name)

    # Vectorize the cleaned emails using TF-IDF
    tfidf_vectorizer = TfidfVectorizer()
    X_tfidf = tfidf_vectorizer.fit_transform(cleaned_emails)

    # Predict using the model
    classification_results = ml_model.predict(X_tfidf)

    # Add classification results to the DataFrame
    email_df['classification'] = classification_results

    # Count the number of abusive and non-abusive emails
    total_observations = len(cleaned_emails)
    total_abusive = sum(classification_results)
    total_non_abusive = total_observations - total_abusive

    st.write(f"## Analysis of Uploaded Email Data")
    st.write(f"Total Observations: {total_observations}")
    st.write(f"Total Abusive Emails: {total_abusive}")
    st.write(f"Total Non-Abusive Emails: {total_non_abusive}")

    st.write(f"## Column Names:")
    st.write(email_df.columns)

    st.write(f"## First Five Rows:")
    st.write(email_df.head())

    # You can display additional information or visualizations here

# Define the cleantext function (you can customize it further)
def cleantext(text):
    text = re.sub(r"\n", " ", text)  # remove newline characters
    text = re.sub(r"[\d-]", "", text)  # remove digits and hyphens
    text = re.sub(r'[^A-Za-z0-9]+', ' ', text)  # remove special characters
    text = text.lower()
    return text