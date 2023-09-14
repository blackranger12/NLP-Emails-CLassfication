import re
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix
from textblob import TextBlob

st.title("Email Classification")

st.title('Abusive Email Classifier')

# Add a file upload widget
uploaded_file = st.file_uploader("Upload an email file (CSV, Excel,Txt etc.)", type=["csv", "xlsx","txt"])

model_name = st.sidebar.selectbox(
    'Select Model',
    ('Machine Learning', 'Ensemble Learning')
)

if model_name == 'Machine Learning':
    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('Logistic Regression', 'Decision Tree', 'Multinomial Naive Bayes', 'K-Nearest Neighbors', 'Support Vector Machine')
    )
    st.markdown(f"# {model_name} : {classifier_name}\n")
elif model_name == 'Ensemble Learning':
    classifier_name = st.sidebar.selectbox(
        'Select classifier',
        ('Random Forest', 'Bagging', 'AdaBoost', 'Gradient Boosting')
    )
    st.markdown(f"# {model_name} : {classifier_name}\n")

user_input = st.text_area("Enter content to check for abuse", "")


@st.cache(allow_output_mutation=True)
def load_ml(model):
    if model == 'Logistic Regression':
        return joblib.load('D:\\New folder\\model_lgr.joblib')
    elif model == 'Multinomial Naive Bayes':
        return joblib.load('D:\\New folder\\model_NB.joblib')
    elif model == 'Random Forest':
        return joblib.load('D:\\New folder\\model_RF.joblib')
    elif model == 'Decision Tree':
        return joblib.load('D:\\New folder\\model_DT.joblib')
    elif model == 'K-Nearest Neighbors':
        return joblib.load('D:\\New folder\\model_knn.joblib')
    elif model == 'Support Vector Machine':
        return joblib.load('D:\\New folder\\model_svm.joblib')
    elif model == 'Bagging':
        return joblib.load('D:\\New folder\\model_bagg.joblib')
    elif model == 'AdaBoost':
        return joblib.load('D:\\New folder\\model_Adaboost.joblib')
    elif model == 'Gradient Boosting':
        return joblib.load('D:\\New folder\\model_GradientBoost.joblib')


# Function to clean input data
def cleantext(text):
    text = re.sub(r"\n", " ", text)  # remove next "\n"
    text = re.sub(r"[\d-]", "", text)  # remove all digits
    text = re.sub(r'[^A-Za-z0-9]+', " ", text)  # remove all special characters
    text = text.lower()
    return text

# Function to get sentiment scores
def sentiscore(text):
    sentialz = SentimentIntensityAnalyzer()
    analysis = sentialz.polarity_scores(text)
    return analysis["compound"]

# Function to predict for ML and Ensemble
def predictor_ml(text, model):
    text = cleantext(text)
    if not text:
        return 0
    cv = CountVectorizer()
    X_count = cv.fit_transform([text])
    tfidf_transformer = TfidfTransformer()
    X_tfid = tfidf_transformer.fit_transform(X_count)
    X = csr_matrix((X_tfid.data, X_tfid.indices, X_tfid.indptr), shape=(X_tfid.shape[0], 76071))
    return model.predict(X)
    predictions = []
   
# Function to display output
@st.cache_data
def out(a):
    if a == 0:
        t1 = "<div> <span class='highlight blue'><span class='bold'>Non Abusive</span> </span></div>"
        st.markdown(t1, unsafe_allow_html=True)
    else:
        t2 = "<div> <span class='highlight red'><span class='bold'>Abusive</span> </span></div>"
        st.markdown(t2, unsafe_allow_html=True)


if uploaded_file is not None:
    # Read the uploaded file
    with open(uploaded_file.name, 'r') as file:
        email_text = file.read()

    # Split the text into individual emails (assuming they are separated by a common delimiter)
    emails = email_text.split('\n\n')

    # Clean and preprocess each email
    cleaned_emails = [cleantext(email) for email in emails[:100]]
    classification_results = []

    # Remove empty emails
    cleaned_emails = [email for email in cleaned_emails if email.strip()]
    email_df = pd.DataFrame({'text': cleaned_emails})
    ml_model = load_ml(classifier_name)
     
    total_observations = len(cleaned_emails)
    total_abusive = sum(classification_results)
    total_non_abusive = total_observations - total_abusive
    # Total number of observations (emails)
    total_observations = len(cleaned_emails)

    email_df = pd.DataFrame({'text': cleaned_emails})

        
    email_df = pd.DataFrame({'target': [cleantext(email) for email in emails]})

    # Display the column names and the first five rows of the DataFrame
    st.write(f"## Column Names:")
    st.write(email_df.columns)

    st.write(f"## First Five Rows:")
    st.write(email_df.sample(5))
    # Perform classification to determine if emails are abusive or non-abusive (you should use your classification model here)

    st.write(f"## Analysis of Uploaded Email Data")
    st.write(f"Total Observations: {total_observations}")
   # st.write(f"Total Abusive Emails: {total_abusive}")
    #st.write(f"Total Non-Abusive Emails: {total_non_abusive}")
  

    st.write("### Example of the First Email:")
    if cleaned_emails:
        st.write(cleaned_emails[5])

if st.button("Check for Abuse"):
    y = cleantext(user_input)
    st.write(f"## Sentiment Score {sentiscore(y)}")

    if model_name == 'Machine Learning' or model_name == 'Ensemble Learning':
        with st.spinner("Predicting..."):
            o = predictor_ml(y, load_ml(classifier_name))
        out(o)
