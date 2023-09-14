import re
import streamlit as st
import pickle
import pdfplumber
import docx2txt
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer,TfidfTransformer
from sklearn.svm import SVC
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer  

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from scipy.sparse import csr_matrix

nltk.download('all')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

st.title("Email Classfication")

st.title('Abusive Email Classifier')

st.write("""
### Let your serene mind be more productive
and far away from someone spoiling it for you.
""")

model_name = st.sidebar.selectbox(
    'Select Model',
    ('Machine Learning', 'Ensemble Learning')
)

if model_name=='Machine Learning':
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Decision Tree', 'Multinomial Naive Bayes','K-Nearest Neighbors','Support Vector Machine')
)
    st.markdown(f"# {model_name} : {classifier_name}\n"
             "These baseline machine learning approaches are not very apt for NLP problems. They yield poor outputs as semantics is not taken into consideration.")

elif model_name=='Ensemble Learning':
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Random Forest','Bagging','AdaBoost','Gradient Boosting')
)
    st.markdown(f"# {model_name} : {classifier_name}\n"
             "Ensemble results are average since it is a collection of baseline models. The overal outcome adds a lot of generalization which is good. We recommed trying out in the deep learning model.")


user_input = st.text_area("Enter content to check for abuse", "")

#For ML models
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

#======>FUNCTION DEFINITIONS<======#

#Function to clean i/p data:
def cleantext(text):
    text = re.sub(r"\n", " ", text) #remove next "\n"
    text = re.sub(r"[\d-]", "", text) #remove all digits
    text = re.sub(r'[^A-Za-z0-9]+', " ", text) #remove all special charcters
    text = text.lower()
    return text



#Function to get sentiment scores
def sentiscore(text):
    sentialz = SentimentIntensityAnalyzer()
    analysis = sentialz.polarity_scores(text)
    return analysis["compound"]


#Function to predict for ML and Ensemble
def predictor_ml(text,model):
    cv = CountVectorizer()
    X_count = cv.fit_transform([text])
    tfidf_transformer = TfidfTransformer()
    X_tfid = tfidf_transformer.fit_transform(X_count)
    X = csr_matrix((X_tfid.data, X_tfid.indices, X_tfid.indptr), shape=(X_tfid.shape[0], 76071))
    return model.predict(X)

#Function to display output
@st.cache_data
def out(a):
    if a == 0:
        t1 = "<div> <span class='highlight blue'><span class='bold'>Non Abusive</span> </span></div>"
        st.markdown(t1,unsafe_allow_html=True)
    else:
        t2 = "<div> <span class='highlight red'><span class='bold'>Abusive</span> </span></div>"
        st.markdown(t2,unsafe_allow_html=True)


if st.button("Check for Abuse"):
    y = cleantext(user_input)
    st.write(f"## Sentiment Score {sentiscore(y)}")
    
    if model_name == 'Machine Learning' or model_name == 'Ensemble Learning':
        with st.spinner("Predicting..."):
            o = predictor_ml(y, load_ml(classifier_name))
        out(o)
