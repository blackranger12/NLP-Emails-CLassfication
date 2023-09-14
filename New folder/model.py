import re
import streamlit as st



import pickle
import pdfplumber
import docx2txt
import pandas as pd
import joblib
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
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
    ('Machine Learning', 'Ensemble Learning', 'Deep Learning')
)

if model_name=='Machine Learning':
    classifier_name = st.sidebar.selectbox(
    'Select classifier',
    ('Logistic Regression', 'Decision Tree', 'Multinomial Naive Bayes', 'Random Forest','K-Nearest Neighbors','Support Vector Machine')
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



#function
def extract_skills(Mails_text):
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(Mails_text)
    tokens = [word.lower() for word in tokens if word.lower() not in stop_words]

    data = pd.read_csv(r"D:\New folder\Cleaned_mails")
    skills = list(data.columns.values)
    skillset = []

    for token in tokens:
        if token in skills:
            skillset.append(token)

    return [i.capitalize() for i in set([i.lower() for i in skillset])]

def getText(filename):
    fullText = ''  # Create an empty string
    if filename.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        doc = docx2txt.process(filename)
        for para in doc:
            fullText = fullText + para
    else:
        # Using pdfplumber instead of PyPDF2
            try:
                with pdfplumber.open(filename) as pdf_file:
            # Read the first page
                  page = pdf_file.pages[0]
                  page_content = page.extract_text()
                  fullText = fullText + page_content
            except Exception as e:
                print(f"Error: {e}")
    return fullText

def display(txt_file):
    Mails = []
    try:
        with open(txt_file, 'r', encoding='utf-8') as txt:
            content = txt.read()
            Mails.append(content)
    except FileNotFoundError:
        print(f"File not found: {txt_file}")
    except Exception as e:
        print(f"An error occurred while processing the file: {e}")
    return Mails

file_type = pd.DataFrame([], columns=['Uploaded File Name', 'content', 'class'])

filename = []
predicted = []
skills = []


# Load the saved SVM model and TF-IDF vectorizer
model = joblib.load('D:\\New folder\\model_svm.joblib')
tfidf_vectorizer = joblib.load('D:\\New folder\\tfidf_vector .joblib')

# Create a Streamlit web app
st.title('Text Classification: Abusive vs Non-Abusive')
upload_file = st.file_uploader('Upload Your Mails', type=['docx', 'pdf', 'txt', 'csv'], accept_multiple_files=True)

def preprocess(sentence):
    sentence = str(sentence)
    sentence = sentence.lower()
    sentence = sentence.replace('{html}',"")
    cleanr = re.compile('<.*?>')
    cleantext = re.sub(cleanr, '', sentence)
    rem_url = re.sub(r'http\S+', '',cleantext)
    rem_num = re.sub('[0-9]+', '', rem_url)
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(rem_num)
    filtered_words = [w for w in tokens if len(w) > 2 if not w in stopwords.words('english')]

    return " ".join(filtered_words)
    
if upload_file is not None:
    # Read the uploaded CSV file
 try:
    emails_df = pd.read_csv(upload_file)
 except FileNotFoundError:
    print("File not found. Please check the file path.")
 except pd.errors.ParserError:
    print("Error parsing the CSV file. Check the file format and encoding.")
 except Exception as e:
    print(f"An error occurred: {e}")
# Create columns for abusive and non-abusive emails
abusive_emails = []
non_abusive_emails = []

file_name_list = []
for doc_file in upload_file:
    if doc_file is not None:    
        cleaned = preprocess(display(doc_file))
        prediction = model.predict(tfidf_vectorizer.transform([cleaned]))[0]
        predicted.append(prediction)
        extText = getText(doc_file)
        skills.append(extract_skills(extText))
        file_name_list.append(doc_file.name)

file_type['Uploaded File Name'] = file_name_list

if len(predicted) > 0:
    file_type['Skills'] = skills
    file_type['content'] = predicted
    st.table(file_type.style.format())

select = ['abusive', 'Non-abusive']
st.subheader('Select as per Requirement')
option = st.selectbox('Fields', select)

if option == 'abusive':
    st.table(file_type[file_type['content'] == 'abusive'])
elif option == 'Non-abusive':
    st.table(file_type[file_type['content'] == 'Non-abusive'])

