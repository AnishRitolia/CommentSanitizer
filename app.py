import streamlit as st
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import streamlit as st
from PIL import Image

#Changing App Name and Icon
img = Image.open("img/icon.png")
st.set_page_config(page_title="CommentSanitizer",page_icon=img)

#Removing header and Footer of the Web-App
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

# Inject the CSS code
st.markdown(hide_st_style, unsafe_allow_html=True)


#Comment sanitizer checker
def load_tfidf():
    tfidf = pickle.load(open("tf_idf.pkt", "rb"))
    return tfidf

def load_model():
    nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
    return nb_model

def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

st.header("Toxicity Detection App")

st.subheader("Input your text")

text_input = st.text_input("Enter your text")

if text_input is not None:
    if st.button("Analyse"):
        result = toxicity_prediction(text_input)
        st.subheader("Result:")
        st.info("The result is "+ result + ".")
