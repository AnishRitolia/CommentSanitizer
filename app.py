import streamlit as st
import pickle 
import numpy as np 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu

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

selected = option_menu(
    menu_title=None,
    options=["Home", "Project","FAQ","Contact"],
    icons=["house","book","question-circle","envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
            "icon": {"color": "black", "font-size": "22px"},
            "nav-link": { "text-align": "center","font-size": "22px", "--hover-color": "#FFEF99"},
        },
)


#page 1
if selected == "Home":
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


#page 2
if selected == "Project":
            uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

            if uploaded_file is not None:
                # Read the CSV file
                df = pd.read_csv(uploaded_file)
            
                # Display the dataframe
                st.write("DataFrame:")
                st.write(df)
            
                # Load the model
                def load_tfidf():
                            tfidf = pickle.load(open("tf_idf.pkt", "rb"))
                            return tfidf
            
                def load_model():
                            nb_model = pickle.load(open("toxicity_model.pkt", "rb"))
                            return nb_model
            
                # Run predictions
                predictions = load_model(predict(df))
            
                # Display predictions
                st.write("Predictions:")
                st.write(predictions)
            
