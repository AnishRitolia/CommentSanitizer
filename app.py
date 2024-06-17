import streamlit as st
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
from PIL import Image
from streamlit_option_menu import option_menu
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Changing App Name and Icon
img = Image.open("img/icon.png")
st.set_page_config(page_title="CommentSanitizer", page_icon=img)

# Removing header and Footer of the Web-App
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """

# Inject the CSS code
st.markdown(hide_st_style, unsafe_allow_html=True)

# Define the functions to load the tfidf vectorizer and the model
def load_tfidf():
    with open('tf_idf.pkt', 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf

def load_model():
    with open('toxicity_model.pkt', 'rb') as f:
        nb_model = pickle.load(f)
    return nb_model

# Define the prediction function
def toxicity_prediction(text):
    tfidf = load_tfidf()
    text_tfidf = tfidf.transform([text]).toarray()
    nb_model = load_model()
    prediction = nb_model.predict(text_tfidf)
    class_name = "Toxic" if prediction == 1 else "Non-Toxic"
    return class_name

# Option menu
selected = option_menu(
    menu_title=None,
    options=["Home", "Project", "FAQ", "Contact"],
    icons=["house", "book", "question-circle", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "icon": {"color": "black", "font-size": "22px"},
        "nav-link": {"text-align": "center", "font-size": "22px", "--hover-color": "#FFEF99"},
    },
)

# Page 1: Home
if selected == "Home":
    st.header("Toxicity Detection App")
    st.subheader("Input your text")
    text_input = st.text_input("Enter your text")

    if text_input:
        if st.button("Analyse"):
            result = toxicity_prediction(text_input)
            st.subheader("Result:")
            st.info(f"The result is {result}.")

# Page 2: Project
if selected == "Project":
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    if uploaded_file is not None:
        # Read the CSV file
        df = pd.read_csv(uploaded_file)
        
        # Display the dataframe
        st.write("DataFrame:")
        st.write(df)

        if 'text' in df.columns:
            # Run predictions
            df['Prediction'] = df['text'].apply(toxicity_prediction)

            # Display predictions
            st.write("Predictions:")
            st.write(df[['text', 'Prediction']])
            
            # Display the count of each class
            st.write("Class Distribution:")
            class_distribution = df['Prediction'].value_counts()
            st.bar_chart(class_distribution)

            # Word Cloud for Toxic and Non-Toxic comments
            toxic_comments = " ".join(df[df['Prediction'] == "Toxic"]['text'])
            non_toxic_comments = " ".join(df[df['Prediction'] == "Non-Toxic"]['text'])

            st.write("Word Cloud for Toxic Comments")
            toxic_wordcloud = WordCloud(width=800, height=400, background_color='black', colormap='Reds').generate(toxic_comments)
            plt.figure(figsize=(10, 5))
            plt.imshow(toxic_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)

            st.write("Word Cloud for Non-Toxic Comments")
            non_toxic_wordcloud = WordCloud(width=800, height=400, background_color='white', colormap='Blues').generate(non_toxic_comments)
            plt.figure(figsize=(10, 5))
            plt.imshow(non_toxic_wordcloud, interpolation='bilinear')
            plt.axis('off')
            st.pyplot(plt)
        else:
            st.error("CSV file must contain a 'text' column.")
