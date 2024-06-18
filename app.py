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
from streamlit_lottie import st_lottie
import json

# Changing App Name and Icon
img = Image.open("img/icon.png")
st.set_page_config(page_title="Comment Sanitizer – Purifying Comments for Safer Web", page_icon=img, layout="wide")

# Removing header and Footer of the Web-App
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            .css-1d391kg {margin-left: 5% !important; margin-right: 5% !important;}
            .stButton button {background-color: #4CAF50; color: white; border-radius: 5px;}
            .stButton button:hover {background-color: #45a049;}
            .stTextInput input {border: 1px solid #ccc; padding: 10px; border-radius: 5px;}
            .stTextArea textarea {border: 1px solid #ccc; padding: 10px; border-radius: 5px;}
            </style>
            """
# Inject the CSS code
st.markdown(hide_st_style, unsafe_allow_html=True)

#Importing json animation into project from file
def load_lottiefile(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# Load the tfidf vectorizer and the model
@st.cache_resource
def load_tfidf():
    with open('tf_idf.pkt', 'rb') as f:
        tfidf = pickle.load(f)
    return tfidf

@st.cache_resource
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
    class_name = "Harmful" if prediction == 1 else "Safe"
    return class_name

# Option menu
selected = option_menu(
    menu_title="Comment Sanitizer – Purifying Comments for Safer Web",
    options=["Home", "Input Comment", "CSV File", "FAQ", "Contact"],
    icons=["house", "keyboard", "file-text", "question-circle", "envelope"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "black", "font-size": "22px"},
        "nav-link": {"text-align": "center", "font-size": "22px", "--hover-color": "#FFEF99"},
        "nav-link-selected": {"background-color": "#4CAF50", "color": "white"},
    },
)

# Home Section
if selected == "Home":
    st.title("Comment Sanitizer – Purifying Comments for Safer Web")
    st.header("Welcome to Comment Sanitizer!")
    st.subheader("Project Overview")
    st.write("""
    Comment Sanitizer is a project developed by Aastha Mahato and Anish Ritolia. It aims to detect and classify harmful comments using 
    machine learning techniques. This tool provides users with the ability to input individual comments for analysis or upload CSV files 
    containing multiple comments for batch processing. Additionally, the tool offers insightful visualizations, including class distributions 
    and word clouds, to help understand the nature of the comments being analyzed.
    
    **Key Features:**
    - Analyze individual comments for harmful content.
    - Batch process multiple comments from a CSV file.
    - Visualize the distribution of harmful and safe comments.
    - Generate word clouds for harmful and safe comments.
    
    Navigate through the sections using the menu above to explore the different functionalities of Comment Sanitizer.
    """)

    lottie_coading = load_lottiefile("img/home.json")
    st_lottie(
        lottie_coading,
        speed = 1,
        reverse = False,
        loop=True,
        quality="high",
        height = "300px",
        width = "100%",
        key = None,
    )

    

# Input Comment Section
if selected == "Input Comment":
    st.header("Input Comment for Analysis")
    st.subheader("Enter your text below:")
    text_input = st.text_area("Enter your text", height=150)

    if st.button("Sanitize Comment"):
        if text_input:
            result = toxicity_prediction(text_input)
            st.subheader("Result:")
            st.info(f"The result is {result}.")

# CSV File Section
if selected == "CSV File":
    st.header("Upload CSV File for Batch Comment Analysis")
    st.subheader("Upload a CSV file containing comments:")
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.bar_chart(class_distribution)

            with col2:
                # Pie Chart for Class Distribution
                fig1, ax1 = plt.subplots(figsize=(5, 5))
                ax1.pie(class_distribution, labels=class_distribution.index, autopct='%1.1f%%', startangle=90)
                ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                st.pyplot(fig1)
            
            # Align histograms and word clouds side by side
            col3, col4 = st.columns(2)
            
            with col3:
                # Histogram of Comment Lengths
                st.write("Histogram of Comment Lengths:")
                df['text_length'] = df['text'].apply(len)
                fig2, ax2 = plt.subplots(figsize=(5, 5))
                sns.histplot(data=df, x='text_length', hue='Prediction', multiple='stack', ax=ax2, bins=30)
                st.pyplot(fig2)
            
            with col4:
                # Word Cloud for Harmful and Safe comments
                st.write("Word Cloud for Harmful Comments")
                harmful_comments = " ".join(df[df['Prediction'] == "Harmful"]['text'])
                harmful_wordcloud = WordCloud(width=400, height=200, background_color='black', colormap='Reds').generate(harmful_comments)
                plt.figure(figsize=(5, 5))
                plt.imshow(harmful_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
                
                st.write("Word Cloud for Safe Comments")
                safe_comments = " ".join(df[df['Prediction'] == "Safe"]['text'])
                safe_wordcloud = WordCloud(width=400, height=200, background_color='white', colormap='Blues').generate(safe_comments)
                plt.figure(figsize=(5, 5))
                plt.imshow(safe_wordcloud, interpolation='bilinear')
                plt.axis('off')
                st.pyplot(plt)
        else:
            st.error("CSV file must contain a 'text' column.")

# FAQ Section
if selected == "FAQ":
    st.header("Frequently Asked Questions")

    ottie_coading = load_lottiefile("img/faq.json")
    st_lottie(
        # lottie_coading,
        speed=0.9,
        reverse=False,
        loop=True,
        quality="high",
        height="300px",
        width="100%",
        key=None,
    )
    
    faq_content = """
    ### What is Comment Sanitizer?
    Comment Sanitizer is a tool designed to detect and classify harmful comments using machine learning techniques. It allows users to analyze individual comments or batch process multiple comments from a CSV file.

    ### How does Comment Sanitizer work?
    Comment Sanitizer uses a pre-trained machine learning model to analyze the text of comments and classify them as either "Harmful" or "Safe." The model is based on a TfidfVectorizer for text feature extraction and a Multinomial Naive Bayes classifier for prediction.

    ### What kind of comments can be analyzed?
    Any text-based comments can be analyzed, including those from social media, forums, or any other platforms where user-generated content is present.

    ### How can I upload a CSV file for batch processing?
    Navigate to the "CSV File" section, and upload a CSV file containing a column named "text" which includes the comments you want to analyze.

    ### Who developed Comment Sanitizer?
    Comment Sanitizer was developed by Aastha Mahato and Anish Ritolia as part of their final year project.

    ### Can I visualize the results?
    Yes, Comment Sanitizer provides various visualizations, including class distribution charts, comment length histograms, and word clouds for harmful and safe comments.

    ### How can I get in touch for further questions?
    You can contact us via email at: aasthanikku2001@gmail.com and anishritolia6@gmail.com
    """
    st.markdown(faq_content)

# Contact Section
if selected == "Contact":
    st.header("Contact Us")
    st.write("For any further questions or inquiries, please reach out to us at:")
    st.write("**Aastha Mahato**: [aasthanikku2001@gmail.com](mailto:aasthanikku2001@gmail.com)")
    st.write("**Anish Ritolia**: [anishritolia6@gmail.com](mailto:anishritolia6@gmail.com)")

    st.write("We look forward to hearing from you!")
            
    lottie_coading = load_lottiefile("img/contact.json")
    st_lottie(
        lottie_coading,
        speed=0.9,
        reverse=False,
        loop=True,
        quality="high",
        height="150px",
        width="100%",
        key=None,
    )

