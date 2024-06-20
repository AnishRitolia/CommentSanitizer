# Comment Sanitizer

Comment Sanitizer is a web application designed to detect and classify harmful comments using machine learning techniques. It provides users with tools to analyze individual comments, batch process CSV files, visualize comment distributions, and generate word clouds for harmful and safe comments.

![Comment Sanitizer Demo](demo.gif)

## Features

- **Input Comment Analysis:** Analyze individual comments for toxicity classification.
- **CSV Batch Processing:** Upload CSV files containing multiple comments for batch analysis.
- **Visualization:** Visualize distributions of toxic and non-toxic comments.
- **Word Clouds:** Generate word clouds to identify common words in harmful and safe comments.
- **FAQ and Contact Sections:** Provide information and contact options for users.

## Technologies Used

- **Backend:** Python, FastAPI
- **Machine Learning:** TensorFlow, Scikit-learn
- **Web Framework:** Streamlit, HTML/CSS
- **Visualization:** Matplotlib, Seaborn, WordCloud
- **Deployment:** Heroku, Docker

## Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/comment-sanitizer.git
   cd comment-sanitizer
   ```

2. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Run the Streamlit web application:

   ```bash
   streamlit run app.py
   ```

4. Access the application at [http://localhost:8501](http://localhost:8501) in your web browser.

## Usage

- **Input Comment Analysis:** Enter text in the input box and click "Sanitize Comment" to classify it as "Safe" or "Harmful."
- **CSV Batch Processing:** Upload a CSV file containing a "text" column to analyze multiple comments at once.
- **Visualization:** Explore visualizations such as histograms of comment lengths and word clouds.
- **FAQ and Contact:** Navigate through the FAQ and Contact sections to get more information about the project and reach out to developers.

## Project Structure

```
├── app.py              # Streamlit web application code
├── api.py              # FastAPI backend code for prediction endpoint
├── requirements.txt    # Python dependencies
├── tf_idf.pkt          # Pickled TF-IDF vectorizer
├── toxicity_model.pkt  # Pickled Multinomial Naive Bayes model
├── README.md           # Project README file
├── img/                # Directory for images used in UI
│   ├── icon.png        # Project icon
│   ├── FAQ.json        # Lottie animation for FAQ section
│   ├── contact.json    # Lottie animation for Contact section
│   └── ...             # Other UI-related images and animations
└── data/               # Directory for dataset and data-related files
    └── train.csv       # Example dataset (replace with your actual dataset)
```

## Contact

For further questions or inquiries, please reach out to:

- **Aastha Mahato**: [aasthanikku2001@gmail.com](mailto:aasthanikku2001@gmail.com)
- **Anish Ritolia**: [anishritolia6@gmail.com](mailto:anishritolia6@gmail.com)
