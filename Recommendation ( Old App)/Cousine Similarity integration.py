from flask import Flask, render_template, request
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)
app.static_folder = 'uploads'

# Load and preprocess data
def load_data():
    data = pd.read_csv("output.csv")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    # data["preprocessed_title"] = data["Title"].apply(preprocess_text)  # Uncomment if needed
    data["preprocessed_text"] = data["preprocessed_description"] + '' + data['Tags']
    data.dropna(subset=['preprocessed_text'], inplace=True)
    return data

# Text preprocessing function
def preprocess_text(text):
    # Remove HTML tags and entities, convert to lowercase
    text = re.sub(r'<[^>]*>', '', text)
    text = re.sub(r'&[^;]*;', '', text)
    text = text.lower()
    return text

# Collect user answers
def get_user_answers():
    # ... (Code for collecting user answers)

# Calculate similarity and recommendations using cosine similarity
def get_recommendations_cosine(preprocessed_answers, tfidf_matrix):
    answers_vector = tfidf_vectorizer.transform([preprocessed_answers])
    similarities = cosine_similarity(answers_vector, tfidf_matrix)
    top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices
    recommended_products = data.iloc[top_indices]
    return recommended_products

# Render question form
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html', product_ids=[], product_titles=[])

# Process recommendations
@app.route('/recommend', methods=['POST'])
def recommend():
    # Get user answers
    emotions, occasion, interests, audience, personality = get_user_answers()

    # Preprocess answers
    preprocessed_answers = preprocess_text(" ".join([emotions, occasion, interests, audience, personality]))

    # Load data and create TF-IDF matrix
    data = load_data()
    tfidf_vectorizer = TfidfVectorizer()
    tfidf_matrix = tfidf_vectorizer.fit_transform(data["preprocessed_text"])

    # Get recommendations using cosine similarity
    recommended_products = get_recommendations_cosine(preprocessed_answers, tfidf_matrix)

    # Render recommendations
    return render_template('index.html', product_ids=recommended_products["Product_ID"].tolist(),
                                       product_titles=recommended_products["Title"].tolist())

if __name__ == '__main__':
    app.run(debug=True)