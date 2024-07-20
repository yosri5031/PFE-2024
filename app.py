from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from langdetect import detect
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
import os
import joblib
import pandas as pd
from flask_caching import Cache
"""Key optimizations in this version:

Added caching using Flask-Caching for the initialize_data function.
Increased the cache size for preprocess_text and preprocess_tags functions.
Made initialize_data run only once when the app starts, not on every request.
Used all available cores for NearestNeighbors with n_jobs=-1.
Added threading support when running the app with threaded=True."""

app = Flask(__name__)

# Configure caching
cache = Cache(app, config={'CACHE_TYPE': 'simple'})

# Global variables
data = None
vectorizer = None
tfidf_matrix = None
normalized_tfidf_matrix = None
nn_model = None
classifier = None
options = None
category_order = None
classifier_path = 'classifier_pipeline.joblib'

# Predefined options for each preference category
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Entertainment", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests" : ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Life Style", "Plants", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs","Zodiac",
           "Technology", "Nature Wonders", "Names"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}

# Order of the categories
category_order = ["emotions", "occasion", "interests", "audience", "personality"]

@lru_cache(maxsize=10000)
def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[^;]*;', '', text)
    # Language detection
    language = detect(text)
    
    # Preprocessing for English
    if language == 'en':
        # Remove non-alphabetic characters, numbers, #, \, /
        text = re.sub(r'[^a-zA-ZÀ-ÿ]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Lemmatization
        lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
        # Filter stop words
        stop_words = set(stopwords.words('english'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
    
    # Preprocessing for French
    elif language == 'fr':
        # Remove non-alphabetic characters, numbers, #, \, /
        text = re.sub(r'[^a-zA-ZàâçéèêëîïôûùüÿñæœÀÂÇÉÈÊËÎÏÔÛÙÜŸÑÆŒ]', ' ', text)
        # Convert to lowercase
        text = text.lower()
        # Lemmatization
        lemmatizer = WordNetLemmatizer()  # Lemmatizer for French is not available in NLTK, consider using a French lemmatizer library
        words = text.split()
        lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
        text = ' '.join(lemmatized_words)
        # Filter stop words
        stop_words = set(stopwords.words('french'))
        words = text.split()
        filtered_words = [word for word in words if word not in stop_words]
        text = ' '.join(filtered_words)
    return text

exclude_words = {"LED light", "lamp", "Figurine", "Collectible", 
                 "Crystal", "Decoration", "Home decor", "3D engraved", 
                 "Novelty", "Keepsake", "Gift", "Decor", "Souvenir"} 

@lru_cache(maxsize=10000)
def preprocess_tags(tags):
    tags = str(tags)  #convert to string if there are numbers or float
    tags = tags.strip() #remove only whitespaces from the start and end of the string
    tags = tags.lower() #minuscule
    tags = re.sub(",","",tags) # remove commas
    tags = " ".join([word for word in tags.split() 
                     if word not in exclude_words])
    return tags

@cache.cached(timeout=3600)  # Cache for 1 hour
def initialize_data():
    global data, vectorizer, tfidf_matrix, normalized_tfidf_matrix, nn_model, classifier
    
    # Load or initialize the classifier
    if os.path.exists(classifier_path):
        classifier = joblib.load(classifier_path)
    else:
        classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")
        joblib.dump(classifier, classifier_path)
    
    # Rest of the initialization code remains the same
    data = pd.read_csv("output_preprocessed.csv")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    data["preprocessed_title"] = data["Title"].apply(preprocess_text)
    data["preprocessed_tags"] = data["Tags"].apply(preprocess_tags)
    data["preprocessed_text"] = data.apply(lambda row: " ".join(set(str(row["preprocessed_tags"]).split() + str(row["preprocessed_title"]).split() + str(row["preprocessed_description"]).split())), axis=1)

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])
    nn_model = NearestNeighbors(metric='cosine', n_jobs=-1)  # Use all available cores
    nn_model.fit(tfidf_matrix)
    normalized_tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)

@lru_cache(maxsize=1000)
def cached_classifier(text, labels_tuple):
    return classifier(text, list(labels_tuple))

def process_category(category, user_input, options):
    labels = options[category]
    result = cached_classifier(user_input, tuple(labels))
    return result, category

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/questions', methods=['POST'])
def questions():
    if request.method == 'POST':
        initialize_data()
        emotions = request.form['emotions']
        occasion = request.form['occasion']
        interests = request.form['interests']
        audience = request.form['audience']
        personality = request.form['personality']

        all_text = f" {interests} {interests}  {occasion}  {audience} {personality} {emotions} "
        preprocessed_answers = preprocess_text(all_text)
        input_vector = vectorizer.transform([preprocessed_answers])

        _, indices = nn_model.kneighbors(input_vector, n_neighbors=3)

        recommended_products = data.iloc[indices[0]]
        titles = recommended_products["Title"].tolist()
        urls = recommended_products["URL (Web)"].tolist()
        images = recommended_products["Image 1"].tolist()

        return render_template('index2.html', questions_data=zip(titles, urls, images), preprocessed_answers=preprocessed_answers)
    
    return "Method not allowed", 405

@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    if request.method == 'POST':
        initialize_data()
        user_input = request.form['user_input']
        user_input = preprocess_text(user_input)
        
        predicted_labels = []
        seen_categories = set()
        processed_classes = []

        with ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(process_category, category, user_input, options) for category in category_order]
            for future in futures:
                result, category = future.result()
                for label in result["labels"]:
                    if label not in seen_categories:
                        predicted_labels.append(label)
                        processed_classes.append(label)
                        seen_categories.add(label)
                        break

        processed_classes = " ".join(processed_classes)

        weighted_processed_classes = ""
        if len(predicted_labels) >= 1:
            repeated_label = predicted_labels[2]
            for _ in range(2):
                weighted_processed_classes += f" {repeated_label}" 

        weighted_processed_classes += " " + predicted_labels[4] + "" + predicted_labels[3] + predicted_labels[1] + predicted_labels[0]

        # Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([weighted_processed_classes])
        normalized_answers_vector = normalize(answers_vector, norm='l2', axis=1)
        similarities = normalized_answers_vector.dot(normalized_tfidf_matrix.T).toarray()[0]
        top_indices = np.argsort(similarities)[-3:][::-1]  # Get top 3 indices

        # Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        recommended_titles = recommended_products["Title"].tolist()
        recommended_urls = recommended_products["URL (Web)"].tolist()
        recommended_images = recommended_products["Image 1"].tolist()

        if 'nlp-submit' in request.form:
            return render_template('index1.html', user_input=user_input, predicted_labels=predicted_labels, 
                                   processed_classes=processed_classes, weighted_processed_classes=weighted_processed_classes,
                                   recommended_data=zip(recommended_titles, recommended_urls, recommended_images))

    return render_template('index1.html')

if __name__ == '__main__':
    initialize_data()  # Initialize data when the app starts
    app.run(threaded=True)