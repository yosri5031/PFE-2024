from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize
import numpy as np
import re
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.neighbors import NearestNeighbors
from transformers import pipeline
import pandas as pd
from concurrent.futures import ThreadPoolExecutor

app = Flask(__name__)

# Global variables
data = None
vectorizer = None
tfidf_matrix = None
normalized_tfidf_matrix = None
nn_model = None
classifier = None
options = None
category_order = None
stemmer = PorterStemmer()
stop_words = set(stopwords.words('english'))

@app.before_first_request
def load_data_and_models():
    global data, vectorizer, tfidf_matrix, normalized_tfidf_matrix, nn_model, classifier, options, category_order

    # Load the zero-shot classification pipeline with a smaller model
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Predefined options for each preference category
    options = {
        "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
        "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Entertainment", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
        "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Life Style", "Plants", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs","Zodiac", "Technology", "Nature Wonders", "Names"],
        "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
        "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
    }

    category_order = ["emotions", "occasion", "interests", "audience", "personality"]

    # Load and preprocess data
    data = pd.read_parquet("output_preprocessed.parquet")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    data["preprocessed_title"] = data["Title"].apply(preprocess_text)
    data["preprocessed_tags"] = data["Tags"].apply(preprocess_tags)
    data["preprocessed_text"] = data.apply(lambda row: " ".join(set(str(row["preprocessed_tags"]).split() + str(row["preprocessed_title"]).split() + str(row["preprocessed_description"]).split())), axis=1)
    # Create TF-IDF matrix
    vectorizer = TfidfVectorizer(max_features=5000)  # Limit features to 5000 most frequent words
    tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

    # Create and fit the NearestNeighbors model
    nn_model = NearestNeighbors(metric='cosine', n_neighbors=3, algorithm='brute', n_jobs=-1)
    nn_model.fit(tfidf_matrix)

    # Normalize the TF-IDF matrix
    normalized_tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)

def preprocess_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])

exclude_words = {"LED light", "lamp", "Figurine", "Collectible", 
                 "Crystal", "Decoration", "Home decor", "3D engraved", 
                 "Novelty", "Keepsake", "Gift", "Decor", "Souvenir"} 

def preprocess_tags(tags):
  tags = str(tags)  #convert to string if there are numbers or float
  tags = tags.strip() #remove only whitespaces from the start and end of the string
  tags = tags.lower() #minuscule
  tags = re.sub(",","",tags) # remove commas
  tags = " ".join([word for word in tags.split() 
                   if word not in exclude_words])
  return tags

def classify_text(user_input, category):
    labels = options[category]
    result = classifier(user_input, labels)
    return result['labels'][0], category

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/questions', methods=['POST'])
def questions():
    if request.method == 'POST':
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
        user_input = preprocess_text(request.form['user_input'])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda cat: classify_text(user_input, cat), category_order))

        predicted_labels = [label for label, _ in results]
        processed_classes = ' '.join(predicted_labels)

        weighted_processed_classes = f"{predicted_labels[2]} {predicted_labels[2]} {predicted_labels[4]} {predicted_labels[3]} {predicted_labels[1]} {predicted_labels[0]}"

        answers_vector = vectorizer.transform([weighted_processed_classes])
        normalized_answers_vector = normalize(answers_vector, norm='l2', axis=1)
        
        # Use NearestNeighbors for faster similarity search
        _, top_indices = nn_model.kneighbors(normalized_answers_vector)

        recommended_products = data.iloc[top_indices[0]]
        recommended_titles = recommended_products["Title"].tolist()
        recommended_urls = recommended_products["URL (Web)"].tolist()
        recommended_images = recommended_products["Image 1"].tolist()

        if 'nlp-submit' in request.form:
            return render_template('index1.html', user_input=user_input, predicted_labels=predicted_labels, 
                                   processed_classes=processed_classes, weighted_processed_classes=weighted_processed_classes,
                                   recommended_data=zip(recommended_titles, recommended_urls, recommended_images))

    return render_template('index1.html')

if __name__ == '__main__':
    app.run(threaded=True)