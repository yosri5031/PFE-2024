import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd

app = Flask(__name__)

# Load the zero-shot classification pipeline with a pre-trained model
classifier = pipeline("zero-shot-classification")

# Predefined options for each preference category
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Casual and laid-back", "Elegant and sophisticated", "Edgy and avant-garde", "Bohemian and free-spirited", "Classic and timeless"]
}

# Order of the categories
category_order = ["emotions", "occasion", "interests", "audience", "personality"]

@app.route('/', methods=['GET', 'POST'])
def questions():

  if request.method == 'POST':

    # Get form inputs
    emotions = request.form['emotions']
    occasion = request.form['occasion']
    interests = request.form['interests']
    audience = request.form['audience']
    personality = request.form['personality']
    # Preprocess inputs
    preprocessed_answers = preprocess(inputs)

    # Load data
    data = pd.read_csv("output.csv")

    # Preprocess data
    data = preprocess_data(data)  

    # Create TF-IDF matrix 
    tfidf_matrix = create_matrix(data)

    # Calculate recommendations
    recommendations = get_recommendations(preprocessed_answers, tfidf_matrix)

  else:
    preprocessed_answers = ""

  return render_template('index.html',
                         preprocessed_answers=preprocessed_answers,
                         recommendations=recommendations)

def preprocess(inputs):
  # preprocessing logic
  return preprocessed

def preprocess_data(data):
  # preprocessing logic
  return data

def create_matrix(data):  
  # TF-IDF matrix creation
  return matrix

def get_recommendations(text, matrix):
  # recommendation logic
  return recommendations 
def nlp():
    if request.method == 'POST':
        # Retrieve the user input from the request
        user_input = request.form['user_input']

        # Classification of the new text
        predicted_labels = []
        seen_categories = set()
        processed_classes = []  # Store processed classes as a list

        for category in category_order:
            labels = options[category]
            result = classifier(user_input, labels)

            for label in result["labels"]:
                if label not in seen_categories:
                    predicted_labels.append(label)
                    processed_classes.append(label)  # Append label to processed classes list
                    seen_categories.add(label)
                    break

        processed_classes = " ".join(processed_classes)  # Join all labels into a single line
        # Load data from CSV
        data = pd.read_csv("output.csv")
        data["Description"] = data["Description"].astype(str)

#pre-process data
        def preprocess_text(text):
    # Remove HTML tags
            text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
            text = re.sub(r'&[^;]*;', '', text)
    # Convert to lowercase
            text = text.lower()
            return text

        data["preprocessed_description"] = data["Description"].apply(preprocess_text)
        data["preprocessed_title"] = data["Title"].apply(preprocess_text)
        data["preprocessed_text"] = data["preprocessed_title"] + " " + data["preprocessed_description"]
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

# Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([processed_classes])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

# Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        recommended_titles = recommended_products["Title"].tolist()  # Get the recommended titles as a list
        recommended_urls = recommended_products["URL (Web)"].tolist()  # Get the recommended urls as a list
        recommended_images = recommended_products["Image 1"].tolist()  # Get the recommended images as a list
        return render_template('index.html', user_input=user_input, predicted_labels=predicted_labels, processed_classes=processed_classes,
                                recommended_data=zip(recommended_titles, recommended_urls, recommended_images))

    return render_template('index.html')

if __name__ == '__main__':
    app.run()