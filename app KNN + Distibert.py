import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import DistilBertModel, DistilBertTokenizer,pipeline
import torch
import os
from torch import nn
#from sklearn.neighbors import NearestNeighbors
#from transformers import pipeline
import pandas as pd

# Download required NLTK data
nltk.download('wordnet', quiet=True)
nltk.download('stopwords', quiet=True)

app = Flask(__name__)

# Global variables
data = None
vectorizer = None
tfidf_matrix = None
nn_model = None
# Define model structure
class DistilBertForClassification(nn.Module):
    def __init__(self, num_labels):
        super().__init__()
        self.distilbert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.distilbert.config.hidden_size, num_labels)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.distilbert(input_ids=input_ids, attention_mask=attention_mask)
        return self.classifier(outputs.last_hidden_state[:, 0, :])

# Initialize model and classifiers
model = DistilBertModel.from_pretrained('distilbert-base-uncased')
categories = ["emotions", "occasion", "interests", "audience", "personality"]
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science","Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}
classifiers = nn.ModuleDict({
    category: nn.Linear(model.config.hidden_size, len(options[category]))
    for category in categories
})

# Load the saved model
def load_model(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classifiers.load_state_dict(checkpoint['classifiers_state_dict'])

# Initialize tokenizer
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Load the model at the start of your application
if os.path.exists('model.pth'):
    load_model('model.pth')
    print("Model loaded from file.")
else:
    print("Model file not found. Using default model.")

# Prediction function
def predict(phrase):
    model.eval()
    for classifier in classifiers.values():
        classifier.eval()

    inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
    with torch.no_grad():
        output = model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token
        predictions = {}
        for category, classifier in classifiers.items():
            logits = classifier(output)
            probabilities = torch.nn.functional.softmax(logits, dim=1)
            predicted_index = probabilities.argmax().item()
            predicted_option = list(options[category])[predicted_index]
            predictions[category] = predicted_option

    return predictions
# Load the zero-shot classification pipeline
classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")


def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[^;]*;', '', text)
    # Remove non-alphabetic characters
    text = re.sub(r'[^a-zA-Z]', ' ', text)
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
    return text

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

def initialize_data():
    global data, vectorizer, tfidf_matrix,nn_model
    data = pd.read_csv("output.csv")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    data["preprocessed_title"] = data["Title"].apply(preprocess_text)
    data["preprocessed_tags"] = data["Tags"].apply(preprocess_tags)
    data["preprocessed_text"] = data.apply(lambda row: " ".join(set(str(row["preprocessed_tags"]).split() + str(row["preprocessed_title"]).split() + str(row["preprocessed_description"]).split())), axis=1)
    # Create TF-IDF matrix - expliquer ici : https://imgur.com/Zn4rxWS

    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])
    # Create and fit the NearestNeighbors model
    #nn_model = NearestNeighbors(metric='cosine')
    #nn_model.fit(tfidf_matrix)

@app.route("/")
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

        # Combine all inputs into a single string
        all_text = f" {interests} {interests}  {occasion}  {audience} {personality} {emotions} "

        # Preprocess user answers
        preprocessed_answers = preprocess_text(all_text)

        # Create the input vector
        input_vector = vectorizer.transform([preprocessed_answers])

        # Find the nearest neighbors
        _, indices = nn_model.kneighbors(input_vector, n_neighbors=3)

        # Get the recommended products
        recommended_products = data.iloc[indices[0]]
        titles = recommended_products["Title"].tolist()
        urls = recommended_products["URL (Web)"].tolist()
        images = recommended_products["Image 1"].tolist()

        return render_template('index2.html', questions_data=zip(titles, urls, images), preprocessed_answers=preprocessed_answers)
    
    # This line should never be reached as we're only allowing POST requests
    return "Method not allowed", 405


@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    if request.method == 'POST':
        initialize_data()
        user_input = request.form['user_input']
        user_input = preprocess_text(user_input)
        
        # Use the trained model for prediction
        predictions = predict(user_input)
        
        predicted_labels = list(predictions.values())
        processed_classes = " ".join(predicted_labels)

        # Give more weight to certain labels (adjust as needed)
        weighted_processed_classes = f"{predicted_labels[2]} {predicted_labels[2]} {predicted_labels[4]} {predicted_labels[3]} {predicted_labels[1]} {predicted_labels[0]}"

        # Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([weighted_processed_classes])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

        # Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        recommended_titles = recommended_products["Title"].tolist()
        recommended_urls = recommended_products["URL (Web)"].tolist()
        recommended_images = recommended_products["Image 1"].tolist()

        if 'nlp-submit' in request.form:
            return render_template('index1.html', user_input=user_input, predicted_labels=predicted_labels, processed_classes=processed_classes, weighted_processed_classes=weighted_processed_classes,
                            recommended_data=zip(recommended_titles, recommended_urls, recommended_images))

    return render_template('index1.html')
    

if __name__ == '__main__':
    app.run()