import re
import nltk
nltk.download('wordnet')
nltk.download('stopwords')
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import pipeline
import pandas as pd
import random 


app = Flask(__name__)

# Load the zero-shot classification pipeline with a pre-trained model
# the origin model https://huggingface.co/facebook/bart-large-mnli
classifier = pipeline("zero-shot-classification")

#Classifier = pipeline("facebook/bart-model")
#model = KNN.new()
#model.add(Dense=3,method="Adam")
#model.add(Dense=2,method="Adam")
# Predefined options for each preference category
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science","Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}

# Order of the categories
category_order = ["emotions", "occasion", "interests", "audience", "personality"]

@app.route("/")
def home():    
    return render_template("index.html")

@app.route('/question', methods=['GET', 'POST'])
def questions():
    if request.method == 'POST':
        inputs = list(request.form.values())
        random.shuffle(inputs)
        emotions = request.form['emotions']
        occasion = request.form['occasion']
        #interests = request.form['interests']
        audience = request.form['audience']
        personality = request.form['personality']

        # Preprocess user answers
        def preprocess_text(text):
            # Suppression des balises HTML
            text = re.sub(r'<[^>]*>', '', text)
            # Suppression des entités HTML
            text = re.sub(r'&[^;]*;', '', text)
            # Suppression des caractères non alphabétiques
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            # Passage en minuscules
            text = text.lower()
            # Lemmatisation
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            text = ' '.join(lemmatized_words)
            # Filtrage des mots vides
            stop_words = set(stopwords.words('english'))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            text = ' '.join(filtered_words)
            return text

        # Load data from CSV
        data = pd.read_csv("output.csv")
        data["Description"] = data["Description"].astype(str)
        data["preprocessed_description"] = data["Description"].apply(preprocess_text)
        data["preprocessed_title"] = data["Title"].apply(preprocess_text)
        data["preprocessed_text"] = " ".join(set(str(data["preprocessed_title"]).split() + str(data["preprocessed_description"]).split()))

        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

        # Store answers in a list
        answers = [emotions, occasion,audience, personality]

        # Preprocess user answers
        preprocessed_answers = preprocess_text(" ".join(answers))

        # Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([preprocessed_answers])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

        # Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        titles = recommended_products["Title"].tolist()  # Get the recommended titles as a list
        urls = recommended_products["URL (Web)"].tolist()  # Get the recommended urls as a list
        images = recommended_products["Image 1"].tolist()  # Get the recommended images as a list

        # Prepare the data to pass to the template
        if 'questions-submit' in request.form:
            # Process questions form submission
            return render_template('index2.html', questions_data=zip(titles, urls, images), preprocessed_answers=preprocessed_answers)
    
    # Render the form template for GET requests or other cases
    return render_template('index2.html')


@app.route('/nlp', methods=['GET', 'POST'])
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
        """
        The role of the classifier function (zero shot) in the provided code is to classify the user's input text into relevant categories or labels. It performs a zero-shot classification, which means it can classify the text into categories that were not seen during training by leveraging a pre-trained model's knowledge.

        Here's a breakdown of how the classifier function is used in the code:

        The classifier function is called with the user_input text and a list of labels (labels) specific to a certain category. 
        The classifier function returns a result that contains the predicted labels for the given text.
        The predicted labels are then iterated through in the for loop. The loop checks if a label has already been seen 
        (label not in seen_categories). If the label is new, it is appended to the predicted_labels list and the processed_classes list. 
        The purpose of the processed_classes list is to store all the unique labels that have been predicted for later use.
        After iterating through all the categories, the processed_classes list is joined into a single line using " ".join(processed_classes). 
        This creates a string representation of all the unique labels predicted for the user's input.
        The processed_classes string is then used in subsequent steps for similarity calculations and generating recommended products.
        """
        
        # Load data from CSV
        data = pd.read_csv("output.csv")
        data["Description"] = data["Description"].astype(str)

#pre-process data
        def preprocess_text(text):
            # Suppression des balises HTML
            text = re.sub(r'<[^>]*>', '', text)
            # Suppression des entités HTML
            text = re.sub(r'&[^;]*;', '', text)
            # Suppression des caractères non alphabétiques
            text = re.sub(r'[^a-zA-Z]', ' ', text)
            # Passage en minuscules
            text = text.lower()
            # Lemmatisation
            lemmatizer = WordNetLemmatizer()
            words = text.split()
            lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
            text = ' '.join(lemmatized_words)
            # Filtrage des mots vides
            stop_words = set(stopwords.words('english'))
            words = text.split()
            filtered_words = [word for word in words if word not in stop_words]
            text = ' '.join(filtered_words)
            return text

        data["preprocessed_description"] = data["Description"].apply(preprocess_text)
        data["preprocessed_title"] = data["Title"].apply(preprocess_text)
        data["preprocessed_text"] = data["preprocessed_title"] + " " + data["preprocessed_description"]
        # Create TF-IDF matrix - expliquer ici : https://imgur.com/Zn4rxWS
        vectorizer = TfidfVectorizer()
        tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])
        ####################################################################
# Calculate similarity with user answers (cosine similarity)
        answers_vector = vectorizer.transform([processed_classes])
        similarities = cosine_similarity(answers_vector, tfidf_matrix)
        top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

# Rank and display recommended products
        recommended_products = data.iloc[top_indices]
        recommended_titles = recommended_products["Title"].tolist()  # Get the recommended titles as a list
        recommended_urls = recommended_products["URL (Web)"].tolist()  # Get the recommended urls as a list
        recommended_images = recommended_products["Image 1"].tolist()  # Get the recommended images as a list
        #recommended_images = recommended_products["Image 2"].tolist() # Get the recommended images as a list
        if 'nlp-submit' in request.form:
                return render_template('index1.html', user_input=user_input, predicted_labels=predicted_labels, processed_classes=processed_classes,
                                recommended_data=zip(recommended_titles, recommended_urls, recommended_images))
        
#test Ranking and displaying for recommended products
    #recommended_p = data.iloc[top_indiceshil]

if __name__ == '__main__':
    app.run(debug=True)
