# Importation des bibliothèques nécessaires
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
#from functools import partial
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# Initialisation de l'application Flask
app = Flask(__name__)

# Déclaration des variables globales
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

# Fonction exécutée avant la première requête pour charger les données et les modèles
@app.before_first_request
def load_data_and_models():
    global data, vectorizer, tfidf_matrix, normalized_tfidf_matrix, nn_model, classifier, options, category_order

    """
    Cette fonction initialise les modèles et charge les données nécessaires au fonctionnement de l'application.
    Elle est exécutée une seule fois, avant la première requête reçue par le serveur.

    Principales étapes :
    1. Chargement du modèle de classification zero-shot
    2. Définition des options pour chaque catégorie de préférence
    3. Chargement et prétraitement des données
    4. Création de la matrice TF-IDF
    5. Initialisation du modèle NearestNeighbors
    6. Normalisation de la matrice TF-IDF
    """

    # Chargement du pipeline de classification zero-shot avec un modèle plus petit
    classifier = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

    # Options prédéfinies pour chaque catégorie de préférence
    options = {
        "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
        "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Entertainment", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
        "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Life Style", "Plants", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs","Zodiac", "Technology", "Nature Wonders"],
        "audience": ["Child", "Teen", "Adult", "Senior"],
        "personality": ["Design créatif","Style rétro","Style nature","Tenues illustrées","Style artisanal","Corps et âme","Style symbolique","Univers imaginaire","Mode végétale","Passion animale"]
    }

    category_order = ["emotions", "occasion", "interests", "audience", "personality"]

    # Chargement et prétraitement des données
    data = pd.read_parquet("output_preprocessed.parquet")
    data["Description"] = data["Description"].astype(str)
    data["preprocessed_description"] = data["Description"].apply(preprocess_text)
    data["preprocessed_title"] = data["Title"].apply(preprocess_text)
    data["preprocessed_tags"] = data["Tags"].apply(preprocess_tags)
    data["preprocessed_text"] = data.apply(lambda row: " ".join(set(str(row["preprocessed_tags"]).split() + str(row["preprocessed_title"]).split() + str(row["preprocessed_description"]).split())), axis=1)

    # Création de la matrice TF-IDF
    """
    TF-IDF (Term Frequency-Inverse Document Frequency) est une technique de vectorisation du texte.
    Elle permet de représenter l'importance d'un mot dans un document par rapport à une collection de documents.
    - TF (Term Frequency) : fréquence d'un terme dans un document
    - IDF (Inverse Document Frequency) : mesure de l'importance du terme dans l'ensemble du corpus
    
    L'utilisation de max_features=5000 limite le vocabulaire aux 5000 mots les plus fréquents,
    ce qui aide à réduire la dimensionnalité et à se concentrer sur les termes les plus pertinents.
    """
    vectorizer = TfidfVectorizer(max_features=5000)
    tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

    # Création et ajustement du modèle NearestNeighbors
    """
    NearestNeighbors est un algorithme de recherche des plus proches voisins.
    Il est utilisé ici pour trouver les produits les plus similaires à une requête donnée.
    
    Paramètres :
    - metric='cosine' : utilise la similarité cosinus pour mesurer la distance entre les vecteurs
    - n_neighbors=3 : recherche les 3 voisins les plus proches
    - algorithm='brute' : calcule la distance avec tous les points (efficace pour les petits à moyens ensembles de données)
    - n_jobs=4 : utilise 4 cœurs de processeur pour accélérer le calcul
    """
    nn_model = NearestNeighbors(metric='cosine', n_neighbors=3, algorithm='brute', n_jobs=4)
    nn_model.fit(tfidf_matrix)

    # Normalisation de la matrice TF-IDF
    """
    La normalisation L2 (aussi appelée normalisation euclidienne) est appliquée à chaque vecteur de la matrice TF-IDF.
    Cela permet de mettre tous les documents à la même échelle, ce qui est particulièrement utile
    lors de l'utilisation de la similarité cosinus dans le modèle NearestNeighbors.
    """
    normalized_tfidf_matrix = normalize(tfidf_matrix, norm='l2', axis=1)

# Fonction de prétraitement du texte
def preprocess_text(text):
    """
    Cette fonction nettoie et prépare le texte pour l'analyse.
    Elle effectue les opérations suivantes :
    1. Suppression des caractères non alphabétiques
    2. Conversion en minuscules
    3. Tokenisation (séparation en mots)
    4. Suppression des mots vides (stop words)
    5. Stemming (réduction des mots à leur racine)
    """
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    return ' '.join([stemmer.stem(word) for word in text.split() if word not in stop_words])

# Mots à exclure lors du prétraitement des tags
exclude_words = {"LED light", "lamp", "Figurine", "Collectible", 
                 "Crystal", "Decoration", "Home decor", "3D engraved", 
                 "Novelty", "Keepsake", "Gift", "Decor", "Souvenir"} 

# Fonction de prétraitement des tags
def preprocess_tags(tags):
    """
    Cette fonction nettoie et prépare les tags pour l'analyse.
    Elle effectue les opérations suivantes :
    1. Conversion en chaîne de caractères
    2. Suppression des espaces au début et à la fin
    3. Conversion en minuscules
    4. Suppression des virgules
    5. Suppression des mots exclus définis dans 'exclude_words'
    """
    tags = str(tags)
    tags = tags.strip()
    tags = tags.lower()
    tags = re.sub(",","",tags)
    tags = " ".join([word for word in tags.split() 
                     if word not in exclude_words])
    return tags

# Fonction de classification du texte
def classify_text(user_input, category):
    """
    Cette fonction utilise le modèle de classification zero-shot pour classer le texte de l'utilisateur
    dans une catégorie donnée. Elle renvoie l'étiquette la plus probable pour cette catégorie.
    """
    labels = options[category]
    result = classifier(user_input, labels)
    return result['labels'][0], category

# Route pour la page d'accueil
@app.route('/')
def home():
    return render_template("index.html")

# Route pour le traitement des questions
@app.route('/questions', methods=['POST'])
def questions():
    """
    Cette fonction traite les réponses aux questions de l'utilisateur.
    Elle combine les réponses, les prétraite, et utilise le modèle NearestNeighbors
    pour trouver les produits les plus similaires.
    """
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

# Route pour le traitement NLP
@app.route('/nlp', methods=['GET', 'POST'])
def nlp():
    """
    Cette fonction traite les entrées textuelles libres de l'utilisateur.
    Elle utilise le modèle de classification zero-shot pour classer le texte dans différentes catégories,
    puis utilise ces classifications pour trouver les produits les plus similaires.
    
    Le processus comprend :
    1. Prétraitement du texte de l'utilisateur
    2. Classification du texte dans chaque catégorie (en parallèle avec ThreadPoolExecutor)
    3. Combinaison pondérée des classifications
    4. Utilisation de NearestNeighbors pour trouver les produits similaires
    5. Renvoi des résultats pour affichage
    """
    if request.method == 'POST':
        user_input = preprocess_text(request.form['user_input'])
        
        with ThreadPoolExecutor(max_workers=5) as executor:
            results = list(executor.map(lambda cat: classify_text(user_input, cat), category_order))

        predicted_labels = [label for label, _ in results]
        processed_classes = ' '.join(predicted_labels)

        weighted_processed_classes = f"{predicted_labels[2]} {predicted_labels[2]} {predicted_labels[4]} {predicted_labels[3]} {predicted_labels[1]} {predicted_labels[0]}"

        answers_vector = vectorizer.transform([weighted_processed_classes])
        normalized_answers_vector = normalize(answers_vector, norm='l2', axis=1)
        
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

"""
#EVALUATION Functions


def evaluate_questions(test_data):
    correct_predictions = 0
    total_predictions = 0
    total_time = 0

    for test_case in test_data:
        start_time = time.time()
        
        # Appel direct à la fonction de recommandation
        recommended_products = questions(test_case['input'])
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        expected_products = test_case['expected_products']
        
        correct_predictions += len(set(recommended_products) & set(expected_products))
        total_predictions += len(recommended_products)

    precision = correct_predictions / total_predictions if total_predictions > 0 else 0
    average_time = total_time / len(test_data)

    return {
        'precision': precision,
        'average_time': average_time
    }

def evaluate_nlp(test_data):
    y_true = []
    y_pred = []
    total_time = 0

    for test_case in test_data:
        start_time = time.time()
        
        # Appel direct à la fonction NLP
        predicted_labels, recommended_products = nlp(test_case['input'])
        
        end_time = time.time()
        total_time += (end_time - start_time)
        
        y_true.append(test_case['expected_categories'])
        y_pred.append(predicted_labels)

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    f1 = f1_score(y_true, y_pred, average='weighted')
    average_time = total_time / len(test_data)

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'average_time': average_time
    }

# Fonction de recommandation pour les questions (à adapter selon votre implémentation)
def questions(input_data):
    # Simuler le traitement des questions et retourner les produits recommandés
    # Cette fonction doit être adaptée pour correspondre à votre logique de recommandation réelle
    all_text = f"{input_data['interests']} {input_data['interests']} {input_data['occasion']} {input_data['audience']} {input_data['personality']} {input_data['emotions']}"
    preprocessed_answers = preprocess_text(all_text)
    input_vector = vectorizer.transform([preprocessed_answers])
    _, indices = nn_model.kneighbors(input_vector, n_neighbors=3)
    recommended_products = data.iloc[indices[0]]["Title"].tolist()
    return recommended_products

# Fonction NLP (à adapter selon votre implémentation)
def nlp(user_input):
    # Simuler le traitement NLP et retourner les étiquettes prédites et les produits recommandés
    # Cette fonction doit être adaptée pour correspondre à votre logique NLP réelle
    preprocessed_input = preprocess_text(user_input)
    
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(lambda cat: classify_text(preprocessed_input, cat), category_order))

    predicted_labels = [label for label, _ in results]
    
    weighted_processed_classes = f"{predicted_labels[2]} {predicted_labels[2]} {predicted_labels[4]} {predicted_labels[3]} {predicted_labels[1]} {predicted_labels[0]}"
    answers_vector = vectorizer.transform([weighted_processed_classes])
    normalized_answers_vector = normalize(answers_vector, norm='l2', axis=1)
    
    _, top_indices = nn_model.kneighbors(normalized_answers_vector)
    recommended_products = data.iloc[top_indices[0]]["Title"].tolist()

    return predicted_labels, recommended_products

# Exemple d'utilisation
test_questions_data = [
    {'input': {'emotions': 'Love', 'occasion': 'Birthday', 'interests': 'Art', 'audience': 'Adult', 'personality': 'Artisan Aesthetic'},
     'expected_products': ['Artistic Birthday Canvas', 'Handcrafted Love Sculpture', 'Adult Coloring Book Set']},
    # Ajoutez d'autres cas de test...
]

test_nlp_data = [
    {'input': 'I need a gift for my artistic friend who loves nature',
     'expected_categories': ['Art', 'Nature Wonders', 'Gift']},
    # Ajoutez d'autres cas de test...
]

questions_results = evaluate_questions(test_questions_data)
nlp_results = evaluate_nlp(test_nlp_data)

#print("Questions Evaluation:", questions_results)
#print("NLP Evaluation:", nlp_results)
"""

# Point d'entrée de l'application
if __name__ == '__main__':
    app.run(threaded=True)