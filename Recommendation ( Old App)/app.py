from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
import re

app = Flask(__name__)
app.static_folder = 'uploads'
# Render question form 
@app.route('/', methods=['GET'])
def index():
  return render_template('index.html',product_ids=[],
                         product_titles=[])

# Get form data and call recommendation function  
@app.route('/recommend', methods=['POST'])
def recommend():

  emotions = request.form['emotions']
  occasion = request.form['occasions']
  interests = request.form['interests']
  audience = request.form['audience']
  personality = request.form['personality']


  try:
    top_product_ids,top_product_titles = get_recommendations(emotions, occasion, interests,audience,personality)
  except Exception as e:
    print(e)
    top_product_ids,top_product_titles = None

  return render_template('index.html', product_ids=top_product_ids,
                         product_titles=top_product_titles)

# Recommendation logic function
def get_recommendations(emotions, occasion, interests, audience, personality):
  def preprocess_text(text):
    # Remove HTML tags
    text = re.sub(r'<[^>]*>', '', text)
    # Remove HTML entities
    text = re.sub(r'&[^;]*;', '', text)
    # Convert to lowercase
    text = text.lower()
    return text
  # Preprocess user answers 
  preprocessed_answers = preprocess_text(" ".join([emotions, occasion, interests, audience, personality]))

  # Load data
  data = pd.read_csv("output.csv")
  data["preprocessed_description"] = data["Description"].apply(preprocess_text)
  #data["preprocessed_title"] = data["Title"].apply(preprocess_text)
  data["preprocessed_text"] = data["preprocessed_description"] + '' + data['Tags']

  #Dropping the rows with NaN values:
  data.dropna(subset=['preprocessed_text'], inplace=True)
  #####
  # Preprocess descriptions
  vectorizer = TfidfVectorizer()
  X = vectorizer.fit_transform(data["preprocessed_text"])  

  # Fit KNN model
  knn = NearestNeighbors(n_neighbors=3)
  knn.fit(X)

  # Transform user answers to vector
  answer_vector = vectorizer.transform([preprocessed_answers])

  # Find nearest neighbors  
  distances, indices = knn.kneighbors(answer_vector)

  # Get top recommendation indices
  top_indices = indices[0]  

  # Get product IDs and titles
  top_product_ids = data.iloc[top_indices]["Product_ID"].tolist()
  top_product_titles = data.iloc[top_indices]["Title"].tolist()

  return top_product_ids, top_product_titles

# Run the app
if __name__ == '__main__':
  app.run(debug=True)