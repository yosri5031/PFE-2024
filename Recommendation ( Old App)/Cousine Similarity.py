## Code from my colabs
import pandas as pd
import re
import matplotlib.pyplot as plt
import csv
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
import seaborn as sns


#optimise product descriptions
def process_csv(file_path):
    categories = [
        "Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools",
        "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art",
        "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving",
        "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter",
        "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms",
        "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style",
        "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY",
        "Science", "Travel", "Cats"
    ]

    with open(file_path, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames + ['Description']
        rows = []

        for row in reader:
            new_description = ""
            for category in categories:
                if category.lower() in row['Tags'].lower():
                    new_description += f" {category} 3D Engraved Crystal"
            row['Description'] = new_description + " " + row['Description']
            rows.append(row)

        # Output the modified data to a new CSV file
        output_file = 'output.csv'
        with open(output_file, 'w', newline='') as outfile:
            writer = csv.DictWriter(outfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

        print(f"Processed CSV file and saved the results to {output_file}.")

# actual file path
process_csv('shopify.csv')

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
#data["preprocessed_title"] = data["Title"].apply(preprocess_text)
data["preprocessed_text"] = data["preprocessed_description"] + '' + data['Tags']

#Dropping the rows with NaN values:
data.dropna(subset=['preprocessed_text'], inplace=True)
#####

# Function to display options and get the user's choice
def get_user_choice(question, options):
    print(question)
    for i, option in enumerate(options, start=1):
        print(f"{i}. {option}")
    while True:
        choice = input(f"Choose one option (1-{len(options)}): ")
        if choice.isdigit() and 1 <= int(choice) <= len(options):
            if options[int(choice) - 1] == "Other":
                other_value = input("Enter your own value: ")
                return other_value
            else:
                return options[int(choice) - 1]
        else:
            print("Invalid choice. Please enter a valid number.")

# Collect user answers to the 5 questions
questions = {
    "emotions": "What emotions or feelings would you like shown in your gift?",
    "occasion": "What is the occasion/event you need a gift for?",
    "interests": "Could you please tell me about the hobbies and interests of the receiver?",
    "audience": "Is the gift targeted for:",
    "Personality": "How would you describe your personal style and aesthetic preferences?"
}

answers = {}

for key, question in questions.items():
    options = []  # Define the options for each question
    if key == "emotions":
        options = ["Love and romance", "Happiness and joy", "Peace and tranquility",
                   "Inspiration and motivation", "Sentimental and nostalgic"]
    elif key == "occasion":
        options = ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Other"]
    elif key == "interests":
        options = ["Animals", "Nature", "Inspiring quotes", "Art/Design", "Constructions", "Zodiac", "Other"]
    elif key == "audience":
        options = ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"]
    elif key == "Personality":
        options = ["Casual and laid-back", "Elegant and sophisticated", "Edgy and avant-garde",
                   "Bohemian and free-spirited", "Classic and timeless"]

    answer = get_user_choice(question, options)
    answers[key] = answer

# Preprocess user answers
preprocessed_answers = preprocess_text(" ".join(answers.values()))

# Create TF-IDF matrix+

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(data["preprocessed_text"])

# Calculate similarity with user answers (cosine similarity)
answers_vector = vectorizer.transform([preprocessed_answers])
similarities = cosine_similarity(answers_vector, tfidf_matrix)
top_indices = similarities.argsort()[0][-3:][::-1]  # Get top 3 indices

# Rank and display recommended products
recommended_products = data.iloc[top_indices]
print("Recommended products for you:")
print(recommended_products[["Title"]])

# Get the similarity values
similarity_values = similarities[0][top_indices]

# Reverse the order of similarity values
similarity_values = similarity_values[::-1]

# Create a figure and axis
fig, ax = plt.subplots()

# Plot thesimilarity values
y_pos = range(len(similarity_values))
ax.barh(y_pos, similarity_values)

# Customize the plot
ax.set_xlabel("Similarity")
ax.set_ylabel("Titles")
ax.set_yticks(y_pos)
ax.set_yticklabels(recommended_products["Title"])

# Invert the y-axis
ax.invert_yaxis()

# Adjust layout and display the plot
plt.tight_layout()
plt.show()
