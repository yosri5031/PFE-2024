import torch
from transformers import DistilBertModel, DistilBertTokenizer
from torch import nn
import random
import os

# Options prédéfinies pour chaque catégorie de préférence
options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science","Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}

categories = list(options.keys())

# Create a dictionary to map each option to its category
option_to_category = {option: category for category, options_list in options.items() for option in options_list}

# Create a dictionary to map each option to a unique index within its category
classes = {category: {option: i for i, option in enumerate(options_list)} for category, options_list in options.items()}

# Exemple de génération de phrase
def generate_random_phrase(label):
    words = ["I", "feel", "am", "This", "is", label]
    return " ".join(random.sample(words, random.randint(2,5)))

# Generate training data
train_data = []
for category, values in options.items():
    for value in values:
        phrase = generate_random_phrase(value)
        train_data.append((phrase, value))

# Ajouter du bruit
for i in range(len(train_data)*10):
    phrase = random.choice(train_data)[0]
    label = random.choice(list(option_to_category.keys()))
    train_data.append((phrase, label))

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Initialize tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
model = DistilBertModel.from_pretrained('distilbert-base-uncased')

# Initialize classifiers (one for each category)
classifiers = nn.ModuleDict({
    category: nn.Linear(model.config.hidden_size, len(options[category]))
    for category in categories
})

# Initialize optimizer
optimizer = torch.optim.Adam(list(model.parameters()) + [p for c in classifiers.values() for p in c.parameters()], lr=1e-5)

# Loss function
loss_fn = nn.CrossEntropyLoss()

# Hyperparameters
accumulation_steps = 32
num_epochs = 5

# Training function
def train_model():
    model.train()
    for classifier in classifiers.values():
        classifier.train()

    for epoch in range(num_epochs):
        total_loss = 0
        for i, (phrase, label) in enumerate(train_data):
            inputs = tokenizer(phrase, return_tensors="pt", padding=True, truncation=True)
            category = option_to_category[label]
            label_index = classes[category][label]
            label_tensor = torch.tensor([label_index])

            output = model(**inputs).last_hidden_state[:, 0, :]  # Use CLS token
            logits = classifiers[category](output)

            # Calculate loss
            loss = loss_fn(logits, label_tensor)
            loss = loss / accumulation_steps  # Normalize loss
            total_loss += loss.item()

            # Backward pass
            loss.backward()

            # Optimization step
            if (i + 1) % accumulation_steps == 0 or (i + 1) == len(train_data):
                optimizer.step()
                optimizer.zero_grad()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {i+1}/{len(train_data)}, Loss: {total_loss:.4f}")
                total_loss = 0

# Save model function
def save_model(path):
    torch.save({
        'model_state_dict': model.state_dict(),
        'classifiers_state_dict': classifiers.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)

# Load model function
def load_model(path):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    classifiers.load_state_dict(checkpoint['classifiers_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

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

# Train and save the model
if not os.path.exists('model.pth'):
    train_model()
    save_model('model.pth')
    print("Model trained and saved.")
else:
    load_model('model.pth')
    print("Model loaded from file.")