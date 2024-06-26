import spacy
import random

# Load the spaCy English model
nlp = spacy.load("en_core_web_sm")

options = {
    "emotions": ["Love and romance", "Happiness and joy", "Peace and tranquility", "Inspiration and motivation", "Sentimental and nostalgic"],
    "occasion": ["Birthday", "Anniversary", "Housewarming", "Holiday celebration", "Graduation", "Retirement", "Baby shower", "Engagement", "Prom", "Valentine's Day", "New Year's Eve", "Easter", "Mother's Day", "Father's Day", "Wedding", "Bachelorette party", "Bachelor party", "Job promotion", "Thank you", "Get well soon", "Sympathy", "Congratulations", "Good luck", "Just because"],
    "interests": ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science","Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"],
    "audience": ["Child Audience", "Teen Audience", "Adult Audience", "Senior Audience"],
    "personality": ["Deconstructionist Design", "Motorhead Chic", "Spiritual Stylista", "Narrative Garb", "Artisan Aesthetic", "Anatomical Appeal", "Symbolic Style", "Cosmic Chic", "Flora Fashion", "Animal Admirer"]
}

def search_options(user_input, options):
    processed_input = nlp(user_input.lower())
    matched_options = {}

    for category, category_options in options.items():
        category_matches = []
        for option in category_options:
            if nlp(option.lower()).similarity(processed_input) > 0.7:
                category_matches.append(option)
        
        if category_matches:
            matched_options[category] = category_matches

    # Generate creative options
    for category in options.keys():
        if category not in matched_options:
            creative_option = generate_creative_option(user_input, category)
            if creative_option:
                matched_options[category] = [creative_option]

    return matched_options

def generate_creative_option(user_input, category):
    # Implement your creative option generation logic here
    # You can use the user input and category to generate a unique creative option
    # Return the generated creative option or None if not applicable

    # Example: Generating a creative option by combining user input and category
    user_keywords = [token.text.lower() for token in nlp(user_input) if not token.is_stop]
    random.shuffle(user_keywords)
    creative_option = '-'.join(user_keywords[:2]) + '-' + category.replace(' ', '-')
    return creative_option

# Example usage
user_input = "I want a gift for a birthday celebration with a musical theme"
matched_options = search_options(user_input, options)
print(matched_options)

# Example 2
#user_input2 ="kids play with football i want gift for them"
#kids_options = search_options(user_input, options)
#print(kids_options)