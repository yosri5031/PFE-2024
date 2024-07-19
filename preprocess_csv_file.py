import pandas as pd
import numpy as np

# Read the CSV file
df = pd.read_csv('input.csv')

# Remove empty rows
df.dropna(how='all', inplace=True)

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# List of keywords to check in the description
keywords = ["Architecture", "Cars & Vehicules", "Religious", "Fiction", "Tools", "Human Organes", "Symbols", "Astronomy", "Plants", "Animals", "Art", "Celebrities", "Flags", "HALLOWEEN", "Quotes", "Sports", "Thanksgiving", "Maps", "Romance", "Kitchen", "Musical Instruments", "Black Lives Matter", "Cannabis", "Vegan", "Birds", "Dinosaurs", "rock and roll", "Firearms", "Dances", "Sailing", "Jazz", "Christmas", "Greek Methology", "Life Style", "Planes", "Plants", "Vintage", "Alphabets", "Weapons", "Insects", "Games", "JEWELRY", "Science", "Medical", "Travel", "Cats", "Circus", "Lucky charms", "Wild West", "Dogs"]

# Function to check description and update tags
def update_tags(row):
    description = str(row['Description']).lower()
    
    # Check if Tags is a float (NaN) and convert to empty list if so
    if isinstance(row['Tags'], float) and np.isnan(row['Tags']):
        current_tags = []
    else:
        current_tags = [tag.strip() for tag in str(row['Tags']).split(',') if tag.strip()]
    
    for keyword in keywords:
        if keyword.lower() in description and keyword not in current_tags:
            current_tags.append(keyword)
    
    row['Tags'] = ', '.join(current_tags) if current_tags else np.nan
    return row

# Apply the update_tags function to each row
df = df.apply(update_tags, axis=1)

# Save the processed data to a new CSV file
df.to_csv('output_preprocessed.csv', index=False)