import pandas as pd
import csv

def delete_empty_title_rows(csv_file):
    """
    Deletes rows from a CSV file that have an empty Title column.

    Args:
        csv_file (str): Path to the CSV file.

    Returns:
        None
    """

    try:
        # Read the CSV file into a DataFrame
        df = pd.read_csv(csv_file)

        # Drop rows with empty Title column
        df.dropna(subset=['Title'], inplace=True)

        # Save the modified DataFrame back to the CSV file
        df.to_csv(csv_file, index=False)
        print(f"Successfully deleted rows with empty Title from {csv_file}.")
    except FileNotFoundError:
        print(f"Error: CSV file '{csv_file}' not found.")

# Example usage:
delete_empty_title_rows('shopify.csv')

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