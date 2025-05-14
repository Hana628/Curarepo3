import pandas as pd

# Load the original dataset
data = pd.read_csv('attached_assets/Original_Dataset.csv')

# Create set to store all unique symptoms
all_symptoms = set()

# Extract symptoms from symptom columns
for col in data.columns[1:]:  # Skip the 'Disease' column
    for symptom in data[col].dropna():
        # Clean the symptom
        cleaned = symptom.strip()
        if cleaned:
            all_symptoms.add(cleaned)

# Print the total number of unique symptoms and the sorted list
print(f"Total unique symptoms in Original_Dataset.csv: {len(all_symptoms)}")
print("Unique symptoms:")
for symptom in sorted(all_symptoms):
    print(f"- {symptom}")
