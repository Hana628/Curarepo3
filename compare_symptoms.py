import pandas as pd
import sys
from models.symptoms_list import ALL_SYMPTOMS

# Load the original dataset
data = pd.read_csv('attached_assets/Original_Dataset.csv')

# Create set to store all unique symptoms from original dataset
original_symptoms = set()

# Extract symptoms from symptom columns
for col in data.columns[1:]:  # Skip the 'Disease' column
    for symptom in data[col].dropna():
        # Clean the symptom
        cleaned = symptom.strip()
        if cleaned:
            original_symptoms.add(cleaned)

# Create set from current symptoms list
current_symptoms = set(ALL_SYMPTOMS)

# Find symptoms in original dataset that are not in our current list
missing_symptoms = original_symptoms - current_symptoms
# Find symptoms in our current list that are not in the original dataset
extra_symptoms = current_symptoms - original_symptoms

# Find symptoms that need formatting fixes (e.g., "dischromic _patches" vs "dischromic_patches")
formatting_issues = set()
for symptom in original_symptoms:
    # Check for common formatting issues
    if ' ' in symptom and symptom.replace(' ', '_') in current_symptoms:
        formatting_issues.add((symptom, symptom.replace(' ', '_')))
    if '_' in symptom and symptom.replace('_', ' ') in current_symptoms:
        formatting_issues.add((symptom, symptom.replace('_', ' ')))

print(f"Total symptoms in Original_Dataset.csv: {len(original_symptoms)}")
print(f"Total symptoms in our current list: {len(current_symptoms)}")

if missing_symptoms:
    print("\nSymptoms in original dataset but missing from our list:")
    for symptom in sorted(missing_symptoms):
        print(f"- {symptom}")
else:
    print("\nNo missing symptoms.")

if extra_symptoms:
    print("\nSymptoms in our list but not in the original dataset:")
    for symptom in sorted(extra_symptoms):
        print(f"- {symptom}")
else:
    print("\nNo extra symptoms.")

if formatting_issues:
    print("\nSymptoms with formatting differences:")
    for original, current in sorted(formatting_issues):
        print(f"- Original: '{original}' vs Current: '{current}'")
else:
    print("\nNo formatting issues detected.")
