"""
Script to retrain the disease prediction model following the exact approach from Kaggle.
Based on: https://www.kaggle.com/code/ray0911/disease-prediction-doctor-and-specialist-recomenda
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
DATASET_PATH = os.path.join("attached_assets", "Original_Dataset.csv")
MODEL_OUTPUT_PATH = os.path.join("attached_assets", "disease_prediction_model.pkl")

def main():
    """Main function to retrain the disease prediction model following the Kaggle approach."""
    logger.info("Starting disease prediction model retraining process (Kaggle approach)")
    
    try:
        # Step 1: Load the dataset
        logger.info(f"Loading disease dataset from {DATASET_PATH}")
        dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded with shape: {dataset.shape}")
        
        # Step 2: Data preprocessing
        logger.info("Preprocessing data...")
        
        # Clean disease names by removing trailing spaces and fixing typos
        dataset['Disease'] = dataset['Disease'].str.strip()
        
        # Fix specific disease names that might have typos or variations
        dataset['Disease'] = dataset['Disease'].replace({
            '(vertigo) Paroymsal  Positional Vertigo': 'Vertigo',
            'Dimorphic hemmorhoids(piles)': 'Dimorphic hemorrhoids'
        })
        
        # Step 3: Get list of unique diseases and symptoms
        diseases = dataset['Disease'].unique()
        logger.info(f"Number of unique diseases: {len(diseases)}")
        logger.info(f"Diseases: {diseases}")
        
        # Extract symptoms from columns (except the first one which is 'Disease')
        all_symptoms_list = []
        for col in dataset.columns[1:]:
            all_symptoms_list.extend(dataset[col].dropna().unique())
        
        # Clean symptoms and remove duplicates
        all_symptoms_list = [symptom.strip() for symptom in all_symptoms_list if isinstance(symptom, str)]
        all_symptoms = sorted(list(set(all_symptoms_list)))
        logger.info(f"Number of unique symptoms: {len(all_symptoms)}")
        
        # Step 4: Create symptom-encoded dataframe following the Kaggle approach
        logger.info("Creating encoded dataframe...")
        
        # Create dictionary to map symptoms to 0/1 values
        symptom_columns = {symptom: [0] * len(dataset) for symptom in all_symptoms}
        symptom_columns['Disease'] = dataset['Disease']
        
        # Create dataframe from dictionary
        encoded_df = pd.DataFrame(symptom_columns)
        
        # Fill the dataframe with 1s where symptoms are present
        for index, row in dataset.iterrows():
            for col in dataset.columns[1:]:  # Skip the Disease column
                if pd.notna(row[col]):
                    symptom = row[col].strip()
                    if symptom in all_symptoms:
                        encoded_df.at[index, symptom] = 1
        
        logger.info(f"Encoded dataframe created with shape: {encoded_df.shape}")
        
        # Step 5: Split features and target variable
        X = encoded_df.drop('Disease', axis=1)
        y = encoded_df['Disease']
        
        # Step 6: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
        
        # Step 7: Train the Decision Tree Classifier (following the Kaggle approach)
        logger.info("Training Decision Tree Classifier...")
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        
        # Step 8: Evaluate model
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy on test set: {accuracy * 100:.2f}%")
        
        # Step 9: Test model with sample symptoms
        logger.info("Testing model with sample symptom combinations:")
        test_cases = {
            "Headache": {"headache": 1},
            "Headache and Nausea": {"headache": 1, "nausea": 1},
            "Fever and Cough": {"high_fever": 1, "cough": 1},
            "Skin Rash and Itching": {"skin_rash": 1, "itching": 1}
        }
        
        for case_name, symptoms in test_cases.items():
            # Create test input (all zeros)
            test_input = pd.DataFrame({symptom: [0] for symptom in all_symptoms})
            
            # Set the specified symptoms to 1
            for symptom, value in symptoms.items():
                if symptom in test_input.columns:
                    test_input[symptom] = value
            
            # Make prediction
            prediction = classifier.predict(test_input)
            probabilities = classifier.predict_proba(test_input)
            confidence = round(max(probabilities[0]) * 100, 2)
            
            logger.info(f"Case: {case_name}")
            logger.info(f"  Prediction: {prediction[0]}")
            logger.info(f"  Confidence: {confidence}%")
        
        # Step 10: Save the model
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(classifier, f)
        
        # Save list of symptoms used for model training in exact column order
        symptoms_path = "models/symptoms_list.py"
        with open(symptoms_path, 'w') as f:
            f.write("# Auto-generated list of symptoms from disease prediction model training\n")
            f.write("# These symptoms are in the exact order expected by the model\n")
            f.write("ALL_SYMPTOMS = [\n")
            for i, symptom in enumerate(X.columns):
                if i < len(X.columns) - 1:
                    f.write(f"    \"{symptom}\",\n")
                else:
                    f.write(f"    \"{symptom}\"\n")
            f.write("]\n")
            
            # Add a special comment to mark when this was last updated
            import datetime
            f.write(f"\n# Last updated: {datetime.datetime.now()}\n")
            f.write(f"# Model accuracy: {accuracy * 100:.2f}%\n")
        
        logger.info(f"Saved symptoms list in exact model order to {symptoms_path}")
        logger.info("Disease prediction model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")

if __name__ == "__main__":
    main()