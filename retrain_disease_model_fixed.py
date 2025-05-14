"""
Script to retrain the disease prediction model with proper symptom handling.
Based on the Kaggle notebook but with improved preprocessing.
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
SYMPTOM_WEIGHTS_PATH = os.path.join("attached_assets", "Symptom_Weights.csv")
MODEL_OUTPUT_PATH = os.path.join("attached_assets", "disease_prediction_model.pkl")

def clean_symptom(symptom):
    """Clean a symptom string by removing extra spaces and fixing common issues."""
    if not isinstance(symptom, str):
        return ""
    
    # Remove leading/trailing spaces and convert to lowercase
    cleaned = symptom.strip().lower()
    
    # Fix specific formatting issues found in the dataset
    if "dischromic _patches" in cleaned:
        cleaned = cleaned.replace("dischromic _patches", "dischromic_patches")
    
    if "foul_smell_of urine" in cleaned:
        cleaned = cleaned.replace("foul_smell_of urine", "foul_smell_of_urine")
    
    if "spotting_ urination" in cleaned:
        cleaned = cleaned.replace("spotting_ urination", "spotting_urination")
    
    return cleaned

def main():
    """Main function to retrain the disease prediction model."""
    logger.info("Starting disease prediction model retraining process with improved preprocessing")
    
    try:
        # Step 1: Load and clean the dataset
        logger.info(f"Loading disease dataset from {DATASET_PATH}")
        dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded with shape: {dataset.shape}")
        
        # Clean disease names
        dataset['Disease'] = dataset['Disease'].str.strip()
        
        # Fix specific disease names that might have typos or variations
        dataset['Disease'] = dataset['Disease'].replace({
            '(vertigo) Paroymsal  Positional Vertigo': 'Vertigo',
            'Dimorphic hemmorhoids(piles)': 'Dimorphic hemorrhoids'
        })
        
        # Step 2: Load symptom weights to get a complete list of symptoms
        logger.info(f"Loading symptom weights from {SYMPTOM_WEIGHTS_PATH}")
        symptom_weights = pd.read_csv(SYMPTOM_WEIGHTS_PATH, header=None, names=['symptom', 'weight'])
        
        # Clean symptom names
        symptom_weights['symptom'] = symptom_weights['symptom'].apply(clean_symptom)
        
        # Get list of all symptoms from the weights file (cleaner than extracting from dataset)
        all_symptoms = sorted(list(symptom_weights['symptom'].unique()))
        logger.info(f"Number of unique symptoms from weights file: {len(all_symptoms)}")
        
        # Step 3: Create symptom-encoded dataframe
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
                    # Clean the symptom string before matching
                    symptom = clean_symptom(row[col])
                    if symptom in all_symptoms:
                        encoded_df.at[index, symptom] = 1
        
        logger.info(f"Encoded dataframe created with shape: {encoded_df.shape}")
        
        # Step 4: Split features and target variable
        X = encoded_df.drop('Disease', axis=1)
        y = encoded_df['Disease']
        
        # Log feature names for debugging
        logger.info(f"Feature column names: {X.columns.tolist()[:5]}... (showing first 5)")
        
        # Step 5: Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        logger.info(f"Training set size: {X_train.shape[0]}, Testing set size: {X_test.shape[0]}")
        
        # Step 6: Train the Decision Tree Classifier
        logger.info("Training Decision Tree Classifier...")
        classifier = DecisionTreeClassifier(random_state=42)
        classifier.fit(X_train, y_train)
        
        # Step 7: Evaluate model
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        logger.info(f"Model accuracy on test set: {accuracy * 100:.2f}%")
        
        # Step 8: Test with some common symptom combinations
        logger.info("Testing model with sample symptom combinations:")
        test_cases = {
            "Headache": ["headache"],
            "Headache and Nausea": ["headache", "nausea"],
            "Fever and Cough": ["high_fever", "cough"],
            "Skin Rash and Itching": ["skin_rash", "itching"]
        }
        
        for case_name, symptoms in test_cases.items():
            # Create a test input DataFrame with the same columns as X
            test_input = pd.DataFrame(columns=X.columns)
            test_input.loc[0] = 0  # Initialize all to 0
            
            # Set the symptoms to 1
            for symptom in symptoms:
                if symptom in test_input.columns:
                    test_input[symptom] = 1
            
            # Make prediction
            prediction = classifier.predict(test_input)
            probabilities = classifier.predict_proba(test_input)
            confidence = round(max(probabilities[0]) * 100, 2)
            
            logger.info(f"Case: {case_name}")
            logger.info(f"  Prediction: {prediction[0]}")
            logger.info(f"  Confidence: {confidence}%")
        
        # Step 9: Save the model and feature column names
        logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
        
        # Create a dictionary to save both the model and feature names
        model_data = {
            'model': classifier,
            'features': X.columns.tolist()
        }
        
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model_data, f)
        
        # Save list of symptoms used for model training
        symptoms_path = "models/symptoms_list.py"
        with open(symptoms_path, 'w') as f:
            f.write("# Auto-generated list of symptoms from disease prediction model training\n")
            f.write("# IMPORTANT: These must be in exact order used during model training\n")
            f.write("ALL_SYMPTOMS = [\n")
            for i, symptom in enumerate(X.columns):
                if i < len(X.columns) - 1:
                    f.write(f"    \"{symptom}\",\n")
                else:
                    f.write(f"    \"{symptom}\"\n")
            f.write("]\n")
            
            # Add metadata
            import datetime
            f.write(f"\n# Last updated: {datetime.datetime.now()}\n")
            f.write(f"# Model accuracy: {accuracy * 100:.2f}%\n")
            f.write(f"# Number of symptoms: {len(X.columns)}\n")
        
        logger.info(f"Saved symptoms list to {symptoms_path}")
        logger.info("Disease prediction model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()