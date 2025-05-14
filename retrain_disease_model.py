"""
Script to retrain the disease prediction model.

Based on: https://www.kaggle.com/code/ray0911/disease-prediction-doctor-and-specialist-recomenda
"""

import os
import numpy as np
import pandas as pd
import pickle
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Define paths
DATASET_PATH = os.path.join("attached_assets", "Original_Dataset.csv")
SYMPTOM_WEIGHTS_PATH = os.path.join("attached_assets", "Symptom_Weights.csv")
MODEL_OUTPUT_PATH = os.path.join("attached_assets", "disease_prediction_model.pkl")

def load_and_preprocess_data():
    """Load and preprocess the disease dataset."""
    logger.info(f"Loading disease dataset from {DATASET_PATH}")
    
    try:
        # Load the dataset
        dataset = pd.read_csv(DATASET_PATH)
        logger.info(f"Dataset loaded with shape: {dataset.shape}")
        
        # Load the symptom weights
        symptom_weights = pd.read_csv(SYMPTOM_WEIGHTS_PATH, header=None)
        symptom_weights.columns = ['Symptom', 'Weight']
        symptom_weights['Symptom'] = symptom_weights['Symptom'].str.strip()
        logger.info(f"Symptom weights loaded with {len(symptom_weights)} entries")
        
        # Get the list of symptoms
        symptoms = symptom_weights['Symptom'].tolist()
        logger.info(f"Number of unique symptoms: {len(symptoms)}")
        
        # Create a data dictionary with all symptoms
        data_dict = {symptom: np.zeros(len(dataset)) for symptom in symptoms}
        data_dict['Disease'] = dataset['Disease']
        
        # Fill in the symptoms that are present
        for index, row in dataset.iterrows():
            for symptom_col in dataset.columns[1:]:  # Skip the Disease column
                if pd.notna(row[symptom_col]) and row[symptom_col] in symptoms:
                    data_dict[row[symptom_col]][index] = 1
        
        # Convert to DataFrame
        df = pd.DataFrame(data_dict)
        
        return df, symptoms
        
    except Exception as e:
        logger.error(f"Error loading and preprocessing data: {str(e)}")
        raise

def train_model(df, symptoms):
    """Train the disease prediction model."""
    logger.info("Training the model...")
    
    try:
        # Fix disease names with special characters 
        df['Disease'] = df['Disease'].replace({
            '(vertigo) Paroymsal  Positional Vertigo': 'Vertigo',
            'Dimorphic hemmorhoids(piles)': 'Dimorphic hemorrhoids'
        })
        
        # Normalize disease names by removing trailing spaces
        df['Disease'] = df['Disease'].str.strip()
        
        # Split the data into features and target
        X = df[symptoms]
        y = df['Disease']
        
        logger.info(f"Number of unique diseases: {len(y.unique())}")
        
        # Initialize and train the model
        model = GaussianNB()
        model.fit(X, y)
        
        logger.info("Model trained successfully")
        
        return model
        
    except Exception as e:
        logger.error(f"Error training model: {str(e)}")
        raise

def save_model(model):
    """Save the trained model to a pickle file."""
    logger.info(f"Saving model to {MODEL_OUTPUT_PATH}")
    
    try:
        with open(MODEL_OUTPUT_PATH, 'wb') as f:
            pickle.dump(model, f)
        logger.info("Model saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model: {str(e)}")
        raise

def main():
    """Main function to retrain the disease prediction model."""
    logger.info("Starting disease prediction model retraining process")
    
    try:
        # Load and preprocess the data
        df, symptoms = load_and_preprocess_data()
        
        # Train the model
        model = train_model(df, symptoms)
        
        # Save the model
        save_model(model)
        
        logger.info("Disease prediction model retraining completed successfully")
        
    except Exception as e:
        logger.error(f"Model retraining failed: {str(e)}")

if __name__ == "__main__":
    main()