import os
import csv
import json
import logging
import pandas as pd
from typing import List, Dict, Any, Union
# Helper functions for text processing
def extract_keywords(text):
    """Extract keywords from a text string."""
    # Simple implementation - split by spaces and remove common words
    common_words = {'a', 'an', 'the', 'is', 'are', 'was', 'were', 'be', 'been',
                  'have', 'has', 'had', 'do', 'does', 'did', 'can', 'could',
                  'will', 'would', 'should', 'i', 'you', 'he', 'she', 'it',
                  'we', 'they', 'my', 'your', 'his', 'her', 'its', 'our',
                  'their', 'this', 'that', 'these', 'those', 'what', 'which',
                  'who', 'whom', 'when', 'where', 'why', 'how', 'and', 'but',
                  'or', 'if', 'because', 'as', 'until', 'while', 'of', 'at',
                  'by', 'for', 'with', 'about', 'against', 'between', 'into',
                  'through', 'during', 'before', 'after', 'above', 'below',
                  'to', 'from', 'up', 'down', 'in', 'out', 'on', 'off', 'over',
                  'under', 'again', 'further', 'then', 'once', 'here', 'there',
                  'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                  'than', 'too', 'very', 'just', 'don', 'don\'t', 'should', 'now'}
    
    words = text.lower().split()
    return [word for word in words if word not in common_words]

def is_medical_query(text):
    """Determine if a query is medical in nature."""
    medical_keywords = {'disease', 'symptom', 'diagnosis', 'treatment', 'cure',
                       'medicine', 'drug', 'prescription', 'doctor', 'hospital',
                       'clinic', 'health', 'medical', 'pain', 'ache', 'hurt',
                       'blood', 'heart', 'lung', 'brain', 'nerve', 'muscle', 
                       'bone', 'joint', 'skin', 'infection', 'cancer', 'diabetes',
                       'pressure', 'high', 'low', 'fever', 'cough', 'cold', 'flu',
                       'virus', 'bacteria', 'chronic', 'acute', 'condition', 'disorder'}
    
    words = text.lower().split()
    return any(word in medical_keywords for word in words)

logger = logging.getLogger(__name__)

# Default dataset location - can be overridden by environment variable
DEFAULT_DATASET_PATH = os.environ.get('MEDICAL_DATASET_PATH', 'medical_dataset.csv')

def load_medical_dataset() -> List[Dict[str, str]]:
    """
    Load the medical dataset from a CSV or JSON file.
    
    Returns:
        List[Dict[str, str]]: A list of dictionaries containing medical data
    """
    dataset_path = DEFAULT_DATASET_PATH
    
    # Check if dataset exists
    if not os.path.exists(dataset_path):
        logger.warning(f"Medical dataset not found at {dataset_path}. Using empty dataset.")
        return []
    
    try:
        # Try loading as CSV first
        if dataset_path.endswith('.csv'):
            df = pd.read_csv(dataset_path)
            logger.debug(f"Loaded CSV dataset with {len(df)} entries")
            return df.to_dict('records')
        
        # Try loading as JSON
        elif dataset_path.endswith('.json'):
            with open(dataset_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            logger.debug(f"Loaded JSON dataset with {len(data)} entries")
            return data
        
        # Fallback to trying both formats
        else:
            try:
                df = pd.read_csv(dataset_path)
                logger.debug(f"Loaded CSV dataset with {len(df)} entries")
                return df.to_dict('records')
            except:
                with open(dataset_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                logger.debug(f"Loaded JSON dataset with {len(data)} entries")
                return data
    
    except Exception as e:
        logger.error(f"Error loading medical dataset: {e}")
        return []

def create_medical_context(query: str, medical_data: List[Dict[str, str]]) -> str:
    """
    Create relevant medical context based on the user's query and the medical dataset.
    
    Args:
        query: The user's query
        medical_data: The medical dataset
        
    Returns:
        str: Relevant medical context or empty string if none found
    """
    if not medical_data or not is_medical_query(query):
        return ""
    
    # Extract keywords from the query
    keywords = extract_keywords(query)
    if not keywords:
        return ""
    
    # Find relevant medical information
    relevant_entries = []
    
    for entry in medical_data:
        # Skip entries with missing keys
        if not all(k in entry for k in ['question', 'answer']):
            continue
            
        # Check if any keyword matches
        entry_text = f"{entry.get('question', '')} {entry.get('answer', '')}"
        if any(keyword in entry_text.lower() for keyword in keywords):
            relevant_entries.append(entry)
    
    # Limit to top 3 most relevant entries to avoid context length issues
    relevant_entries = relevant_entries[:3]
    
    # Format the context
    if relevant_entries:
        context = "Medical information:\n"
        for i, entry in enumerate(relevant_entries, 1):
            question = entry.get('question', 'Unknown question')
            answer = entry.get('answer', 'Unknown answer')
            context += f"{i}. Q: {question}\nA: {answer}\n\n"
        return context
    
    return ""

def download_kaggle_dataset(kaggle_dataset_url: str, target_directory: str = '.') -> str:
    """
    Function to download a dataset from Kaggle.
    Note: This requires the kaggle API and proper authentication.
    
    Args:
        kaggle_dataset_url: The URL of the Kaggle dataset
        target_directory: Directory to save the dataset to
        
    Returns:
        str: Path to the downloaded dataset
    """
    try:
        import kaggle
        import os
        import json
        from pathlib import Path
        
        # Set up Kaggle credentials from environment variables
        kaggle_username = os.environ.get('KAGGLE_USERNAME')
        kaggle_key = os.environ.get('KAGGLE_KEY')
        
        if not kaggle_username or not kaggle_key:
            logger.error("Kaggle credentials not found in environment variables.")
            return os.path.join(target_directory, 'medical_dataset.csv')
            
        # Create Kaggle API credentials file
        kaggle_dir = Path.home() / '.kaggle'
        kaggle_dir.mkdir(exist_ok=True)
        
        with open(kaggle_dir / 'kaggle.json', 'w') as f:
            json.dump({
                'username': kaggle_username,
                'key': kaggle_key
            }, f)
            
        # Set permissions
        os.chmod(kaggle_dir / 'kaggle.json', 0o600)
        
        # Extract dataset name from URL
        # Example URL: https://www.kaggle.com/datasets/yousefsaeedian/ai-medical-chatbot/data
        dataset_parts = kaggle_dataset_url.split('/')
        if 'datasets' in dataset_parts:
            idx = dataset_parts.index('datasets')
            if idx + 2 < len(dataset_parts):
                username = dataset_parts[idx + 1]
                dataset_name = dataset_parts[idx + 2]
                dataset_ref = f"{username}/{dataset_name}"
                
                logger.info(f"Downloading dataset {dataset_ref} from Kaggle...")
                kaggle.api.dataset_download_files(dataset_ref, path=target_directory, unzip=True)
                logger.info("Download completed successfully.")
                
                # Find dataset files
                dataset_files = [f for f in os.listdir(target_directory) 
                               if f.endswith('.csv') or f.endswith('.json')]
                
                if dataset_files:
                    # If there are multiple files, prefer CSV format
                    csv_files = [f for f in dataset_files if f.endswith('.csv')]
                    if csv_files:
                        return os.path.join(target_directory, csv_files[0])
                    else:
                        return os.path.join(target_directory, dataset_files[0])
                else:
                    logger.error("No dataset files found after download.")
        
        return os.path.join(target_directory, 'medical_dataset.csv')
    except Exception as e:
        logger.error(f"Error downloading Kaggle dataset: {e}")
        return os.path.join(target_directory, 'medical_dataset.csv')

def prepare_medical_dataset(raw_dataset_path: str, output_path: str = 'medical_dataset.csv') -> str:
    """
    Prepare the raw medical dataset for use with the chatbot.
    
    Args:
        raw_dataset_path: Path to the raw dataset file
        output_path: Path to save the processed dataset
        
    Returns:
        str: Path to the processed dataset
    """
    try:
        # Load the dataset
        if raw_dataset_path.endswith('.csv'):
            df = pd.read_csv(raw_dataset_path)
        elif raw_dataset_path.endswith('.json'):
            df = pd.read_json(raw_dataset_path)
        else:
            raise ValueError(f"Unsupported file format: {raw_dataset_path}")
        
        logger.info(f"Loaded dataset with {len(df)} entries and columns: {df.columns.tolist()}")
        
        # Process the dataset based on its structure
        # This is a generic implementation and may need adjustment based on actual dataset structure
        if 'question' in df.columns and 'answer' in df.columns:
            # Dataset already has the right format
            processed_df = df[['question', 'answer']]
        else:
            # Try to adapt the dataset format
            # The actual implementation depends on the specific dataset structure
            logger.warning("Dataset format doesn't match expected structure, attempting to adapt")
            
            # Here we try to map any common column patterns to our expected format
            column_mapping = {}
            for col in df.columns:
                col_lower = col.lower()
                if any(q in col_lower for q in ['question', 'query', 'prompt']):
                    column_mapping['question'] = col
                elif any(a in col_lower for a in ['answer', 'response', 'reply']):
                    column_mapping['answer'] = col
            
            if len(column_mapping) == 2:
                processed_df = df.rename(columns=column_mapping)[['question', 'answer']]
            else:
                raise ValueError(f"Could not determine appropriate column mapping from {df.columns}")
        
        # Save the processed dataset
        processed_df.to_csv(output_path, index=False)
        logger.info(f"Processed dataset saved to {output_path}")
        
        return output_path
        
    except Exception as e:
        logger.error(f"Error preparing medical dataset: {e}")
        raise
