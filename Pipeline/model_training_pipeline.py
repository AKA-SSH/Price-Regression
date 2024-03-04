import os
import base64
import pickle
import pandas as pd
from catboost import CatBoostRegressor  # Assuming CatBoost is used for training
from utils.unpickle_file import unpickle_file

CBR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

def train_catboost(uploaded_file):
    try:
        # Read the uploaded data
        df = pd.read_csv(uploaded_file)
        
        # Check if 'price' column is present
        if 'price' not in df.columns:
            return None, "The 'price' column is missing in the uploaded file."
        
        # Split features and target
        features = df.drop('price', axis=1)
        target = df['price']
        
        # Train the CatBoost model
        CBR.fit(features, target)
        
        # Return trained model
        return CBR, "Model trained successfully!"
    
    except Exception as e:
        return None, f"An error occurred: {str(e)}"

def save_model(model, filename="trained_model.pkl"):
    """Saves the trained model to a file."""
    with open(filename, 'wb') as f:
        pickle.dump(model, f)
