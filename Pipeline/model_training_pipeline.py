import os
import base64
import pickle
import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file

CBR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

def train_catboost():
    st.title('CatBoost Model Training App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")
        
        try:
            # Read the uploaded data
            df = pd.read_csv(uploaded_file)
            
            # Check if 'price' column is present
            if 'price' not in df.columns:
                st.error("The 'price' column is missing in the uploaded file.")
                return
            
            # Split features and target
            features = df.drop('price', axis=1)
            target = df['price']
            
            # Train the CatBoost model
            CBR.fit(features, target)
            
            # Display success message
            processing_spinner.text("Model trained successfully!")
            
            # Add a button to download the trained model
            trained_model_file = "trained_catboost_model.pkl"
            trained_model_link = get_model_download_link(CBR, trained_model_file)
            st.markdown(trained_model_link, unsafe_allow_html=True)
        
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def get_model_download_link(model, filename="trained_model.pkl"):
    """Generates a download link for the trained model."""
    model_binary = base64.b64encode(pickle.dumps(model)).decode()
    href = f'<a href="data:application/octet-stream;base64,{model_binary}" download="{filename}">Download Trained Model</a>'
    return href

# Run the Streamlit app
if __name__ == "__main__":
    train_catboost()
