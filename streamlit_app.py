import pandas as pd
import streamlit as st

from Pipeline.data_processing_pipeline import data_processing
from Pipeline.model_training_pipeline import train_catboost
from Pipeline.model_prediction_pipeline import predict

def main():
    st.title("Product Price Prediction Model")

    # Sidebar navigation
    page = st.sidebar.selectbox("Select a page", ["Data Preprocessing", "Model Training", "Prediction"], index=2)

    if page == "Data Preprocessing":
        data_processing()
    elif page == "Model Training":
        train_catboost()
    elif page == "Prediction":
        predict()

# Run the Streamlit app
if __name__ == "__main__":
    main()