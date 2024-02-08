import os
import base64
import streamlit as st
import pandas as pd
from utils.unpickle_file import unpickle_file

CBR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

def predict():
    st.title('Prediction App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")

        try:
            # Read the uploaded data
            df = pd.read_csv(uploaded_file)
            
            # Make predictions
            predictions = CBR.predict(df)
            
            # Add predictions column to the DataFrame
            df['predictions'] = predictions
            
            # Display DataFrame with predictions
            st.subheader("DataFrame with Predictions")
            st.write(df)
            
            # Optionally, you can also provide an option to download the modified DataFrame as a CSV file
            download_link = get_download_link(df, "data_with_predictions.csv")
            st.markdown(download_link, unsafe_allow_html=True)
            
            processing_spinner.text("Prediction completed successfully!")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

def get_download_link(df, filename="data.csv"):
    """Generates a download link for the DataFrame."""
    csv = df.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download Data with Predictions CSV File</a>'
    return href
