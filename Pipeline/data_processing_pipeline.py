import os
import base64
import pandas as pd
import streamlit as st
from utils.unpickle_file import unpickle_file

label_encoder_file_path = os.path.join('artifacts', 'label-encoder.pkl')
min_max_scaler_file_path = os.path.join('artifacts', 'min-max-scaler.pkl')
yeo_johnson_transformer_file_path = os.path.join('artifacts', 'yeo-johnson-transformer.pkl')

LE = unpickle_file(label_encoder_file_path)
MM = unpickle_file(min_max_scaler_file_path)
YJ = unpickle_file(yeo_johnson_transformer_file_path)

def clean_raw_data(raw_data):
    raw_dataframe = pd.read_csv(raw_data)
    selected_columns = ['stars', 'reviews', 'price', 'category', 'isBestSeller', 'boughtInLastMonth']
    dataframe = raw_dataframe[selected_columns].copy()

    dataframe.loc[:, 'estimatedSaleCount'] = round(66.67 * dataframe['reviews'])
    dataframe.loc[:, 'estimatedRevenue'] = dataframe['price'] * dataframe['estimatedSaleCount']
    dataframe = dataframe.drop('estimatedSaleCount', axis=1)
    bin_edges = [0, 10, 100, 1000, 10_000, 100_000, float('inf')]
    bin_labels = [0, 1, 2, 3, 4, 5]
    dataframe['priceOptimality'] = pd.cut(dataframe.reviews, bins=bin_edges, labels=bin_labels, right=False)
    
    dataframe['isBestSeller'] = dataframe['isBestSeller'].astype(int)
    dataframe['category'] = LE.fit_transform(dataframe['category'])

    return dataframe

def data_manipulation(cleaned_data):
    features, target = cleaned_data.drop('price', axis=1), cleaned_data['price']
    
    transfomed_features = YJ.fit_transform(features)
    scaled_features = MM.fit_transform(transfomed_features)
    features = pd.DataFrame(data=scaled_features, index=features.index, columns=features.columns)

    return pd.concat([features, target], axis=1)  # Combine features and target into a single DataFrame

def data_processing():
    st.title('Data Processing App')
    uploaded_file = st.file_uploader("Upload a CSV file", type=["csv", "xls", "xlsx"])

    if uploaded_file is not None:
        processing_spinner = st.empty()
        processing_spinner.text("Processing uploaded data...")
        
        # Clean the uploaded data
        cleaned_data = clean_raw_data(uploaded_file)
        
        # Perform data manipulation
        processed_data = data_manipulation(cleaned_data)
        
        processing_spinner.text("Data processed successfully!")
        
        # Display processed data
        st.subheader('Processed Data')
        st.write('Combined Features and Target:')
        st.write(processed_data.head())
        
        # Download link for processed data
        st.subheader('Download Processed Data')
        
        csv_file_processed = processed_data.to_csv(index=False)
        b64_processed = base64.b64encode(csv_file_processed.encode()).decode()
        href_processed = f'<a href="data:file/csv;base64,{b64_processed}" download="processed_data.csv">Download Processed Data CSV File</a>'
        st.markdown(href_processed, unsafe_allow_html=True)
