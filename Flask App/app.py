from flask import Flask, render_template, request, send_file
from Pipeline.data_processing_pipeline import data_processing
from Pipeline.model_training_pipeline import train_catboost, save_model, unpickle_file
import pandas as pd
import os

app = Flask(__name__)

# Load the model
CBR = unpickle_file(os.path.join('artifacts', 'model.pkl'))

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/preprocessing')
def preprocessing():
    return render_template('preprocessing.html')

@app.route('/training')
def training():
    return render_template('model_training.html')

@app.route('/process_data', methods=['POST'])
def process_data():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            processed_data = data_processing(uploaded_file)
            processed_table = processed_data.head().to_html(index=False)
            return render_template('preprocessing.html', processed_data=processed_table)
        else:
            return "No file uploaded!"

@app.route('/download_processed_data', methods=['GET'])
def download_processed_data():
    processed_data = request.args.get('data')
    return send_file(processed_data, as_attachment=True, attachment_filename='processed_data.csv', mimetype='text/csv')

@app.route('/train_model', methods=['POST'])
def train_model():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            global CBR
            CBR, message = train_catboost(uploaded_file)
            
            if CBR is not None:
                # Save the trained model
                save_model(CBR)
                return "Model trained successfully!"
            else:
                return message
        else:
            return "No file uploaded!"

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        uploaded_file = request.files['file']
        
        if uploaded_file.filename != '':
            try:
                # Read the uploaded data
                df = pd.read_csv(uploaded_file)
                
                # Make predictions
                predictions = CBR.predict(df)
                
                # Add predictions column to the DataFrame
                df['predictions'] = predictions
                
                # Display DataFrame with predictions
                return df.to_html(index=False)
                
            except Exception as e:
                return f"An error occurred: {str(e)}"
        else:
            return "No file uploaded!"

if __name__ == '__main__':
    app.run(debug=True)
