from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
from preprocess import preprocessed_data
from predict import predict_dict

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

import os

directory = os.getcwd()
file_path = os.path.join(directory, 'python_server/Bicycle_Thefts_Open_Data.csv')
bike_data = pd.read_csv(file_path)

# API Routes
# Get options of the location names in the dataset ,the bike_type,the bike cost ,premises, neighbourhood
@app.route('/options', methods=['GET'])
def get_options():
    options = {
        'DIVISION': bike_data['DIVISION'].unique().tolist(),
        'LOCATION_TYPE': bike_data['LOCATION_TYPE'].unique().tolist(),
        'PREMISES_TYPE': bike_data['PREMISES_TYPE'].unique().tolist(),
        'NEIGHBOURHOOD_158': bike_data['NEIGHBOURHOOD_158'].unique().tolist(),
        'BIKE_TYPE': bike_data['BIKE_TYPE'].unique().tolist(),
    }
    response = jsonify(options)
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response

# Predict the status of a bike theft
@app.route('/predict', methods=['POST'])
def predict():
    print("Predicting...")

    # Get the input data (JSON)
    data = request.get_json()
    print(data)

    # 

    # Use model to predict outcome
    prediction = predict_dict(data)

    return jsonify({'prediction': prediction}) 
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)