from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

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
    print("Predicting")

    # Get the input data
    data = request.get_json()
    print(data)

    # Convert the input data into a DataFrame
    # input_data = pd.DataFrame(data, index=[0])

    # Preprocess the input data
    scaled_numerical_data = scaler.fit_transform(data[numerical_columns])

    # Create a DataFrame for scaled numerical data
    scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns, index=data.index)

    # Replace numerical columns in the original dataset with scaled values
    scaled_bike_dataf = encoded_bike_dataf.copy()
    scaled_bike_dataf[numerical_columns] = scaled_numerical_df

    # Verify the scaled dataset
    scaled_bike_dataf.head()

    # input_data = preprocessor.preprocess(input_data)

    # Make predictions
    prediction = model_under.predict(input_data)

    return jsonify({'prediction': prediction[0]})
    
# Run the app
if __name__ == '__main__':
    app.run(debug=True, port=5000)