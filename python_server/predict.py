import pickle
import pandas as pd
from preprocess import encode_categorical_features, scale_numerical_features 

def predict_status(JSONdata): 
    # Get the input data (JSON)
    print("Predicting...")
    print(JSONdata)

    # load model
    with open('../models/RandomForestClassifier.pkl', 'rb') as f:
        model = pickle.load(f)

        # Preprocess the input data
        # Convert JSON to DataFrame
        df = pd.DataFrame([JSONdata])
        print(df)

        # Encode categorical features
        df = encode_categorical_features(df)

        # Scale numerical features
        df = scale_numerical_features(df)

        # Make a prediction
        prediction = model.predict(df)
        print("Prediction: " + str(prediction))

        return prediction[0]
    
    # # Make a prediction
    # prediction = model.predict([features])
    # print(prediction)
    # # Return the prediction
    # response = jsonify({'status': prediction[0]})
    # response.headers.add("Access-Control-Allow-Origin", "*")
    # return response
