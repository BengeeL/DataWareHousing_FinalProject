import pandas as pd
import pickle

def predict_status(JSONdata): 
    # Get the input data (JSON)
    print("Predicting...")
    print(JSONdata)

    # Load model
    with open('../models/RandomForestClassifier.pkl', 'rb') as f:
        model = pickle.load(f)

    # Preprocess the input data
    # Convert JSON to DataFrame
    df = pd.DataFrame([JSONdata])
    print(df)

    # Encode categorical features
    with open('../python_server/models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)

    categorical_columns = df.select_dtypes(include=['object']).columns

    # remove OCC_YEAR and BIKE_COST columns from the list of categorical columns
    categorical_columns = categorical_columns.drop(['OCC_YEAR', 'BIKE_COST'], errors='ignore')

    # Use transform instead of fit_transform
    df[categorical_columns] = encoder.transform(df[categorical_columns].astype(str))

    print(df.head())

    # Make a prediction
    prediction = model.predict(df)
    print("Prediction: " + str(prediction))

    return prediction[0]