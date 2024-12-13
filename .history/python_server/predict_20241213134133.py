import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from preprocess import preprocess_df, save_preprocessed_data, load_dataset

# Constants
TARGET = 'STATUS'
DATA_PATH = 'python_server/Bicycle_Thefts_Open_Data.csv'  
MODEL_DIR = 'models/'
MODEL_NAME = 'RandomForestClassifier.pkl'

def print_results(classifier, y_test,y_pred, verbose=DEBUG):
    """Predict the target variable using the classifier."""
    if verbose:
        print("Predicting target variable...")
    
        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))
    print(f' Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f' Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}')


def predict_status(df):
    print("Predicting Status...")try:
        # Preprocess the data
        preprocessed_data = preprocess_df(df)
    except Exception as e:
        print(f"Error during preprocessing: {e}")
        return pd.Series()

    # Load the model
    model = load_model()

    # Make predictions
    try:
        y_pred = model.predict(preprocessed_data)
        if DEBUG:
            print(f"\nPredictions:\n{y_pred[:10]}")
        return y_pred
    except Exception as e:
        print(f"Error during prediction: {e}")
        return pd.Series()
    
def predict_dict(input_dict):
    try:
        df = pd.DataFrame([input_dict])  # Convert dict to DataFrame
        y_pred = predict_status(df)
        if y_pred.empty:
            return {"status": None}
        # Map prediction to human-readable labels
        status_mapping = {0: "Not Recovered", 1: "Recovered"}
        return {"status": status_mapping.get(y_pred[0], "Unknown")}
    except Exception as e:
        print(f"Error during prediction: {e}")
        return {"status": None}