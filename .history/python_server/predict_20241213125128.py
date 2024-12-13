import pickle
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
)
from preprocess import preprocess_df
from settings import MODEL_DIR, MODEL_NAME, TARGET, DEBUG

def print_results(classifier, y_test,y_pred, verbose=DEBUG):
    """Predict the target variable using the classifier."""
    if verbose:
        print("Predicting target variable...")
    
        print(classification_report(y_test, y_pred, digits=3))
        print(confusion_matrix(y_test, y_pred))
    print(f' Accuracy: {accuracy_score(y_test, y_pred):.3f}')
    print(f' Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.3f}')


def predict_status(model_dir=MODEL_DIR, df,M):
    print("Predicting Status...")
    try:
        model_path =f{model_dir}