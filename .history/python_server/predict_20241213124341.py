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

def predict_df(classifi)