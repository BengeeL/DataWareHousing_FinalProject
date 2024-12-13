from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier

import os
import pickle

from preprocess_bicycle_theft import load_data  # Load preprocessed data
from settings import DEBUG, MODEL_DIR, TARGET  # isort:skip

DEBUG = True  # Override global settings
SAVE_MODEL = True


def prepare_training(df, params):
    """Prepare training and test datasets."""
    test_size = params.get('test_size', 0.2)
    target = TARGET
    cols = df.columns.to_list()
    cols.remove(target)

    random_state = params.get('random_state', 42)
    X_train, X_test, y_train, y_test = train_test_split(
        df[cols], df[target], test_size=test_size, random_state=random_state, stratify=df[target]
    )
    return X_train, X_test, y_train, y_test, params

