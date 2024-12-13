from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd


PREPROCESSED_FILE = 'python_server/preprocessed_bike_data.csv'
MODEL_DIR = './models/'
TARGET = 'STATUS'

# Check if the data is clean
print(df.info())
print(df.head())

import os
import pickle

from preprocess import load_dataset  # Load preprocessed data
from preprocess import  MODEL_DIR, TARGET  # isort:skip

DEBUG = True  # Override global settings
SAVE_MODEL = True

from sklearn.model_selection import train_test_split

def prepare_training(preprocessed_file, params):
    """
    Prepare training and test datasets.

    Args:
        preprocessed_file (str): Path to the preprocessed CSV file.
        params (dict): Dictionary containing training parameters like test size and random state.

    Returns:
        X_train, X_test, y_train, y_test, params
    """
    # Load preprocessed data
    df = pd.read_csv(preprocessed_file)

    # Define target and features
    target = 'STATUS'
    feature_columns = [col for col in df.columns if col != target]

    # Parameters
    test_size = params.get('test_size', 0.2)
    random_state = params.get('random_state', 42)

    # Train-test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_columns], df[target],
        test_size=test_size,
        random_state=random_state,
        stratify=df[target]
    )

    return X_train, X_test, y_train, y_test, params


def train_model(df, params, random_state=42):
    """Train a model with hyperparameter optimization and handle class imbalance."""
    X_train, X_test, y_train, y_test, _ = prepare_training(df, params)
    print("Checking data types before SMOTE:")
    print(X_train.dtypes)
    assert X_train.select_dtypes(include=['object']).empty, "Non-numeric columns found in X_train!"
    non_numeric_columns = X_train.select_dtypes(include=['object']).columns
    print("Non-numeric columns:", non_numeric_columns)
    X_train = X_train.drop(non_numeric_columns, axis=1)

    # Apply SMOTE to handle class imbalance
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    classifier_name = params['classifier']
    print(f"\n>>>>> Starting GridSearchCV {classifier_name}...")

    # Define the parameter grid to tune the hyperparameters
    if classifier_name == 'LogisticRegression':
        param_grid = {
            'C': [0.4, 0.6, 1.0],
            'max_iter': [100, 200],
        }
        classifier = LogisticRegression(solver='liblinear', random_state=random_state)
    elif classifier_name == 'RandomForestClassifier':
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [5, 10, None],
        }
        classifier = RandomForestClassifier(random_state=random_state)
    elif classifier_name == 'DecisionTreeClassifier':
        param_grid = {
            'max_depth': [5, 10, 20],
            'min_samples_leaf': [1, 3, 5],
            'min_samples_split': [2, 6],
        }
        classifier = DecisionTreeClassifier(random_state=random_state)
    else:
        print('!! Unexpected classifier:', classifier_name)
        return

    grid_search = GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        cv=params['cv'],
        scoring=params['estimator'],
        n_jobs=-1,
        verbose=1,
    )
    grid_search.fit(X_train_smote, y_train_smote)
    best_classifier = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")

    # Calculate test prediction metrics
    y_pred = best_classifier.predict(X_test)
    key_metric1 = accuracy_score(y_test, y_pred)
    key_metric2 = balanced_accuracy_score(y_test, y_pred)

    print(f"Accuracy: {key_metric1:.4f}")
    print(f"Balanced Accuracy: {key_metric2:.4f}")

    if SAVE_MODEL:
        os.makedirs(MODEL_DIR, exist_ok=True)
        filename = f'{MODEL_DIR}{classifier_name}.pkl'
        pickle.dump(best_classifier, open(filename, 'wb'))

    return best_classifier, best_params, key_metric1, key_metric2
if __name__ == '__main__':
    from time import time
     # Load the raw dataset
    df = load_dataset()

    # Preprocess the data


    estimator = 'roc_auc'  # Optimize for ROC AUC

    for classifier in [
        'LogisticRegression',
        'DecisionTreeClassifier',
        'RandomForestClassifier',
        
    ]:
        t_start = time()
        params = {
            'classifier': classifier,
            'estimator': estimator,
            'cv': 3,
            'test_size': 0.2,
        }
        best_classifier, best_params, key_metric1, key_metric2 = train_model(df, params)
        print(f"Finished {classifier} in {(time() - t_start):.2f} seconds\n")