from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from preprocess_bicycle_theft import preprocess_df, save_preprocessed_data, load_dataset

# Constants
PREPROCESSED_FILE = 'python_server/preprocessed_bike_data.csv'
MODEL_DIR = './models/'
TARGET = 'STATUS'


def prepare_training(preprocessed_data, params):
    """
    Prepare training and test datasets.
    Args:
        preprocessed_data (DataFrame): Preprocessed DataFrame.
        params (dict): Dictionary containing training parameters.
    Returns:
        X_train, X_test, y_train, y_test, params
    """
    # Ensure target column exists
    target = TARGET
    features = [col for col in preprocessed_data.columns if col != target]

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        preprocessed_data[features], preprocessed_data[target],
        test_size=params.get('test_size', 0.2),
        random_state=params.get('random_state', 42),
        stratify=preprocessed_data[target]
    )
    return X_train, X_test, y_train, y_test


def train_model(preprocessed_data, params, random_state=42):
    """Train a model with GridSearch and SMOTE."""
    X_train, X_test, y_train, y_test = prepare_training(preprocessed_data, params)

    # Apply SMOTE for class balancing
    smote = SMOTE(random_state=random_state)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Define classifier and hyperparameters
    classifier_name = params['classifier']
    if classifier_name == 'LogisticRegression':
        param_grid = {'C': [0.4, 0.6, 1.0], 'max_iter': [100, 200]}
        classifier = LogisticRegression(solver='liblinear', random_state=random_state)
    elif classifier_name == 'RandomForestClassifier':
        param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
        classifier = RandomForestClassifier(random_state=random_state)
    elif classifier_name == 'DecisionTreeClassifier':
        param_grid = {'max_depth': [5, 10], 'min_samples_split': [2, 6]}
        classifier = DecisionTreeClassifier(random_state=random_state)
    else:
        raise ValueError(f"Unsupported classifier: {classifier_name}")

    # Perform grid search
    grid_search = GridSearchCV(
        classifier, param_grid, cv=params['cv'],
        scoring=params['scoring'], n_jobs=-1, verbose=2
    )
    grid_search.fit(X_train_smote, y_train_smote)

    # Save best model
    os.makedirs(MODEL_DIR, exist_ok=True)
    pickle.dump(grid_search.best_estimator_, open(f"{MODEL_DIR}{classifier_name}.pkl", 'wb'))
    print(f"Best {classifier_name} model saved to {MODEL_DIR}")

    return grid_search.best_estimator_, grid_search.best_params_, accuracy_score(y_test, grid_search.best_estimator_.predict(X_test))


if __name__ == '__main__':
    # Preprocess dataset if not already done
    if not os.path.exists(PREPROCESSED_FILE):
        data = load_dataset()
        preprocessed_data = preprocess_df(data)
        save_preprocessed_data(preprocessed_data, PREPROCESSED_FILE)
    else:
        preprocessed_data = pd.read_csv(PREPROCESSED_FILE)

    # Train models
    params = {'cv': 3, 'test_size': 0.2, 'scoring': 'roc_auc'}
    for model in ['LogisticRegression', 'RandomForestClassifier', 'DecisionTreeClassifier']:
        print(f"Training model: {model}")
        params['classifier'] = model
        trained_model, best_params, test_accuracy = train_model(preprocessed_data, params)
        print(f"Completed {model} with test accuracy: {test_accuracy:.4f}\n")
Key Fixes:
Changed Argument for prepare_training:

Accepts a DataFrame (preprocessed_data) instead of trying to load it from a file.
Improved File Handling:

Automatically creates the preprocessed dataset if it doesn't exist.
Class Balancing (SMOTE):

Added in the train_model function.
Error-Free Integration:

All operations are streamlined to avoid path-related errors.
Next Steps:
Run preprocess.py:

bash
Copy code
python preprocess_bicycle_theft.py
Train the models:

bash
Copy code
python train_model.py
If you encounter any further issues, let me know!













ChatGPT can make mistakes. Check important info.
?





Run


1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37


PREPROCESSED_FILE = 'python_server/preprocessed_bike_data.csv'
MODEL_DIR = './models/'
TARGET = 'STATUS'


import os
import pickle

from preprocess import load_dataset  # Load preprocessed data
from preprocess import  MODEL_DIR, TARGET  # isort:skip

DEBUG = True  # Override global settings
SAVE_MODEL = True

from sklearn.model_selection import train_test_split

def prepare_training(preprocessed_file, params):
    
    if not os.path.exists(preprocessed_file):
        raise FileNotFoundError(f"Preprocessed file not found at {preprocessed_file}")
    
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