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