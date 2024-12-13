import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Constants
TARGET = 'STATUS'
DATA_PATH = 'python_server/Bicycle_Thefts_Open_Data.csv'  
# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def load_dataset(file_path=DATA_PATH):
	bike_data = pd.read_csv(file_path)
	print(f'Loaded {bike_data.shape[0]} records and {bike_data.shape[1]} columns.')
	return bike_data

def remove_unknown(df):
    """Remove rows where STATUS is 'UNKNOWN'."""
    initial_count = df.shape[0]
    df = df[df['STATUS'].fillna('').str.upper() != 'UNKNOWN']
    print(f"Removed 'UNKNOWN' STATUS rows. Rows reduced from {initial_count} to {df.shape[0]}.")
    return df
ef fill_missing_values(df):
    """Fill missing values in the dataset."""
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    print("Filled missing values in numerical columns with mean.")
    return df