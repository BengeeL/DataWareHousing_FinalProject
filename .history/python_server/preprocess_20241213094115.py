import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

# Constants
TARGET = 'STATUS'
DATA_PATH = 'python_server/Bicycle_Thefts_Open_Data.csv'  
MODEL_DIR = 'python_server/models/'
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

def fill_missing_values(df):
    """Fill missing values in the dataset."""
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
    df[numerical_columns] = df[numerical_columns].fillna(df[numerical_columns].mean())
    print("Filled missing values in numerical columns with mean.")
    return df


def encode_categorical_features(df):
    """Encode categorical features using OrdinalEncoder."""
    categorical_columns = df.select_dtypes(include=['object']).columns
    ord_enc = OrdinalEncoder()

    for col in categorical_columns:
        df[col] = ord_enc.fit_transform(df[[col]].astype(str))

    # Save the encoder for future use
    with open(f'{MODEL_DIR}encoder.pkl', 'wb') as f:
        pickle.dump(ord_enc, f)
    print("Categorical features encoded and encoder saved.")
    return df
def scale_numerical_features(df):
    """Apply standard scaling to numerical features."""
    numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns.drop('STATUS', errors='ignore')
    scaler = StandardScaler()
    scaled_numerical_data = scaler.fit_transform(df[numerical_columns])
    scaled_numerical_df = pd.DataFrame(scaled_numerical_data, columns=numerical_columns, index=df.index)
    df[numerical_columns] = scaled_numerical_df

    # Save the scaler for future use
    with open(f'{MODEL_DIR}scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("Numerical features scaled and scaler saved.")
    return df


def preprocess_df(df):
    """Preprocess the dataset: handle missing values, encode categorical features, and scale numerical features."""
    print("Preprocessing data...")

    # Keep only relevant columns
    relevant_columns = [
        'OCC_YEAR', 'OCC_MONTH', 'OCC_DOW', 'DIVISION', 'LOCATION_TYPE',
        'PREMISES_TYPE', 'BIKE_TYPE', 'BIKE_COST', 'STATUS',
        'NEIGHBOURHOOD_158'
    ]
    df = df[relevant_columns]
    print(f"Filtered to relevant columns: {relevant_columns}")

    # Remove rows with 'UNKNOWN' in STATUS
    df = remove_unknown(df)

    # Fill missing values
    df = fill_missing_values(df)

    # Map STATUS values to numeric labels
    status_mapping = {'STOLEN': 0, 'RECOVERED': 1}
    df['STATUS'] = df['STATUS'].map(status_mapping)
    print("Mapped STATUS column to numeric labels.")

    # Encode categorical features
    df = encode_categorical_features(df)

    # Scale numerical features
    df = scale_numerical_features(df)

    print("Preprocessing complete.")
    return df


def save_preprocessed_data(df, file_path='preprocessed_bike_data.csv'):
    """Save the preprocessed dataset to a CSV file."""
    df.to_csv(file_path, index=False)
    print(f"Preprocessed data saved to {file_path}.")


if __name__ == '__main__':
    # Load the dataset
   df = preprocess_df()


    # Preprocess the data
    # preprocessed_data = preprocess_df(data)

    # Save the preprocessed data
    save_preprocessed_data(preprocessed_data)
