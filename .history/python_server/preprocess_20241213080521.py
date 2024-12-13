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

