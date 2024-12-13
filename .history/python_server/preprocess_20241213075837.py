import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

# Constants
TARGET = 'STATUS'
DATA_PATH = 'Bicycle_Thefts_Open_Data.csv'  
# Display all columns
pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)


def load_dataset(fi)
file_path = os.path.join(directory, 'python_server/Bicycle_Thefts_Open_Data.csv')

# Load the dataset
bike_data = pd.read_csv(file_path)