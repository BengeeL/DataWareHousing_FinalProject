import numpy as np
import pandas as pd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os


# Get the current working directory
directory = os.getcwd()

# Modify the directory to get the file Bicycle_Thefts_Open_Data.csv (same directory as the notebook)
file_path = os.path.join(directory, 'python_server/Bicycle_Thefts_Open_Data.csv')

# Load the dataset
bike_data = pd.read_csv(file_path)


## 1 -  Data exploration (Benjamin Lefebvre - 301234587)

# a) Load and describe data elements (columns), provide descriptions & types, ranges and values of elements as appropriate - use pandas, numpy and any other python packages.
# b) Statistical assessments including means, averages, and correlations.
# c) Missing data evaluations
# d) Graphs and visualizations

print("-----------------------------------")
print("-------- Data exploration ---------")
print("-----------------------------------")
print("")

# Quick summary of the dataset
print("")
print("--------------------------")
print("-------- Summary ---------")
print("--------------------------")
print("")

print(bike_data.info())

# Statistical summary of the dataset
print("")
print("------------------------------------------")
print("-------- Statistical Assessments ---------")
print("------------------------------------------")
print("")

print(bike_data.describe())

# Display the first few rows of the dataset
print("")
print("---------------------------------")
print("-------- First Few Rows ---------")
print("---------------------------------")
print("")
print(bike_data.head())

# Display the data range of each column
print("")
print("-------------------------------------")
print("-------- Columns Data Range ---------")
print("-------------------------------------")
print("")
for column in bike_data.columns:
    if bike_data[column].dtype in [np.int64, np.float64]:
        min_value = bike_data[column].min()
        max_value = bike_data[column].max()
        print(f"{column}: Min: {min_value}, Max: {max_value}")
    else:
        unique_values = bike_data[column].unique()
        print(f"Unique values in column '{column}':")
        print(unique_values)

    print("")

# Display the data types of each column
print("")
print("------------------------------------")
print("-------- Columns Data Type ---------")
print("------------------------------------")
print("")
print(bike_data.dtypes)

# Check for missing values
print("")
print("-------------------------------")
print("-------- Missing Data ---------")
print("-------------------------------")
print("")
print(bike_data.isnull().sum())

# Select only numeric columns for correlation matrix
numeric_data = bike_data.select_dtypes(include=[np.number])

# Correlation matrix
print("")
print("-------------------------------------")
print("-------- Correlation Matrix ---------")
print("-------------------------------------")
print("")
print(numeric_data.corr())

# Visualizations
import matplotlib.pyplot as plt
import seaborn as sns

# Histogram of bike costs
print("")
print("------------------------------------------")
print("-------- Histogram of bike costs ---------")
print("------------------------------------------")
print("")
plt.figure(figsize=(10, 6))
sns.histplot(bike_data['BIKE_COST'], bins=30, kde=True)
plt.title('Distribution of Bike Costs')
plt.xlabel('Bike Cost')
plt.ylabel('Frequency')
plt.show()

# Correlation heatmap
print("")
print("--------------------------------------")
print("-------- Correlation heatmap ---------")
print("--------------------------------------")
print("")
plt.figure(figsize=(12, 8))
sns.heatmap(numeric_data.corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()

# Histogram of mean of BIKE_COST based on PRIMARY_OFFENCE
# Histogram STATUS based on OCC_YEAR
print("")
print("--------------------------------------------------------")
print("-------- Histogram of STATUS based on OCC_YEAR ---------")
print("--------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='OCC_YEAR', data=bike_data)
plt.title('Status of Bicycle Thefts by Year')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on OCC_MONTH
print("")
print("---------------------------------------------------------")
print("-------- Histogram of STATUS based on OCC_MONTH ---------")
print("---------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='OCC_MONTH', data=bike_data)
plt.title('Status of Bicycle Thefts by Month')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()


# Histogram STATUS based on OCC_DOW
print("")
print("--------------------------------------------------------")
print("-------- Histogram of STATUS based on OCC_DOW ----------")
print("--------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='OCC_DOW', data=bike_data)
plt.title('Status of Bicycle Thefts by Day of Week')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on DIVISION
print("")
print("--------------------------------------------------------")
print("-------- Histogram of STATUS based on DIVISION ---------")
print("--------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='DIVISION', data=bike_data)
plt.title('Status of Bicycle Thefts by Division')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on LOCATION_TYPE
print("")
print("-------------------------------------------------------------")
print("-------- Histogram of STATUS based on LOCATION_TYPE ---------")
print("-------------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='LOCATION_TYPE', data=bike_data)
plt.title('Status of Bicycle Thefts by Location Type')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on PREMISES_TYPE
print("")
print("------------------------------------------------------------")
print("-------- Histogram of STATUS based on PREMISES_TYPE ---------")
print("------------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.countplot(x='STATUS', hue='PREMISES_TYPE', data=bike_data)
plt.title('Status of Bicycle Thefts by Premises Type')
plt.xlabel('Status')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on BIKE_COST
print("")
print("---------------------------------------------------------")
print("-------- Histogram of STATUS based on BIKE_COST ---------")
print("---------------------------------------------------------")
print("")

plt.figure(figsize=(12, 8))
sns.histplot(data=bike_data, x='BIKE_COST', hue='STATUS', bins=30, kde=True)
plt.title('Status of Bicycle Thefts by Bike Cost')
plt.xlabel('Bike Cost')
plt.ylabel('Count')
plt.show()

# Histogram STATUS based on NEIGHBOURHOOD_158
print("")
print("-------------------------------------------------")
print("-------- Histogram of mean of BIKE_COST ---------")
print("-------------------------------------------------")
print("")

# Calculate the mean BIKE_COST for each PRIMARY_OFFENCE category
mean_bike_cost_by_offence = bike_data.groupby('PRIMARY_OFFENCE')['BIKE_COST'].mean()

# Remove categories with NaN values
mean_bike_cost_by_offence = mean_bike_cost_by_offence.dropna()

plt.figure(figsize=(12, 20))
sns.barplot(x=mean_bike_cost_by_offence.values, y=mean_bike_cost_by_offence.index)
plt.title('Average Bike Cost by PRIMARY_OFFENCE Category')
plt.xlabel('Average Bike Cost')
plt.ylabel('PRIMARY_OFFENCE')
plt.show()