# Import required libraries.
import pandas as pd

# Open the cleaned data .csv files and convert content to DataFrame format.
ds = pd.read_csv("clean_dataset_new.csv")
# Remove the index column.
ds = ds.drop(["Unnamed: 0"], axis=1)

# Conver the target variable from 3 classes to 2, combining classes "Slight" and "Severe".
ds.accident_severity = ds.accident_severity.astype('int')
ds.accident_severity.replace({3: 0, 2: 0}, inplace=True)

# Save binary data to .csv format.
ds.to_csv("clean_dataset_binary.csv")
