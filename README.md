# Car Accident Classification

## Overview

Machine Learning project using Data Cleaning techniques, Feature Selection and Classification to predict the accident severity.

## Instructions

1. Ensure the working directory contains raw dataset files named
"2020-accident.csv" and "2020-vehicle.csv". 

2. Run the "data_cleaning.py" file, producing the cleaned data in .csv format
("clean_data.csv").

3. Run the "feature_selection.py" file, producing the post feature selection
data in .csv format ("post_0.05_FS_dataset.csv"). If running multiple feature
selections with varying p-value's (as in this investigation), change the
p_value variable on line 6 of the "feature_selection.py" file to the desired
value (eg. 0.01). This change will automatically be reflected in the naming of
the .csv output file.

4. Run the "classification.py" file to produce .csv output files as well as
a number of results plots. By default this file
will use the "post_0.05_FS_dataset.csv" data file but this can be changed. A
number of configuration options can be changed as follows:

    - The dataset used can be changed by editing vairable "dataset_csv"
    containing the name of the desired .csv filename on line 18.
    - The use of SMOTE can be set to either "True" or "False" by editing the
    variable "useSMOTE" on line 20.
    - When using SMOTE, the strategy can be changed by editing the variable
    "smote_strategy" on line 22
    - The Classification algorithms can be changed and configurations edited by
    editing the array "Classifiers" defined on line 24.
