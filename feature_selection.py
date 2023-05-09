# Import required libraries.
import pandas as pd
import statsmodels.api as sm

# Set the p value to the desired number for the feature selection.
p_value = 0.05

# Open the cleaned data .csv files and convert content to DataFrame format.
dataset = pd.read_csv("clean_dataset.csv")

# Remove the index column.
dataset = dataset.drop(["Unnamed: 0"], axis=1)

# Create two dataset, one containing the classsifier variable (y), and one containing the rest (X).
X = dataset.drop("accident_severity", 1)
y = dataset["accident_severity"]

# Add a column with all values of 1, this is required for the Orindayr Least Squares (OLS) model.
X_1 = sm.add_constant(X)

# Fit the OLS model
model = sm.OLS(y, X_1).fit()

# Create a DataFrame with the results of the feature selection and expor to .csv format to reveiw and present later.
fs_results = pd.DataFrame({"Feature": model.pvalues.index,
                          "p-value": model.pvalues.values, "R-squared": model.rsquared})
fs_results.to_csv("p_values.csv")

# Perform the backwards elimination. Loop through each column and remove the feature if the p value is less than 0.05.
cols = list(X.columns)
pmax = 1
while (len(cols) > 0):
    p = []
    x = X[cols]
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()
    p = pd.Series(model.pvalues.values[1:], index=cols)
    pmax = max(p)
    feature_with_p_max = p.idxmax()
    if (pmax > p_value):
        cols.remove(feature_with_p_max)
    else:
        break

# Store the columns which were selected by the feature selection.
selected_features_BE = cols

# Add the target variable to the "selected features".
selected_features_BE.append("accident_severity")

# Create a refined dataset with the newly selected features and store to .csv format.
refined_dataset = dataset[selected_features_BE]
refined_dataset.to_csv("post_" + str(p_value) + "_FS_dataset.csv")
