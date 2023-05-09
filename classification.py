# Import required libraries.
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import matplotlib.pyplot as plt
from yellowbrick.classifier import ConfusionMatrix, ROCAUC
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.base import clone
import pickle
from imblearn.over_sampling import SMOTE

# Set the dataset to be used in this investigation.
dataset_csv = "post_0.05_FS_dataset.csv"
# Set useSMOTE. To use SMOTE change this variable to true.
useSMOTE = False
# Set the SMOTE strategy. If using smote, set the strategy to either "minority" or "not majority".
smote_strategy = 'minority'
# Define Classifier algorithms used in this investigation.
Classifiers = [
    RandomForestClassifier(
        n_estimators=1000,
        criterion='entropy',
        max_depth=50,
        min_samples_split=2,
        min_samples_leaf=5,
        bootstrap=True,
        verbose=5,
        warm_start=True,
        n_jobs=-1
    ),
    GaussianNB(
        var_smoothing=0.1
    ),
    KNeighborsClassifier(
        algorithm='auto',
        leaf_size=1,
        metric='manhattan',
        n_jobs=-1,
        n_neighbors=3,
        weights='distance'
    )
]

# Function to Run a Classification Model which uses a reduced dataset with one feature removed, and produce results output.
# This allows the importance of removing this feature to be identified.


def RunFeatureImportanceClassification(Classifier, X, y, column):
    # Clone the classifier for this interation of the Classification run
    Classifier = clone(Classifier)
    # Perform a train test split to split the X and y variables into separate train and test datasets with a ratio of 80:20.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=2)
    # Normalize the data using the MinMaxScaler() using the default range of 0,1.
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # If useSMOTE is set to true, perform SMOTE usign the selected strategy.
    if useSMOTE:
        sm = SMOTE(random_state=2, sampling_strategy=smote_strategy)
        X_train, y_train = sm.fit_sample(X_train, y_train)

    # Fit the classifier
    Classifier.fit(X_train, y_train)

    # Create an array of the predicted target variable values from the classifier.
    clf = Classifier.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    # Generate evaluation metrics and store an array of these to the FeatureImportanceClassificationResults DataFrame.
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='macro')
    recall = recall_score(y_test, y_preds, average='macro')
    FeatureImportanceClassificationResults.insert(
        0, column, [accuracy, precision, recall])

# Function to run a Classification Model and produce results output for any algorithm and dataset.


def RunClassification(Classifier, X, y):
    # Clone the classifier for this interation of the Classification run
    Classifier = clone(Classifier)
    # Perform a train test split to split the X and y variables into separate train and test datasets with a ratio of 80:20.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=2)
    # Normalize the data using the MinMaxScaler() using the default range of 0,1.
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # If useSMOTE is set to true, perform SMOTE usign the selected strategy.
    if useSMOTE:
        sm = SMOTE(random_state=2, sampling_strategy=smote_strategy)
        X_train, y_train = sm.fit_sample(X_train, y_train)

    # Fit the classifier
    Classifier.fit(X_train, y_train)
    # Use Picle to save the model to .sav format.
    pickle.dump(Classifier, open(Classifier.__class__.__name__ +
                " General Classification", 'wb'))

    # Generate figure for results graphs.
    title = (Classifier.__class__.__name__ + " Classification")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(title)

    # Create an array of the plots required to visiualize the results.
    visualgrid = [
        ConfusionMatrix(Classifier, ax=ax1, title=" ", cmap="Blues"),
        ROCAUC(Classifier, ax=ax2, micro=False, Macro=True, title=" "),
    ]
    # Loop through the visualisation plots selected and plot to the figure.
    for viz in visualgrid:
        viz.fit(X_train, y_train)
        viz.score(X_test, y_test)
        viz.finalize()

    # Configure plot layout and produce.
    plt.tight_layout(w_pad=4.0)
    plt.show()

    # Create an array of the predicted target variable values from the classifier.
    clf = Classifier.fit(X_train, y_train)
    y_preds = clf.predict(X_test)

    # Generate evaluation metrics and store an array of these to the ClassificationResults DataFrame.
    accuracy = accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, average='macro')
    recall = recall_score(y_test, y_preds, average='macro')
    ClassificationResults.insert(0, Classifier.__class__.__name__, [
                                 accuracy, precision, recall])


# Import the post Feature Selection Dataset from .csv format.
ds = pd.read_csv(dataset_csv)
# Remove the index column.
ds = ds.drop(["Unnamed: 0"], axis=1)

# Drop Id column.
ds = ds.iloc[:, 1:]

# Separate the DataFrame into feature set (X) and classifier variable (y).
X = ds.drop("accident_severity", 1)
y = ds["accident_severity"]

# Define results DataFrame for Classification results.
ClassificationResults = pd.DataFrame(index=["accuracy", "precision", "recall"])
# Loop through the Classification Algorithm.
for Classifier in Classifiers:
    # Define a results DataFrame for Feature Importance Classification results for this Classification Algorithm.
    FeatureImportanceClassificationResults = pd.DataFrame(
        index=["accuracy", "precision", "recall"])
    # Loop through each column (variable) in the feature set (X).
    for column in X:
        # Create a copy of the feature set (X) and drop/remove the column on this iteration.
        X_copy = X.copy()
        X_copy.drop(column, axis=1, inplace=True)
        # Run the Feature Importance Classification for this Classification Algorithm and removed variable.
        RunFeatureImportanceClassification(Classifier, X_copy, y, column)
    # Export Feature Importance Classification Results for this Classification Algorithm to .csv format.
    FeatureImportanceClassificationResults.to_csv(
        Classifier.__class__.__name__ + " Feature Importance Classification.csv")
    # Run General Classification for this Classification Algorithm.
    RunClassification(Classifier, X, y)
# Export Classification Results to .csv format.
ClassificationResults.to_csv("Classification Results.csv")
