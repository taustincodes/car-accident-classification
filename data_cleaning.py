# Import required libraries.
import pandas as pd
import jenkspy

# Open the raw data .csv files and convert content to DataFrame format.
accident = pd.read_csv("2020-accident.csv", dtype={"accident_index": str})
vehicle = pd.read_csv("2020-vehicle.csv", dtype={"accident_index": str})

# Perform a listwise deletion to remove all rows with missing values from the datasets.
accident.dropna()
vehicle.dropna()

# Extract only the require columns from the datasets.
accident = accident[["accident_index",
                    "accident_severity",
                     "police_force",
                     "day_of_week",
                     "time",
                     "local_authority_district",
                     "light_conditions",
                     "weather_conditions"
                     ]]

vehicle = vehicle[["accident_index",
                   "vehicle_type",
                   "towing_and_articulation",
                   "vehicle_left_hand_drive",
                   "journey_purpose_of_driver",
                   "sex_of_driver",
                   "age_band_of_driver",
                   "engine_capacity_cc",
                   "propulsion_code",
                   "age_of_vehicle",
                   "generic_make_model",
                   "driver_imd_decile",
                   "driver_home_area_type"]]

# Merge the two datasets using the foreign key accident index.
dataset = accident.merge(vehicle, on="accident_index", how="inner")

# Remove rows which contain values of -1. The dataset specifies that "-1" represent missing data.
dataset = dataset[dataset.engine_capacity_cc != -1]
dataset = dataset[dataset.local_authority_district != -1]
dataset = dataset[dataset.vehicle_type != -1]
dataset = dataset[dataset.towing_and_articulation != -1]
dataset = dataset[dataset.vehicle_left_hand_drive != -1]
dataset = dataset[dataset.journey_purpose_of_driver != -1]
dataset = dataset[dataset.sex_of_driver != -1]
dataset = dataset[dataset.age_band_of_driver != -1]
dataset = dataset[dataset.engine_capacity_cc != -1]
dataset = dataset[dataset.propulsion_code != -1]
dataset = dataset[dataset.age_of_vehicle != -1]
dataset = dataset[dataset.driver_imd_decile != -1]
dataset = dataset[dataset.driver_home_area_type != -1]
dataset = dataset[dataset.generic_make_model != -1]
# NEED TO LOOP THROUGH COLUMNS AND REMOVE BOTH STRING AND INT VALUES

# Performing binning on the variable "engine_capacity_cc". The breaks for the bins are created using the jens_breaks function.
breaks = jenkspy.jenks_breaks(dataset["engine_capacity_cc"], n_classes=5)
labels = [1, 2, 3, 4, 5]
dataset["engine_capacity_cc_binned"] = pd.cut(
    dataset['engine_capacity_cc'], bins=breaks, labels=labels, include_lowest=True)

# Create a new variable to represent the hour of the accident, extracting the hour from the "time" column.
dataset["accident_hour"] = pd.to_datetime(
    dataset["time"], format="%H:%M").dt.hour

# Create a new variable to represent the make of the vehicle, extracting the make from the "generic_make_model" column.
dataset['vehicle_make'] = dataset['generic_make_model'].str.split(' ').str[0]

# Remove the variables which were used to generate new variables and are no longer needed.
dataset = dataset.drop(["accident_index", "time", "generic_make_model",
                       "vehicle_make", "engine_capacity_cc"], axis=1)

# Drop rows where the age is greater than 35.
dataset = dataset[dataset.age_band_of_driver >= 7]

# Export the cleaned dataset to .csv format.
dataset.to_csv("clean_dataset.csv")
