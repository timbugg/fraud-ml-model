# Fraud data Exploratory Data Analysis

# Import modules
import pandas as pd

# Read in dataset and test for missing values / duplicates etc.
ff_data = pd.read_csv('financial_fraud_detection_dataset.csv')

ff_data.head()