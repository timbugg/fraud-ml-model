# Fraud data Exploratory Data Analysis

# %% 
# Import modules / set key functions
import pandas as pd
import time
from contextlib import contextmanager

# set up timer generator to measure execution time of cells in file
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f'[{name}] done in {end - start:.1f} seconds.')

# Once data cleaned, save to parquet so have option to skip loading
# and cleaning in future
file_type = 'csv'

# %%
# Read in dataset / check data
with timer('read data'):
    # Only conduct csv import and cleaning if filetpye is csv
    if file_type == 'csv':
        ff_data = pd.read_csv('data/financial_fraud_detection_dataset.csv')

        print(ff_data.head())

        # check for missing values
        print(ff_data.isnull().sum())
        print(ff_data[~ff_data['time_since_last_transaction'].isna()])
        
        # This col is null when target is null so fill with meaningful value
        ff_data['fraud_type'] = ff_data['fraud_type'].fillna('missing')
        
        # this col is only null when no transaction has taken place - fillna
        # with int sentinel value
        ff_data['time_since_last_transaction'] = \
            ff_data['time_since_last_transaction'].fillna(999999)
        
        print(ff_data.dtypes) # lots of object types !!!

        # Adjust data types
        ff_data[
            ['transaction_id', 'sender_account', 'receiver_account',
             'ip_address', 'device_hash']
             ]\
            = ff_data[
                ['transaction_id', 'sender_account', 'receiver_account',
                 'ip_address', 'device_hash']]\
                .astype('string')

        ff_data[
            ['transaction_type', 'merchant_category', 'location', 'fraud_type',
             'device_used', 'payment_channel']
        ]\
            = ff_data[
                ['transaction_type', 'merchant_category', 'location',
                 'fraud_type', 'device_used', 'payment_channel']]\
                .astype('category')

        ff_data[
            ['amount', 'spending_deviation_score', 'geo_anomaly_score',
             'time_since_last_transaction']
             ]\
            = ff_data[['amount', 'spending_deviation_score',
                       'geo_anomaly_score', 'time_since_last_transaction']]\
                        .astype('float32')
        
        ff_data['velocity_score'] = ff_data['velocity_score'].astype('int32')

        ff_data['timestamp'] = pd.to_datetime(
            ff_data['timestamp'], format='ISO8601')
        
        # Note that is_fraud is already a boolean type so no need to change

        print('Post-processing dtypes: \n\n', ff_data.dtypes, '\n')

        # Check for duplicates
        print('Duplicates: ', ff_data.duplicated().sum(), '\n') # 0 duplicates - hurrah

        # Post-process null check
        print('Nulls: ', ff_data.isnull().sum().sum(), '\n')

        # Save to parquet
        ff_data.to_parquet('data/cleaned_data.parquet')
    
    if file_type == 'parquet':
        ff_data = pd.read_parquet('data/cleaned_data.parquet')
        print(ff_data.head())
# %%
# %%
