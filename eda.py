# Fraud data Exploratory Data Analysis

# %% 
# Import modules / set key functions
import pandas as pd
import time
from contextlib import contextmanager
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis
import numpy as np


# set up timer generator to measure execution time of cells in file
@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print(f'[{name}] done in {end - start:.1f} seconds.')

def distribution_analysis(series: pd.Series = None, name: str = None) -> None:
    '''
    Function to conduct distribution analysis on a pandas series
    Args:
        series (pd.Series): Series to be analysed
        name (str): Name of the series
        Returns: None
    '''
    series = series.dropna()
    col_skew = skew(series)
    col_kurt = kurtosis(series, fisher=False)
    summary = []

    # Skew direction (if any)
    if abs(col_skew) < 0.5:
        summary.append('distribution is fairly symmetric')
    elif col_skew > 0:
        summary.append(f'distribution is right-skewed (long tail on the right, skew = {col_skew:.2f})')
    else:
        summary.append(f'distribution is left-skewed (long tail on the left, skew = {col_skew:.2f})')

    # Kurtosis direction (if any)
    if col_kurt > 4:
        summary.append(f'heavy-tailed distribution (kurtosis = {col_kurt:.2f}) — strong outliers likely')
    elif col_kurt > 3.5:
        summary.append(f'moderately heavy-tailed (kurtosis = {col_kurt:.2f})')
    elif 2.5 <= col_kurt <= 3.5:
        summary.append(f'approximately normal-tailed (kurtosis = {col_kurt:.2f})')
    elif col_kurt > 1.5:
        summary.append(f'light-tailed (kurtosis = {col_kurt:.2f}) — few extreme values')
    else:
        summary.append(f'very light or no significant tails (kurtosis = {col_kurt:.2f})')

    # Print summary
    print(f' - [{name.replace("_", " ").title()}]\n')
    for line in summary:
        print(f' - {line}')
    
    return None

def summarize_categorical_columns(
        df: pd.DataFrame = None, 
        max_values_display: int = 5, 
        cardinality_threshold = 50,
        plots: bool = True
        ) -> None:
    '''
    Function to summarize categorical columns in a DataFrame
    Args:
        df (pd.DataFrame): DataFrame to be analysed
        max_values_display (int): Maximum number of values to display
        cardinality_threshold (int): Threshold for high cardinality
        plots (bool): Whether to plot the value counts
    Returns: None
    '''
    cat_cols = df.select_dtypes(include=['object', 'category']).columns

    for col in cat_cols:
        # skip fraud_type and is_fraud as they are target / target-related
        if col in ['fraud_type', 'is_fraud']:
            continue
        # select series and get value counts
        series = df[col]
        value_counts = series.value_counts(dropna=False)
        unique_vals = series.nunique(dropna=True)

        # get top value and percentage
        top_value = value_counts.index[0]
        top_count = value_counts.iloc[0]
        top_pct = top_count / len(series) * 100

        # clean column name for display
        clean_col = col.replace('_', ' ').title()

        # cardinality init and min max values for charts
        high_cardinality = False

        print(f'\nColumn: {clean_col}')
        print(f' - Unique values (excluding NaNs): {unique_vals}')
        print(f' - Missing values: {series.isna().sum()}')

        if unique_vals == 1:
            print(' - Constant column (not informative)')
        elif unique_vals <= 5:
            print(f' - Low cardinality')
        elif unique_vals > cardinality_threshold:
            print(f' - High cardinality — may need grouping or special encoding')
            high_cardinality = True

        print(f' - Most common: \'{top_value}\' ({top_pct:.2f}% of total)')

        # Show top N values
        print(f' - Top {min(max_values_display, len(value_counts))} value counts:')
        for val, count in value_counts.head(max_values_display).items():
            pct = count / len(series) * 100
            print(f'     {repr(val):<15} : {count} ({pct:.2f}%)')
        
        # Number of bars to plot
        # if high cardinality, plot top N values
        n_bars = 6 if high_cardinality else unique_vals
        value_counts = value_counts.head(n_bars)
        top_cats = value_counts.index
        filter_df = df[df[col].isin(top_cats)]
        fraud_rates = df.groupby(col, observed=True)['is_fraud'].mean()
        min_val = fraud_rates.values.min()
        max_val = fraud_rates.values.max()

        if plots:
            # plot value counts for distribution analysis
            plt.figure(figsize=(10, 5))
            value_counts.plot(kind='bar')
            plt.title(f'{clean_col} value counts')
            plt.xlabel(clean_col)
            plt.ylabel('Count')
            plt.xticks(rotation=45)
            plt.show()
            plt.close()

            # plot against target variable
            ax = filter_df.groupby(col, observed=True)['is_fraud']\
                .mean()\
                .sort_values()\
                .plot(kind='bar')
            plt.title(f'{clean_col} vs. Target')
            plt.xlabel(clean_col)
            plt.ylabel('Proportion Fraud')
            plt.xticks(rotation=45)
            plt.ylim(min_val / 1.002, max_val * 1.002)
            for i, val in enumerate(ax.patches):
                    height = val.get_height()
                    offset = height * 0.001
                    ax.text(i, height + offset, f'{height * 100:.2f}%', \
                            ha='center', fontsize=8)
                
            plt.show()
            plt.close()
        
    return None

# Once data cleaned, save to parquet so have option to skip loading
# and cleaning in future
file_type = 'parquet' # 'csv' or 'parquet'
# %% ==========================================================================
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
        print('Duplicates: ', ff_data.duplicated().sum(), '\n') # 0 dupes-hurrah

        # Post-process null check
        print('Nulls: ', ff_data.isnull().sum().sum(), '\n')

        # Save to parquet
        ff_data.to_parquet('data/cleaned_data.parquet')
    
    if file_type == 'parquet':
        ff_data = pd.read_parquet('data/cleaned_data.parquet')
        print(ff_data.head())
# %% ==========================================================================
# High Level overview of data
with timer('high level overview'):

    # print summary of data
    pd.set_option('display.float_format', '{:,.2f}'.format)
    print(ff_data.describe())
    print(ff_data.info())
    print(ff_data.shape)

    # review numerical variables
    # we get a rough gauge of numerical variables from describe, but can set
    # the num_vars now for further analysis
    num_vals = ff_data.select_dtypes(include=['float32', 'int32'])

    # review categorical variables
    cat_vals = ff_data.select_dtypes(include=['category'])
    for cat in cat_vals.columns:
        print(f'{cat}:\n{cat_vals[cat].cat.categories}\n')

# %% ==========================================================================
# Descriptive statistics
with timer('descriptive statistics'):
    # review numerical variables
    for col in num_vals.columns:
        if col == 'time_since_last_transaction':
            # sentinel value is misleading - remove from analysis for now
            filter_df = ff_data[ff_data[col] != 999999]
        else:
            filter_df = ff_data
        # some charts look odd with too many bins - set a max
        if col == 'geo_anomaly_score':
            bins = 101
        else:
            bins = num_vals[col].nunique() \
                if num_vals[col].nunique() < 50 else 50

        # print summary
        distribution_analysis(series=filter_df[col], name=col)

        plt.figure(figsize=(10, 5))
        plt.hist(filter_df[col], bins=bins, density=True)
        plt.title(f'{col.replace("_", " ").title()} distribution')
        plt.xlabel(col.replace('_', ' ').title())
        plt.ylabel('Density')
        plt.show()
        plt.close()
    
    # review categorical variables
    # Data distribution - incredibly consistent category proportions
    summarize_categorical_columns(ff_data) # take default values for other args
# %%
