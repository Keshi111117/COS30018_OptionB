import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os

def load_and_process_data(file_path, start_date=None, end_date=None, handle_nan='drop', split_method='ratio',
                          test_size=0.2, split_date=None, random_state=None, save_local=False,
                          local_path=None, scale_features=False, scaler_type='standard'):
    # Load the dataset
    df = pd.read_csv(file_path)
    # Rename the first unnamed column to 'Date'
    if df.columns[0] == 'Unnamed: 0':
        df.rename(columns={'Unnamed: 0': 'Date'}, inplace=True)

    # Convert 'Date' column to datetime format
    df['Date'] = pd.to_datetime(df['Date'])

    # Filter by date range
    if start_date:
        df = df[df['Date'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['Date'] <= pd.to_datetime(end_date)]

    # Handle NaN values
    if handle_nan == 'drop':
        df = df.dropna()
    elif handle_nan == 'fill_mean':
        df = df.fillna(df.mean())
    elif handle_nan == 'interpolate':
        df = df.infer_objects()  # Convert object columns to their inferred types
        numeric_columns = df.select_dtypes(include=[np.number]).columns
        df[numeric_columns] = df[numeric_columns].interpolate()

    # Split the data
    if split_method == 'ratio':
        train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state, shuffle=True)
    elif split_method == 'date':
        if split_date:
            train_df = df[df['Date'] < pd.to_datetime(split_date)]
            test_df = df[df['Date'] >= pd.to_datetime(split_date)]
        else:
            raise ValueError("split_date must be provided if split_method is 'date'")
    else:
        raise ValueError("split_method must be either 'ratio' or 'date'")

    # Feature scaling
    scalers = {}
    if scale_features:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError("scaler_type must be either 'standard' or 'minmax'")

        for column in df.columns:
            if column != 'Date' and df[column].dtype in ['int64', 'float64']:
                scaler = StandardScaler() if scaler_type == 'standard' else MinMaxScaler()
                scaler.fit(train_df[[column]])
                train_df[column] = scaler.transform(train_df[[column]])
                test_df[column] = scaler.transform(test_df[[column]])
                scalers[column] = scaler

    # Save locally if required
    if save_local and local_path:
        train_df.to_csv(os.path.join(local_path, 'train_data.csv'), index=False)
        test_df.to_csv(os.path.join(local_path, 'test_data.csv'), index=False)
        if scale_features:
            np.save(os.path.join(local_path, 'scalers.npy'), scalers)

    return train_df, test_df, scalers

# Usage example:
train_data, test_data, scalers = load_and_process_data(
    file_path=r'C:\Users\keshi\Downloads\Opt B\AMZN_2021-05-31.csv',
    start_date='2021-01-01',
    end_date='2021-05-31',
    handle_nan='interpolate',
    split_method='ratio',
    test_size=0.2,
    scale_features=True,
    scaler_type='minmax'
)

# Print the first few rows of the training data
print("Training Data:")
print(train_data.head())

# Print the first few rows of the testing data
print("\nTesting Data:")
print(test_data.head())

# Print the scalers used
if scalers:
    print("\nScalers Used:")
    for feature, scaler in scalers.items():
        print(f"{feature}: {scaler}")

