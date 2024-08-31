#%%
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_boxplot(data, n=1, date_column='Date', price_column='Close', show_outliers=True):
    """
    Plot a boxplot for stock market data over a moving window of n days.

    Args:
    data (DataFrame): A pandas DataFrame with columns for Date and Close price.
    n (int): Size of the moving window (in days). Default is 1.
    date_column (str): Name of the column containing date information. Default is 'Date'.
    price_column (str): Name of the column containing price information. Default is 'Close'.
    show_outliers (bool): Whether to show outliers in the boxplot. Default is True.

    """
    # Make a copy of the data to avoid modifying the original
    df = data.copy()

    # Check if the required columns are in the DataFrame
    if date_column not in df.columns and df.index.name != date_column:
        date_cols = df.select_dtypes(include=['datetime64']).columns
        if len(date_cols) > 0:
            date_column = date_cols[0]
            print(f"Using '{date_column}' as the date column.")
        else:
            raise ValueError(f"Could not find a suitable date column. Please specify the correct date column name.")

    if price_column not in df.columns:
        raise ValueError(f"'{price_column}' column not found in the DataFrame. Please specify the correct price column name.")

    # If the date column is not the index, set it as the index
    if df.index.name != date_column and date_column in df.columns:
        df.set_index(date_column, inplace=True)

    # Ensure the index is datetime type
    df.index = pd.to_datetime(df.index)

    # Calculate rolling statistics
    rolling_data = df[price_column].rolling(window=n).agg(['min', 'max', 'mean', 'median', 'std'])

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot([rolling_data['min'].dropna(), rolling_data['max'].dropna(),
                     rolling_data['mean'].dropna(), rolling_data['median'].dropna()],
                    labels=['Min', 'Max', 'Mean', 'Median'],
                    patch_artist=True,  # Required for filling the boxes with colors
                    showfliers=show_outliers)

    # Customizing the plot
    ax.set_title(f'Boxplot of {price_column} prices over a {n}-day moving window', fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    # Add some statistics as text
    stats_text = f"Data range: {df.index.min().date()} to {df.index.max().date()}\n"
    stats_text += f"Total trading days: {len(df)}\n"
    stats_text += f"Overall price range: ${df[price_column].min():.2f} - ${df[price_column].max():.2f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, verticalalignment='top', fontsize=10, alpha=0.7)

    # Color the boxes
    colors = ['#FFB3BA', '#BAFFC9', '#BAE1FF', '#FFFFBA']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    plt.tight_layout()
    plt.show()

    # Print some additional statistics
    print(f"Average {price_column} price: ${df[price_column].mean():.2f}")
    print(f"Median {price_column} price: ${df[price_column].median():.2f}")
    print(f"Standard deviation of {price_column} price: ${df[price_column].std():.2f}")

# Example usage:
plot_boxplot(train_data, n=5, date_column='Date', price_column='close', show_outliers=True)