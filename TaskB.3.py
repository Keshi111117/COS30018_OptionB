import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# Step 1: Download historical stock data
def download_stock_data(ticker, start_date, end_date):
    """
    Downloads historical stock data using yfinance.

    Args:
    - ticker: Stock ticker symbol (e.g., 'AAPL' for Apple).
    - start_date: Start date for the stock data (YYYY-MM-DD).
    - end_date: End date for the stock data (YYYY-MM-DD).

    Returns:
    - DataFrame containing the stock data.
    """
    print(f"Downloading data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    print("Data download complete.")
    return stock_data


# Step 2: Preprocess stock data
def preprocess_data(stock_data):
    """
    Prepares the stock data for analysis by selecting relevant columns.

    Args:
    - stock_data: DataFrame containing stock market data.

    Returns:
    - DataFrame with columns: 'Open', 'High', 'Low', 'Close', 'Volume'.
    """
    stock_data = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']]
    stock_data.columns = stock_data.columns.str.lower()  # Convert column names to lowercase
    print("Data preprocessing complete.")
    return stock_data


# Step 3: Plot Candlestick Chart
def plot_candlestick(df, n=1):
    """
    This function plots a candlestick chart for the stock data.

    Parameters:
    - df: DataFrame containing stock market data with 'Date', 'Open', 'High', 'Low', 'Close' columns.
    - n: Number of trading days each candlestick should represent.
    """
    # Check if 'Date' column exists, if not, use the index (assuming it's a DatetimeIndex)
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df.set_index('Date', inplace=True)

    # Resample the data to combine n trading days into one candlestick
    df_resampled = df.resample(f'{n}D').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # Create the candlestick chart
    fig = go.Figure(data=[go.Candlestick(x=df_resampled.index,
                                         open=df_resampled['open'],
                                         high=df_resampled['high'],
                                         low=df_resampled['low'],
                                         close=df_resampled['close'])])

    fig.update_layout(
        title=f'Candlestick chart (each candlestick represents {n} trading days)',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False  # Disable range slider below the x-axis
    )

    fig.show()


# Step 4: Plot Boxplot
def plot_boxplot(data, n=1, date_column='date', price_column='close', show_outliers=True):
    """
    Plots a boxplot for stock market data over a moving window of n days.

    Args:
    - data (DataFrame): A pandas DataFrame with columns for Date and Close price.
    - n (int): Size of the moving window (in days). Default is 1.
    - date_column (str): Name of the column containing date information. Default is 'Date'.
    - price_column (str): Name of the column containing price information. Default is 'Close'.
    - show_outliers (bool): Whether to show outliers in the boxplot. Default is True.
    """
    print("Preparing to display boxplot...")
    df = data.copy()

    if df.index.name != date_column and date_column in df.columns:
        df.set_index(date_column, inplace=True)

    df.index = pd.to_datetime(df.index)

    rolling_data = df[price_column].rolling(window=n).agg(['min', 'max', 'mean', 'median', 'std'])

    fig, ax = plt.subplots(figsize=(12, 6))

    bp = ax.boxplot([rolling_data['min'].dropna(), rolling_data['max'].dropna(),
                     rolling_data['mean'].dropna(), rolling_data['median'].dropna()],
                    labels=['Min', 'Max', 'Mean', 'Median'],
                    patch_artist=True,
                    showfliers=show_outliers)

    ax.set_title(f'Boxplot of {price_column} prices over a {n}-day moving window', fontsize=15)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.7)

    plt.tight_layout()

    # Ensure that the plot is displayed
    print("Displaying boxplot...")
    plt.show(block=True)


# Main function to run the visualizations
def main():
    # Define the stock ticker and date range for analysis
    ticker = 'AAPL'  # You can change this to any other stock symbol
    start_date = '2021-01-01'
    end_date = '2021-12-31'

    # Step 1: Download stock data
    stock_data = download_stock_data(ticker, start_date, end_date)

    # Step 2: Preprocess the data
    stock_data = preprocess_data(stock_data)

    # Step 3: Visualize Candlestick Chart
    print("Displaying candlestick chart...")
    plot_candlestick(stock_data, n=5)  # Each candlestick represents 5 days

    # Step 4: Visualize Boxplot
    print("Calling boxplot function...")
    plot_boxplot(stock_data, n=5, date_column='date', price_column='close', show_outliers=True)


if __name__ == "__main__":
    main()
