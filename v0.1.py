import plotly.graph_objects as go
import pandas as pd
import yfinance as yf

#------------------------------------------------------------------------------
# Load Data
COMPANY = 'CBA.AX'
TRAIN_START = '2020-01-01'  # Start date for training data
TRAIN_END = '2023-08-01'    # End date for training data

# Fetch historical data from Yahoo Finance
train_data = yf.download(COMPANY, TRAIN_START, TRAIN_END)

# Ensure columns are in lowercase for consistency with your function
train_data.columns = train_data.columns.str.lower()

#------------------------------------------------------------------------------
# Plot Candlestick Chart

def plot_candlestick(df, n=1):
    """
    This function plots a candlestick chart for the stock data.

    Parameters:
    - df: DataFrame containing stock market data with 'Open', 'High', 'Low', 'Close' columns.
    - n: Number of trading days each candlestick should represent.
    """
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
        xaxis_rangeslider_visible=False
    )

    fig.show()

# Example usage:
plot_candlestick(train_data, n=1)
