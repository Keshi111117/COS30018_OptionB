import yfinance as yf
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import numpy as np

def download_stock_data(ticker, start_date, end_date):
    data = yf.download(ticker, start=start_date, end=end_date)
    return data

# Example usage:
ticker = 'AMZN'
start_date = '2020-01-01'
end_date = '2023-01-01'
stock_data = download_stock_data(ticker, start_date, end_date)
print(stock_data.head())

def preprocess_data(data, feature='Close', sequence_length=60):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data[feature].values.reshape(-1, 1))

    X_train, y_train = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_train.append(scaled_data[i-sequence_length:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

    return X_train, y_train, scaler

# Example usage:
X_train, y_train, scaler = preprocess_data(stock_data)

def build_lstm_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicit Input layer to avoid warning
    model.add(LSTM(50, activation='tanh', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, activation='tanh'))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Example usage:
input_shape = (X_train.shape[1], 1)
model = build_lstm_model(input_shape)
model.fit(X_train, y_train, epochs=20, batch_size=32)

def predict_future_prices(model, data, scaler, days=5):
    predictions = []
    current_input = data[-1]

    for _ in range(days):
        next_prediction = model.predict(current_input.reshape(1, current_input.shape[0], 1))[0, 0]
        predictions.append(next_prediction)
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_prediction

    predictions = scaler.inverse_transform(np.array(predictions).reshape(-1, 1))
    return predictions.flatten()

# Example usage:
future_predictions = predict_future_prices(model, X_train, scaler, days=5)
print("Predicted prices for the next 5 days:", future_predictions)












