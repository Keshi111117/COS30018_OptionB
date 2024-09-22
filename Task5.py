import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from tensorflow.keras.optimizers import Adam


# Step 1: Data Download and Preprocessing
def download_and_prepare_multivariate_data(ticker, start_date, end_date):
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    features = stock_data[['Open', 'High', 'Low', 'Close', 'Volume']].values

    # Scale data to range [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    sequence_length = 60  # Use 60 days of data to predict the next day's price
    future_days = 10  # Predict the next 10 days

    for i in range(sequence_length, len(scaled_data) - future_days):
        X.append(scaled_data[i - sequence_length:i])
        y.append(scaled_data[i:i + future_days, 3])  # Predict future Close prices

    X, y = np.array(X), np.array(y)
    return X, y, scaler


# Step 2: Model Architectures
def build_multivariate_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(10))  # Predict 10 future steps
    return model


# Step 3: Experimentation with Hyperparameters
def compile_and_train_multivariate(model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32,
                                   learning_rate=0.001):
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss='mean_squared_error')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_val, y_val))
    return history


# Step 4: Training and Evaluation
def evaluate_multivariate_model(model, X_test, y_test, scaler):
    predictions = model.predict(X_test)

    # Inverse transform only the relevant columns (the predicted Close prices)
    # Create an array of the same shape as the input but filled with zeros except for the Close price column
    zeros = np.zeros((predictions.shape[0], scaler.n_features_in_))

    # Insert the predictions and y_test back into the zero-filled arrays
    zeros[:, 3] = predictions[:, 0]
    predictions = scaler.inverse_transform(zeros)[:, 3]

    zeros[:, 3] = y_test[:, 0]
    y_test = scaler.inverse_transform(zeros)[:, 3]

    # Calculate and print MSE for each predicted step
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

    # Visualize the prediction vs actual for the first test case
    plt.figure(figsize=(10, 6))
    plt.plot(y_test[:10], label='True Price')
    plt.plot(predictions[:10], label='Predicted Price')
    plt.legend()
    plt.show()


# Main Function to Run the Experiment
def main():
    ticker = 'AMZN'
    start_date = '2018-01-01'
    end_date = '2023-01-01'

    X, y, scaler = download_and_prepare_multivariate_data(ticker, start_date, end_date)

    # Split into training and validation sets
    split_ratio = 0.8
    train_size = int(len(X) * split_ratio)
    X_train, X_val, y_train, y_val = X[:train_size], X[train_size:], y[:train_size], y[train_size:]

    # Build and train the model
    multivariate_model = build_multivariate_model(X_train.shape[1:])

    print("Training Multivariate LSTM model...")
    compile_and_train_multivariate(multivariate_model, X_train, y_train, X_val, y_val, epochs=20, batch_size=32,
                                   learning_rate=0.001)

    # Evaluate the model
    print("Evaluating Multivariate LSTM model...")
    evaluate_multivariate_model(multivariate_model, X_val, y_val, scaler)


if __name__ == "__main__":
    main()




