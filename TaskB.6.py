import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from statsmodels.tsa.arima.model import ARIMA
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# ARIMA model
def train_arima_model(data, order=(5, 1, 0)):
    model = ARIMA(data, order=order)
    model_fit = model.fit()
    return model_fit

# LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, activation='tanh', input_shape=input_shape))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Random Forest model
def train_random_forest(data, feature_names, n_estimators=100):
    X = data[feature_names]
    y = data['close']
    model = RandomForestRegressor(n_estimators=n_estimators, random_state=42)
    model.fit(X, y)
    return model

# Prepare features for prediction
def prepare_features_for_prediction(data, feature_names):
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data, columns=feature_names)

    X = data[feature_names].iloc[-1].values.reshape(1, -1)
    return X

# Multistep LSTM prediction
def multistep_prediction(model, data, k):
    predictions = []
    current_input = data
    for _ in range(k):
        next_prediction = model.predict(current_input.reshape(1, *current_input.shape))[0, 0]
        predictions.append(next_prediction)
        current_input = np.roll(current_input, -1)
        current_input[-1] = next_prediction
    return np.array(predictions)

# Ensemble prediction with ARIMA, LSTM, and Random Forest
def ensemble_with_rf(arima_model, lstm_model, rf_model, last_data, k, feature_names):
    arima_forecast = arima_model.forecast(steps=k)
    lstm_forecast = multistep_prediction(lstm_model, last_data, k)

    rf_forecast = []
    for _ in range(k):
        X_for_rf = prepare_features_for_prediction(last_data, feature_names)
        rf_prediction = rf_model.predict(X_for_rf)[0]
        rf_forecast.append(rf_prediction)

    ensemble_predictions = (arima_forecast + lstm_forecast + rf_forecast) / 3
    return ensemble_predictions

# Example usage:
# Define your feature names
feature_names = ['open', 'high', 'low', 'volume']  # Adjust this to your actual features

# Assuming train_data is your DataFrame with the appropriate columns
arima_model = train_arima_model(train_data['close'])
input_shape = (10, len(feature_names))  # Assuming you have 10 timesteps and len(feature_names) features
model_lstm = build_lstm_model(input_shape)
rf_model = train_random_forest(train_data, feature_names)

# Define last_data as the last n rows of the relevant features
last_data = train_data[feature_names].tail(10).values  # Use the last 10 rows as input data for LSTM

# Perform the ensemble prediction
k = 5  # Predicting 5 days into the future
ensemble_forecast_with_rf = ensemble_with_rf(arima_model, model_lstm, rf_model, last_data, k, feature_names)
print("Ensemble predictions with RF for the next 5 days:", ensemble_forecast_with_rf)