import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Input
import matplotlib.pyplot as plt

# Define data placeholders (for real use, replace with actual data loading)
X_train = tf.random.normal((500, 10, 5))  # Training data
y_train = tf.random.normal((500, 1))  # Training labels
X_val = tf.random.normal((100, 10, 5))  # Validation data
y_val = tf.random.normal((100, 1))  # Validation labels

# Define multistep prediction function
def multistep_prediction(model, data, k):
    """
    Perform multistep prediction for k timesteps into the future.

    Parameters:
    - model: Trained model.
    - data: The last data sequence used to make the first prediction.
    - k: Number of future timesteps to predict.

    Returns:
    - predictions: Array of predicted values for k timesteps.
    """
    predictions = []
    # Convert `data` to a NumPy array to use the `.copy()` method
    current_input = np.array(data).copy()

    for _ in range(k):
        next_prediction = model.predict(current_input.reshape(1, *current_input.shape))[0, 0]
        predictions.append(next_prediction)
        # Update the input for the next prediction
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1, -1] = next_prediction  # Assuming the last column is the feature to predict

    return np.array(predictions)

# Define the input shape based on X_train data
input_shape = X_train.shape[1:]  # (10, 5)

def build_model(input_shape):
    model = Sequential()
    model.add(Input(shape=input_shape))
    model.add(LSTM(50, return_sequences=False, activation='tanh'))
    model.add(Dense(1, activation='relu'))
    model.compile(optimizer='adam', loss='mse')
    return model

# Build and train the model
model = build_model(input_shape)
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=16)

# Get predictions and plot actual vs predicted values
k_steps = 5
last_data_sample = X_val[-1]  # Use last validation sample for prediction
predicted_outcome = multistep_prediction(model, last_data_sample, k_steps)
actual_outcome = np.array(y_val[-k_steps:]).flatten()  # Updated line

# Plot function
def plot_actual_vs_predicted(actual, predicted):
    plt.figure(figsize=(10, 5))
    plt.plot(actual, label='Actual', marker='o')
    plt.plot(predicted, label='Predicted', marker='x')
    plt.xlabel("Time Steps")
    plt.ylabel("Values")
    plt.title("Actual vs. Predicted Values")
    plt.legend()
    plt.show()

# Print and plot the results
print("Actual values:", actual_outcome)
print("Predicted values:", predicted_outcome)
plot_actual_vs_predicted(actual_outcome, predicted_outcome)



