import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU, SimpleRNN, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt  # Import for visualization


def build_deep_learning_model(input_shape, layer_config):
    model = Sequential()
    model.add(Input(shape=input_shape))  # Explicit Input Layer

    # Add the first layer
    first_layer = layer_config[0]
    if first_layer['type'] == 'LSTM':
        model.add(
            LSTM(first_layer['units'], activation=first_layer['activation'], return_sequences=(len(layer_config) > 1)))
    elif first_layer['type'] == 'GRU':
        model.add(
            GRU(first_layer['units'], activation=first_layer['activation'], return_sequences=(len(layer_config) > 1)))
    elif first_layer['type'] == 'RNN':
        model.add(SimpleRNN(first_layer['units'], activation=first_layer['activation'],
                            return_sequences=(len(layer_config) > 1)))
    elif first_layer['type'] == 'Dense':
        model.add(Dense(first_layer['units'], activation=first_layer['activation']))

    # Add Dropout if specified
    if 'dropout' in first_layer:
        model.add(Dropout(first_layer['dropout']))

    # Add subsequent layers
    for layer in layer_config[1:]:
        if layer['type'] == 'LSTM':
            model.add(
                LSTM(layer['units'], activation=layer['activation'], return_sequences=(layer != layer_config[-1])))
        elif layer['type'] == 'GRU':
            model.add(GRU(layer['units'], activation=layer['activation'], return_sequences=(layer != layer_config[-1])))
        elif layer['type'] == 'RNN':
            model.add(
                SimpleRNN(layer['units'], activation=layer['activation'], return_sequences=(layer != layer_config[-1])))
        elif layer['type'] == 'Dense':
            model.add(Dense(layer['units'], activation=layer['activation']))

        # Add Dropout if specified
        if 'dropout' in layer:
            model.add(Dropout(layer['dropout']))

    # Compile the model
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model


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
    current_input = data.copy()

    for _ in range(k):
        next_prediction = model.predict(current_input.reshape(1, *current_input.shape))[0, 0]
        predictions.append(next_prediction)
        # Update the input for the next prediction
        current_input = np.roll(current_input, -1, axis=0)
        current_input[-1, -1] = next_prediction  # Assuming last column is the feature to predict

    return np.array(predictions)


def plot_training_history(history):
    """
    Plots the training and validation loss over epochs.

    Parameters:
    - history: The history object returned by model.fit()
    """
    plt.figure(figsize=(10, 6))

    # Plot training loss
    plt.plot(history.history['loss'], label='Training Loss')

    # Plot validation loss
    plt.plot(history.history['val_loss'], label='Validation Loss')

    # Add labels and title
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Example usage: Let's experiment with different configurations
input_shape = (10, 5)  # Example input shape (10 timesteps, 5 features)

# Configurations for experiments
experiments = {
    "LSTM_1layer": [
        {'type': 'LSTM', 'units': 50, 'activation': 'tanh', 'dropout': 0.2},
        {'type': 'Dense', 'units': 1, 'activation': 'relu'}  # Changed to 1 unit
    ],
    "GRU_2layers": [
        {'type': 'GRU', 'units': 50, 'activation': 'tanh', 'dropout': 0.2},
        {'type': 'GRU', 'units': 50, 'activation': 'tanh', 'dropout': 0.2},
        {'type': 'Dense', 'units': 1, 'activation': 'relu'}  # Changed to 1 unit
    ],
    "RNN_3layers": [
        {'type': 'RNN', 'units': 50, 'activation': 'tanh', 'dropout': 0.2},
        {'type': 'RNN', 'units': 50, 'activation': 'tanh'},
        {'type': 'RNN', 'units': 30, 'activation': 'tanh'},
        {'type': 'Dense', 'units': 1, 'activation': 'relu'}  # Changed to 1 unit
    ]
}

# Hyperparameters for experiments
epochs_list = [20, 50]  # Number of epochs
batch_size_list = [16, 32]  # Batch sizes

# Data (Placeholder for real data)
X_train = tf.random.normal((500, 10, 5))  # Example data (500 samples, 10 timesteps, 5 features)
y_train = tf.random.normal((500, 1))  # Example target data (500 samples, 1 target per sample)

X_val = tf.random.normal((100, 10, 5))  # Example validation data
y_val = tf.random.normal((100, 1))  # Example validation targets (1 target per sample)

# Multistep prediction parameters
k_steps = 5  # Number of future steps to predict
last_data_sample = X_val[-1]  # Assuming the last validation sample for prediction

# Loop through experiments
for experiment_name, layer_config in experiments.items():
    for epochs in epochs_list:
        for batch_size in batch_size_list:
            print(f"Experiment: {experiment_name}, Epochs: {epochs}, Batch Size: {batch_size}")

            # Build the model
            model = build_deep_learning_model(input_shape, layer_config)
            model.summary()

            # Train the model
            early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
            history = model.fit(
                X_train, y_train,
                validation_data=(X_val, y_val),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stopping],
                verbose=1
            )

            # Plot the training and validation loss
            plot_training_history(history)

            # Print the final validation loss and MAE
            final_val_loss, final_val_mae = model.evaluate(X_val, y_val)
            print(f"Final Validation Loss: {final_val_loss}, Final Validation MAE: {final_val_mae}")

            # Perform multistep prediction
            predicted_future = multistep_prediction(model, last_data_sample, k_steps)
            print(f"Predicted values for the next {k_steps} timesteps: {predicted_future}")
            print("--------------------------------------------------\n")
