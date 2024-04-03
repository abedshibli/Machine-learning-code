import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Bidirectional, LSTM, Dropout, Dense, Activation
from tensorflow.keras.optimizers import Adam


data = pd.read_csv("data.csv")
window_length = 4
num_features = lotto_data.shape[1] - 1  # Excluding the date column
training_data = lotto_data.drop(columns=['Date']).copy()

scaler = StandardScaler()
scaled_training_data = scaler.fit_transform(training_data)

train_samples = []
train_labels = []
for i in range(len(training_data) - window_length):
    train_samples.append(scaled_training_data[i:i + window_length])
    train_labels.append(scaled_training_data[i + window_length])

x_train = np.array(train_samples)
y_train = np.array(train_labels)

model = Sequential([
    Bidirectional(LSTM(160, input_shape=(window_length, num_features), return_sequences=True)),
    Activation('relu'),
    Dropout(0.3),
    Bidirectional(LSTM(160, return_sequences=True)),
    Activation('relu'),
    Dropout(0.3),
    Bidirectional(LSTM(200, return_sequences=True)),
    Activation('relu'),
    Bidirectional(LSTM(240, return_sequences=False)),
    Activation('relu'),
    Dropout(0.5),
    Dense(37),
    Activation('softmax'),
    Dense(num_features)
])

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.0001), loss='mse', metrics=['accuracy'])

# Train the model
model.fit(x_train, y_train, batch_size=160, epochs=25000, verbose=1)

# Make predictions
next_data = training_data.drop(columns=['Date']).head(window_length)
next_data_scaled = scaler.transform(next_data)
y_next_pred = model.predict(np.array([next_data_scaled]))

# Invert scaling and print predictions
predicted_numbers = scaler.inverse_transform(y_next_pred).astype(int)[0]
print("The predicted numbers (without rounding up):", predicted_numbers - 1)
print("The predicted numbers (without rounding up):", predicted_numbers)
print("The predicted numbers (with rounding up):", predicted_numbers + 1)
print("The real numbers are:", lotto_data.iloc[0].drop('Date').tolist())
