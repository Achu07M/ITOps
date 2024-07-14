import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib

# Load dataset
df = pd.read_csv('synthetic_hardware_assets.csv')

# Convert date columns to datetime
df['purchase_date'] = pd.to_datetime(df['purchase_date'])
df['warranty_expiry_date'] = pd.to_datetime(df['warranty_expiry_date'])

# Feature engineering: warranty remaining days
df['warranty_remaining'] = (df['warranty_expiry_date'] - pd.Timestamp('now')).dt.days

# Drop original date columns
df = df.drop(columns=['purchase_date', 'warranty_expiry_date'])

# Select relevant features and target variable
features = ['purchase_cost', 'warranty_remaining', 'assigned_location', 'asset_category', 'manufacturer']
X = df[features]
y = df['lifecycle_status'].apply(lambda x: 1 if x == 'In Maintenance' else 0)  # Simplified target variable

# Handle categorical data using one-hot encoding
X = pd.get_dummies(X, columns=['assigned_location', 'asset_category', 'manufacturer'])

# Scale the features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Convert to pandas DataFrame for easier manipulation
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Reshape input to be 3D (samples, time steps, features)
X_reshaped = X_scaled_df.values.reshape((X_scaled_df.shape[0], 1, X_scaled_df.shape[1]))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_reshaped, y, test_size=0.2, random_state=42)

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=False))
model.add(Dropout(0.2))
model.add(Dense(units=1, activation='sigmoid'))  # Binary classification

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save the model and scaler
model.save('lstm_model_new.h5')
joblib.dump(scaler, 'scaler_new.pkl')

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Loss: {loss}")
print(f"Test Accuracy: {accuracy}")
