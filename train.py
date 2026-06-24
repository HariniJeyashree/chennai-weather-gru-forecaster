import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dense
from tensorflow.keras.optimizers import Adam

# 1. CONNECT TO NEON DATABASE
DATABASE_URL = "postgresql://neondb_owner:npg_ECLnG0UfzPo1@ep-square-hall-attwld7t-pooler.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"
engine = create_engine(DATABASE_URL)

print("Fetching real Chennai data from Neon...")
df = pd.read_sql("SELECT * FROM chennai_temperature", engine)

# Sort by time to ensure the timeline is in the correct order
df['time'] = pd.to_datetime(df['time'])
df = df.sort_values('time')
df.set_index('time', inplace=True)

temperature_data = df[['temperature_2m_mean']].values

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(temperature_data)

joblib.dump(scaler, 'scaler.pkl')
print("Saved Chennai scaler.pkl")


def create_dataset(data, time_step=1):
    X, y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i:(i + time_step), 0])
        y.append(data[i + time_step, 0])
    return np.array(X), np.array(y)


time_step = 100
X, y = create_dataset(scaled_data, time_step)
X = X.reshape(X.shape[0], X.shape[1], 1)

# 2. BUILD AND TRAIN THE GRU MODEL
print("Training AI on Chennai weather patterns...")
model = Sequential()
model.add(GRU(units=50, return_sequences=True, input_shape=(X.shape[1], 1)))
model.add(GRU(units=50))
model.add(Dense(units=1))
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

model.fit(X, y, epochs=100, batch_size=32)

model.save('temperature_model.keras')
print("Training complete! Brain saved.")