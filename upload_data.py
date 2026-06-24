import pandas as pd
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv
load_dotenv() # This loads the variables from your .env file

local_csv_path = "chennai_weather.csv"

# 1. READ CSV (Ignoring the top 3 metadata rows)
df = pd.read_csv(local_csv_path, skiprows=3)

# 2. FORCE CLEAN COLUMNS
# Your file has extra commas at the end of rows (like 22.9,,,,). This grabs just the first 2 columns.
df = df.iloc[:, [0, 1]]
df.columns = ['time', 'temperature_2m_mean']

# 3. ENTERPRISE DATA CLEANING (The Magic Fix)
# format='mixed' handles the sudden switch from "2022-01-01T00" to "13-04-2026"
# dayfirst=True tells it that "13-04" means April 13th, not the 13th month!
df['time'] = pd.to_datetime(df['time'], format='mixed', dayfirst=True, errors='coerce')

# Force the temperature column to be actual numbers, not text/strings!
df['temperature_2m_mean'] = pd.to_numeric(df['temperature_2m_mean'], errors='coerce')

# Now we drop all blank rows so our AI only gets perfect data
df = df.dropna()

# 4. CONVERT HOURLY TO DAILY (Temporal Resampling)
print("Squashing mixed hourly/daily data into uniform daily averages...")
df.set_index('time', inplace=True)
df = df.resample('D').mean().reset_index()

# Drop any days that might have become blank during resampling
df = df.dropna()

# 5. UPLOAD TO NEON
# Put your actual Neon URL inside the quotes!
DATABASE_URL = os.getenv("DATABASE_URL")
engine = create_engine(DATABASE_URL)

print("Uploading clean Chennai data to Neon Cloud...")
df.to_sql('chennai_temperature', engine, if_exists='replace', index=False)
print("Data successfully uploaded to Neon!")