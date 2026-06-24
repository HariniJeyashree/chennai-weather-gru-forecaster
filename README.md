# 🌤️ Chennai AI Weather Forecaster: Predicting Today's Mean Temperature

An end-to-end Machine Learning and Data Engineering pipeline that predicts the true 24-hour mean temperature for Chennai, India, utilizing historical meteorological data.

# 🚀 Project Overview

This project demonstrates a complete, production-level Machine Learning workflow. It bridges the gap between exploratory data science and software engineering by ingesting messy raw data, cleaning and resampling it, storing it in a serverless cloud database, training a Recurrent Neural Network (GRU), and serving predictions via an interactive web dashboard.

# 📊 The "Mean Temperature" Challenge

While consumer weather applications typically highlight the afternoon peak (Daytime High), this project tackles a more complex statistical challenge: predicting the 24-hour mathematical mean.

For example, on a typical June day in Chennai, the temperature might peak at 36°C in the afternoon but drop to 26°C overnight. This model successfully learns the historical volatility to forecast the true average (~31°C) rather than just the extreme peaks, showcasing a deep understanding of time-series smoothing and data granularity.

# 🏗️ Architecture & Tech Stack

Exploratory Data Analysis: Jupyter Notebook (Gru.ipynb)

Data Ingestion & ETL: Pandas, NumPy

Cloud Database: Serverless PostgreSQL via Neon (SQLAlchemy, psycopg2)

Deep Learning: TensorFlow / Keras (GRU Network), Scikit-Learn (MinMaxScaler)

Frontend UI: Streamlit

# 🧠 Key Technical Highlights

1. Automated ETL Pipeline

Engineered a robust data uploading script (upload_data.py) capable of handling dirty real-world data. The pipeline bypasses noisy metadata, dynamically parses mixed string-to-datetime formats (ISO 8601 & European), and executes temporal resampling to squash varying hourly readings into uniform daily metrics. Error handling (errors='coerce') ensures the pipeline does not crash on corrupt data rows.

2. Time-Series Deep Learning

Implemented a Gated Recurrent Unit (GRU) neural network over a traditional LSTM. GRUs provide similar performance for time-series forecasting but are computationally lighter and train faster. The model utilizes a 100-day sliding window mechanism to capture seasonal temperature patterns and long-term dependencies without overfitting.

3. Model Persistence & Decoupled Architecture

Decoupled the heavy training architecture from the production storefront. The model saves its trained weights (temperature_model.keras) and scaling parameters (scaler.pkl), allowing the Streamlit web app to serve predictions in milliseconds using O(1) inference without requiring runtime retraining.

# 📂 Repository Structure

Gru.ipynb: Initial exploratory data analysis, data scaling, and model prototyping.

upload_data.py: The ETL script that cleans the raw CSV and pushes it to the Neon PostgreSQL database.

train.py: The training pipeline that fetches data from the cloud, creates sliding windows, trains the GRU model, and saves the artifacts.

app.py: The production Streamlit application that fetches the latest 100 days of data and outputs the AI forecast.

.gitignore: Ensures virtual environments (.venv) and local datasets are not committed to version control.

# ⚙️ How to Run Locally

Clone the repository

git clone https://github.com/YourUsername/chennai-weather-gru-forecaster.git
cd chennai-weather-gru-forecaster


Set up a Virtual Environment (Recommended)

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate


# Install dependencies

pip install pandas numpy joblib sqlalchemy psycopg2-binary scikit-learn tensorflow streamlit jupyter


# Run the application

streamlit run app.py


(Note: The pipeline requires an active Neon PostgreSQL database URL to fetch the latest historical data).
