import streamlit as st
import pandas as pd
import numpy as np
import joblib
from tensorflow.keras.models import load_model
from sqlalchemy import create_engine

# --- 1. PAGE CONFIGURATION (Makes it look attractive) ---
st.set_page_config(page_title="Chennai Weather AI", page_icon="🌡️", layout="centered")

st.title("🌤️ Chennai AI Forecaster")
st.markdown("Predicting today's temperature in Chennai based on the last 100 days of real-world data.")
st.divider()

# --- 2. LOAD FILES ONCE ---
# This decorator tells Streamlit to only load the heavy AI files once and keep them in memory
@st.cache_resource 
def load_ai():
    loaded_model = load_model('temperature_model.keras')
    loaded_scaler = joblib.load('scaler.pkl')
    return loaded_model, loaded_scaler

model, scaler = load_ai()

# --- 3. FETCH EXACTLY 100 DAYS FROM NEON ---
DATABASE_URL = "postgresql://neondb_owner:npg_ECLnG0UfzPo1@ep-square-hall-attwld7t-pooler.c-9.us-east-1.aws.neon.tech/neondb?sslmode=require&channel_binding=require"  # Put your Neon URL here
engine = create_engine(DATABASE_URL)

# Notice the LIMIT 100 because your time_step is 100!
query = 'SELECT * FROM chennai_temperature ORDER BY "time" DESC LIMIT 100'
df_recent = pd.read_sql(query, engine)

# Reverse it so time flows forward
df_recent = df_recent.sort_values('time', ascending=True)

# --- 4. ATTRACTIVE VISUALIZATION ---
st.subheader("📊 Past 100 Days of Temperature in chennai")
# Streamlit's built in chart makes this look great automatically
st.line_chart(df_recent.set_index('time'))

# --- 5. THE PREDICTION BUTTON ---
st.subheader("🔮 Today's Forecast")

if st.button("Generate AI Prediction", type="primary"):
    with st.spinner("The AI is analyzing the timeline..."):
        # 1. Extract values
        recent_values = df_recent[['temperature_2m_mean']].values

        # 2. Scale the data (0 to 1)
        scaled_input = scaler.transform(recent_values)
        
        # 3. Reshape for your AI: (1 sample, 100 days, 1 feature)
        reshaped_input = scaled_input.reshape(1, 100, 1)
        
        # 4. Predict
        scaled_prediction = model.predict(reshaped_input)
        
        # 5. Translate back to real degrees
        real_prediction = scaler.inverse_transform(scaled_prediction)
        predicted_temp = real_prediction[0][0]
        
        # Display an attractive metric widget
        st.metric(label="Predicted Mean Temperature for June 24", value=f"{predicted_temp:.2f} °C")
        
        if predicted_temp > 30:
            st.warning("⚠️ It's going to be a hot day!")
        else:
            st.info("🌤️ The weather looks manageable.")