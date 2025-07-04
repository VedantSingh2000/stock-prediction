# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- 1. Load Model and Scaler ---
try:
    model = joblib.load('final_stock_regressor_model.pkl')
    scaler = joblib.load('final_stock_scaler.pkl')
    model_features = list(scaler.feature_names_in_)  # required for correct column order
except FileNotFoundError:
    st.error("❌ Model or scaler file not found. Make sure 'final_stock_regressor_model.pkl' and 'final_stock_scaler.pkl' exist.")
    st.stop()

# --- 2. Cached Data Fetching ---
@st.cache_data(ttl=3600)
def get_data(ticker_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
    return df

# --- 3. Feature Engineering ---
def create_features(df):
    df_feat = df.copy()
    df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['SMA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['Price_Change'] = df_feat['Close'].diff()
    df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low']
    df_feat['Volume_Change'] = df_feat['Volume'].diff()
    df_feat['Close_Shift_1'] = df_feat['Close'].shift(1)
    df_feat.dropna(inplace=True)
    return df_feat

# --- 4. Streamlit UI ---
st.set_page_config(page_title="Stock Opening Price Predictor", layout="wide")
st.title('📈 Stock Opening Price Prediction App')
st.markdown("Predict the **next day's opening price** using machine learning.")

# Sidebar
st.sidebar.header('Input Settings')
ticker = st.sidebar.text_input('Enter NSE Stock Ticker', 'TATAMOTORS.NS')

# --- 5. Prediction Trigger ---
if st.sidebar.button('Predict Next Day Open'):
    with st.spinner(f'Fetching data and predicting for {ticker}...'):

        stock_data = get_data(ticker)
        if stock_data.empty:
            st.error(f"⚠️ Could not fetch data for `{ticker}`. Please check the ticker symbol.")
            st.stop()

        featured_data = create_features(stock_data)
        if featured_data.empty:
            st.error("⚠️ Not enough data to compute features.")
            st.stop()

        # Get last row
        last_day = featured_data.iloc[[-1]]

        # Match model features
        try:
            last_day_for_model = last_day[model_features]
        except KeyError as e:
            st.error(f"❌ Feature mismatch: {e}")
            st.stop()

        # Scale & Predict
        last_day_scaled = scaler.transform(last_day_for_model)
        prediction_array = model.predict(last_day_scaled)

        if prediction_array.shape != (1,):
            st.error(f"Unexpected prediction shape: {prediction_array.shape}")
            st.stop()

        predicted_pct_change = float(prediction_array[0])
        last_close_price = float(last_day['Close'].iloc[0])
        predicted_open = last_close_price * (1 + predicted_pct_change / 100)

        # --- 6. Show Results ---
        st.header(f'📊 Prediction for {ticker}')

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Most Recent Close Price", f"₹{last_close_price:.2f}")

        with col2:
            st.metric("Predicted Next Open", f"₹{predicted_open:.2f}", delta=f"{predicted_pct_change:.2f}%")

        st.info("💡 This prediction uses the most recent available data. Model is trained on past 5-year trends.")

        # Optional: show latest rows
        st.subheader("🔍 Recent Data Snapshot")
        st.dataframe(featured_data.tail(5), use_container_width=True)
