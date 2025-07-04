# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- 1. Load Saved Model and Scaler ---
try:
    model = joblib.load('final_stock_regressor_model.pkl')
    scaler = joblib.load('final_stock_scaler.pkl')
    model_features = list(scaler.feature_names_in_)  # ✅ get required features at load time
except FileNotFoundError:
    st.error("❌ Model or scaler files not found.")
    st.stop()

# --- 2. Data Fetch + Feature Functions ---

@st.cache_data(ttl=3600)
def get_data(ticker_symbol):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
    return df

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

# --- 3. UI ---
st.title('📈 Stock Opening Price Prediction')

st.sidebar.header('Input')
ticker = st.sidebar.text_input('Enter NSE Ticker', 'TATAMOTORS.NS')

if st.sidebar.button('Predict Next Day Open'):
    with st.spinner(f'Fetching & processing data for `{ticker}`...'):
        stock_data = get_data(ticker)

        if stock_data.empty:
            st.error("⚠️ Could not fetch data. Check the symbol.")
        else:
            featured_data = create_features(stock_data)

            if featured_data.empty:
                st.error("⚠️ Not enough data to generate features.")
            else:
                # --- Prediction Pipeline ---
                last_day = featured_data.iloc[[-1]]

                # ✅ Match features exactly with those used in training
                if not all(f in last_day.columns for f in model_features):
                    st.error("❌ Required features not found in data.")
                    st.stop()

                last_day_for_model = last_day[model_features]
                last_day_scaled = scaler.transform(last_day_for_model)
                predicted_pct_change = float(model.predict(last_day_scaled)[0])
                last_close_price = float(last_day['Close'].iloc[0])
                predicted_open = last_close_price * (1 + predicted_pct_change / 100)

                # --- Display ---
                st.header(f'📊 Prediction for {ticker}')
                col1, col2 = st.columns(2)

                with col1:
                    st.metric("Most Recent Close", f"₹{last_close_price:.2f}")
                with col2:
                    st.metric("Predicted Next Open", f"₹{predicted_open:.2f}", f"{predicted_pct_change:.2f}%")

                st.subheader("Recent Processed Data")
                st.dataframe(featured_data.tail(5))

                st.info("💡 This predicts the **next day’s opening price** using historical data. It's not financial advice.")
