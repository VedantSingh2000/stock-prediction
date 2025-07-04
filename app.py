# app.py

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import joblib
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta

# --- 1. Load The Saved Model and Scaler ---
# Use a try-except block to handle the case where the files are not found.
try:
    model = joblib.load('final_stock_regressor_model.pkl')
    scaler = joblib.load('final_stock_scaler.pkl')
except FileNotFoundError:
    st.error("Error: Model or scaler files not found. Make sure 'final_stock_regressor_model.pkl' and 'final_stock_scaler.pkl' are in the same directory as this script.")
    st.stop() # Stop the app if files are missing


# --- 2. Helper Functions for Data Fetching and Feature Engineering ---
# These functions contain the same logic we developed in the notebook.

@st.cache_data(ttl=3600) # Cache data for 1 hour to avoid re-fetching on every interaction
def get_data(ticker_symbol):
    """Fetches the last 5 years of stock data from yfinance."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=5*365)
    df = yf.download(ticker_symbol, start=start_date, end=end_date, auto_adjust=True)
    return df

def create_features(df):
    """Creates the features required by the model."""
    df_feat = df.copy()
    df_feat['SMA_10'] = df_feat['Close'].rolling(window=10).mean()
    df_feat['SMA_30'] = df_feat['Close'].rolling(window=30).mean()
    df_feat['Price_Change'] = df_feat['Close'].diff()
    df_feat['High_Low_Diff'] = df_feat['High'] - df_feat['Low']
    df_feat['Volume_Change'] = df_feat['Volume'].diff()
    df_feat['Close_Shift_1'] = df_feat['Close'].shift(1)
    df_feat.dropna(inplace=True)
    return df_feat


# --- 3. Streamlit User Interface ---

st.title('Stock Opening Price Prediction App')

# Sidebar for user inputs
st.sidebar.header('User Input')
ticker = st.sidebar.text_input('Stock Ticker', 'TATAMOTORS.NS')

if st.sidebar.button('Predict Next Day Open'):
    # Show a spinner while processing
    with st.spinner(f'Fetching data and predicting for {ticker}...'):
        
        # --- Data Processing and Prediction Pipeline ---
        # Fetch data
        stock_data = get_data(ticker)
        
        if stock_data.empty:
            st.error(f"Could not fetch data for {ticker}. Please check the ticker symbol.")
        else:
            # Create features
            featured_data = create_features(stock_data)
            
            # Get the most recent day's features for prediction
            last_day_features = featured_data.iloc[[-1]]
            
            # Extract the most recent closing price
            last_close_price = float(last_day_features['Close'].iloc[0])
            
            # Select only the columns that the model was trained on
            # (This prevents errors if create_features adds extra columns)
            model_features = [col for col in last_day_features.columns if col in scaler.feature_names_in_]
            last_day_features_for_model = last_day_features[model_features]

            # Scale the features
            last_day_scaled = scaler.transform(last_day_features_for_model)

            # Predict the percentage change
            predicted_pct_change = float(model.predict(last_day_scaled)[0])

            # Calculate the predicted open price
            predicted_next_open = last_close_price * (1 + predicted_pct_change / 100)

            # --- Display Results ---
            st.header(f'Prediction for {ticker}')
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Most Recent Close Price",
                    value=f"₹{last_close_price:.2f}",
                    delta_color="off"
                )
            
            with col2:
                st.metric(
                    label="Predicted Next Day Open",
                    value=f"₹{predicted_next_open:.2f}",
                    delta=f"{predicted_pct_change:.2f}%"
                )
            
            st.info("💡 **Note:** This model predicts the **opening price** only. Predicting the closing price is a separate, more complex task that would require a different model.")

            # Display the recent data for context
            st.subheader("Recent Data Used for Prediction")
            st.dataframe(featured_data.tail(5))