import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from math import sqrt
import warnings
warnings.filterwarnings('ignore')

# Page Configuration
st.set_page_config(page_title="Copper Price Forecasting", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for better styling
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4ECDC4;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Title and Description
st.markdown('<p class="main-header">🔮 Hybrid ARIMA + LSTM Copper Price Forecasting</p>', unsafe_allow_html=True)
st.markdown("Upload your dataset with two columns: **Date** and **Price** to start forecasting.")

# Helper function for ARIMA parameter selection
@st.cache_data
def find_best_arima_order(train_data, max_p=5, max_d=2, max_q=5):
    """Find best ARIMA parameters using grid search"""
    best_aic = np.inf
    best_order = None
    
    # Check stationarity
    adf_result = adfuller(train_data)
    d = 0 if adf_result[1] < 0.05 else 1
    
    for p in range(0, max_p + 1):
        for q in range(0, max_q + 1):
            try:
                model = ARIMA(train_data, order=(p, d, q))
                fitted = model.fit()
                if fitted.aic < best_aic:
                    best_aic = fitted.aic
                    best_order = (p, d, q)
            except:
                continue
    
    return best_order if best_order else (1, d, 1)

# Sidebar Configuration
with st.sidebar:
    st.header("⚙️ Configuration")
    train_split = st.slider("Train-Test Split Ratio", 0.6, 0.9, 0.8, 0.05)
    
    st.markdown("### ARIMA Parameters")
    use_auto_arima = st.checkbox("Auto-detect ARIMA parameters", value=True)
    if not use_auto_arima:
        arima_p = st.slider("AR order (p)", 0, 5, 1)
        arima_d = st.slider("Differencing (d)", 0, 2, 1)
        arima_q = st.slider("MA order (q)", 0, 5, 1)
    
    st.markdown("### LSTM Parameters")
    lookback_window = st.slider("LSTM Lookback Window", 30, 120, 60, 10)
    lstm_epochs = st.slider("LSTM Training Epochs", 20, 100, 50, 10)
    lstm_units = st.slider("LSTM Units", 32, 128, 64, 32)
    
    st.markdown("### Forecasting")
    future_days = st.slider("Future Forecast Days", 30, 365, 90, 30)
    
    st.markdown("---")
    st.markdown("### 📊 Model Info")
    st.info("**ARIMA**: Captures linear trends and seasonality\n\n**LSTM**: Learns non-linear patterns from residuals")

# File Upload Section
st.markdown('<p class="sub-header">📁 Step 1: Upload Dataset</p>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choose a CSV file (Date, Price)", type=["csv"])

if uploaded_file is not None:
    try:
        # Load and preprocess data
        df = pd.read_csv(uploaded_file)
        
        # Validate columns
        if len(df.columns) < 2:
            st.error("❌ Dataset must have at least 2 columns: Date and Price")
            st.stop()
        
        # Rename columns to standard names
        df.columns = ['Date', 'Price'] + list(df.columns[2:]) if len(df.columns) > 2 else ['Date', 'Price']
        df = df[['Date', 'Price']]  # Keep only required columns
        
        # Parse date with specific format: %d-%m-%Y (e.g., 23-12-2025)
        df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y', errors='coerce')
        
        # Check for parsing errors
        if df['Date'].isnull().any():
            st.error("❌ Some dates could not be parsed. Please check your date format.")
            st.stop()
        
        # Remove duplicates and sort
        df = df.drop_duplicates(subset=['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
        df.set_index('Date', inplace=True)
        
        # Handle missing values
        if df['Price'].isnull().any():
            st.warning("⚠️ Missing values detected. Filling with forward fill method.")
            df['Price'].fillna(method='ffill', inplace=True)
        
        # Display dataset info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Records", len(df))
        with col2:
            st.metric("Date Range", f"{df.index[0].date()} to {df.index[-1].date()}")
        with col3:
            st.metric("Avg Price", f"${df['Price'].mean():.2f}")
        
        # Plot original data
        st.markdown('<p class="sub-header">📈 Historical Copper Prices</p>', unsafe_allow_html=True)
        fig, ax = plt.subplots(figsize=(12, 4))
        ax.plot(df.index, df['Price'], color='#4ECDC4', linewidth=2)
        ax.set_xlabel('Date', fontsize=10)
        ax.set_ylabel('Price', fontsize=10)
        ax.set_title('Historical Copper Price Data', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Train-Test Split
        st.markdown('<p class="sub-header">✂️ Step 2: Train-Test Split</p>', unsafe_allow_html=True)
        split_idx = int(len(df) * train_split)
        train = df.iloc[:split_idx].copy()
        test = df.iloc[split_idx:].copy()
        
        col1, col2 = st.columns(2)
        with col1:
            st.success(f"✅ Training Set: {len(train)} samples ({train_split*100:.0f}%)")
        with col2:
            st.success(f"✅ Testing Set: {len(test)} samples ({(1-train_split)*100:.0f}%)")
        
        # ARIMA Model Training
        st.markdown('<p class="sub-header">🧠 Step 3: ARIMA Model Training</p>', unsafe_allow_html=True)
        
        with st.spinner("🔄 Training ARIMA model... This may take a moment."):
            try:
                # Determine ARIMA order
                if use_auto_arima:
                    with st.spinner("Finding optimal ARIMA parameters..."):
                        best_order = find_best_arima_order(train['Price'])
                    st.info(f"Auto-detected ARIMA order: {best_order}")
                else:
                    best_order = (arima_p, arima_d, arima_q)
                    st.info(f"Using manual ARIMA order: {best_order}")
                
                # Fit ARIMA model
                arima_model = ARIMA(train['Price'], order=best_order)
                arima_fitted = arima_model.fit()
                
                # Forecast on test set
                arima_forecast = arima_fitted.forecast(steps=len(test))
                test['ARIMA_Pred'] = arima_forecast.values
                
                # Calculate ARIMA metrics
                arima_rmse = sqrt(mean_squared_error(test['Price'], test['ARIMA_Pred']))
                arima_mae = mean_absolute_error(test['Price'], test['ARIMA_Pred'])
                
                st.success(f"✅ ARIMA Model Trained: {best_order}")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("ARIMA RMSE", f"{arima_rmse:.4f}")
                with col2:
                    st.metric("ARIMA MAE", f"{arima_mae:.4f}")
                
                # Plot ARIMA results
                fig, ax = plt.subplots(figsize=(12, 4))
                ax.plot(test.index, test['Price'], label='Actual', color='#2C3E50', linewidth=2)
                ax.plot(test.index, test['ARIMA_Pred'], label='ARIMA Forecast', color='#E74C3C', linewidth=2, linestyle='--')
                ax.set_xlabel('Date', fontsize=10)
                ax.set_ylabel('Price', fontsize=10)
                ax.set_title('ARIMA Model Performance', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ ARIMA training failed: {str(e)}")
                st.stop()
        
        # Calculate Residuals
        residuals = test['Price'] - test['ARIMA_Pred']
        
        # LSTM Model Training
        st.markdown('<p class="sub-header">🤖 Step 4: LSTM Model Training on Residuals</p>', unsafe_allow_html=True)
        
        # Check if we have enough data for LSTM
        if len(residuals) <= lookback_window:
            st.error(f"❌ Not enough test data for LSTM. Need at least {lookback_window+1} samples, have {len(residuals)}.")
            st.warning("💡 Try reducing the lookback window in the sidebar or using a smaller train-test split.")
            st.stop()
        
        with st.spinner("🔄 Training LSTM model on ARIMA residuals..."):
            try:
                # Scale residuals
                scaler = MinMaxScaler(feature_range=(0, 1))
                scaled_residuals = scaler.fit_transform(residuals.values.reshape(-1, 1))
                
                # Create sequences
                X, y = [], []
                for i in range(lookback_window, len(scaled_residuals)):
                    X.append(scaled_residuals[i-lookback_window:i, 0])
                    y.append(scaled_residuals[i, 0])
                
                X, y = np.array(X), np.array(y)
                X = np.reshape(X, (X.shape[0], X.shape[1], 1))
                
                # Build LSTM model
                model = Sequential([
                    LSTM(lstm_units, return_sequences=True, input_shape=(lookback_window, 1)),
                    Dropout(0.2),
                    LSTM(lstm_units // 2, return_sequences=False),
                    Dropout(0.2),
                    Dense(1)
                ])
                
                model.compile(optimizer='adam', loss='mse')
                
                # Train model
                history = model.fit(X, y, epochs=lstm_epochs, batch_size=16, verbose=0, validation_split=0.1)
                
                st.success(f"✅ LSTM Model Trained ({lstm_epochs} epochs)")
                
                # Plot training history
                fig, ax = plt.subplots(figsize=(10, 3))
                ax.plot(history.history['loss'], label='Training Loss', color='#3498DB')
                ax.plot(history.history['val_loss'], label='Validation Loss', color='#E67E22')
                ax.set_xlabel('Epoch', fontsize=10)
                ax.set_ylabel('Loss', fontsize=10)
                ax.set_title('LSTM Training History', fontsize=12, fontweight='bold')
                ax.legend()
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
            except Exception as e:
                st.error(f"❌ LSTM training failed: {str(e)}")
                st.stop()
        
        # Hybrid Forecasting
        st.markdown('<p class="sub-header">🔄 Step 5: Hybrid Model Predictions</p>', unsafe_allow_html=True)
        
        # LSTM predictions
        lstm_pred_scaled = model.predict(X, verbose=0)
        lstm_pred = scaler.inverse_transform(lstm_pred_scaled).flatten()
        
        # Combine ARIMA + LSTM
        test['LSTM_Correction'] = np.nan
        test['Hybrid_Pred'] = np.nan
        
        test.iloc[lookback_window:, test.columns.get_loc('LSTM_Correction')] = lstm_pred
        test.iloc[lookback_window:, test.columns.get_loc('Hybrid_Pred')] = (
            test['ARIMA_Pred'].iloc[lookback_window:].values + lstm_pred
        )
        
        # Calculate Hybrid metrics
        valid_mask = ~test['Hybrid_Pred'].isna()
        hybrid_rmse = sqrt(mean_squared_error(test.loc[valid_mask, 'Price'], test.loc[valid_mask, 'Hybrid_Pred']))
        hybrid_mae = mean_absolute_error(test.loc[valid_mask, 'Price'], test.loc[valid_mask, 'Hybrid_Pred'])
        
        # Display metrics comparison
        st.markdown("### 📊 Model Performance Comparison")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("ARIMA RMSE", f"{arima_rmse:.4f}")
        with col2:
            st.metric("ARIMA MAE", f"{arima_mae:.4f}")
        with col3:
            st.metric("Hybrid RMSE", f"{hybrid_rmse:.4f}", delta=f"{hybrid_rmse-arima_rmse:.4f}")
        with col4:
            st.metric("Hybrid MAE", f"{hybrid_mae:.4f}", delta=f"{hybrid_mae-arima_mae:.4f}")
        
        # Calculate improvement
        rmse_improvement = ((arima_rmse - hybrid_rmse) / arima_rmse) * 100
        mae_improvement = ((arima_mae - hybrid_mae) / arima_mae) * 100
        
        if rmse_improvement > 0:
            st.success(f"🎉 Hybrid model improved RMSE by {rmse_improvement:.2f}% and MAE by {mae_improvement:.2f}%")
        else:
            st.warning("⚠️ Hybrid model did not improve over ARIMA. Consider adjusting parameters.")
        
        # Plot all predictions
        fig, ax = plt.subplots(figsize=(14, 5))
        ax.plot(test.index, test['Price'], label='Actual Price', color='#2C3E50', linewidth=2.5)
        ax.plot(test.index, test['ARIMA_Pred'], label='ARIMA Forecast', color='#E74C3C', linewidth=2, linestyle='--', alpha=0.7)
        ax.plot(test.index[lookback_window:], test['Hybrid_Pred'].iloc[lookback_window:], 
                label='Hybrid Forecast', color='#27AE60', linewidth=2.5)
        ax.set_xlabel('Date', fontsize=11)
        ax.set_ylabel('Price', fontsize=11)
        ax.set_title('Model Predictions Comparison', fontsize=13, fontweight='bold')
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)
        st.pyplot(fig)
        plt.close()
        
        # Future Forecasting
        st.markdown('<p class="sub-header">🚀 Step 6: Future Price Forecasting</p>', unsafe_allow_html=True)
        
        with st.spinner(f"🔮 Forecasting next {future_days} days..."):
            try:
                # ARIMA future forecast
                future_arima = arima_fitted.forecast(steps=future_days)
                
                # Create future dates
                last_date = df.index[-1]
                future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=future_days, freq='D')
                
                future_df = pd.DataFrame({
                    'Date': future_dates,
                    'ARIMA_Forecast': future_arima.values
                })
                future_df.set_index('Date', inplace=True)
                
                # Plot future forecast
                fig, ax = plt.subplots(figsize=(14, 5))
                ax.plot(df.index[-180:], df['Price'].iloc[-180:], label='Historical Price', color='#2C3E50', linewidth=2)
                ax.plot(future_df.index, future_df['ARIMA_Forecast'], label='Future Forecast', 
                        color='#9B59B6', linewidth=2.5, linestyle='--')
                ax.axvline(x=df.index[-1], color='red', linestyle=':', linewidth=1.5, label='Forecast Start')
                ax.set_xlabel('Date', fontsize=11)
                ax.set_ylabel('Price', fontsize=11)
                ax.set_title(f'Future Copper Price Forecast ({future_days} days)', fontsize=13, fontweight='bold')
                ax.legend(loc='best')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
                plt.close()
                
                # Display forecast statistics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Forecast Mean", f"${future_arima.mean():.2f}")
                with col2:
                    st.metric("Forecast Min", f"${future_arima.min():.2f}")
                with col3:
                    st.metric("Forecast Max", f"${future_arima.max():.2f}")
                
                # Download forecast
                st.markdown("### 💾 Download Forecast Data")
                csv_buffer = future_df.to_csv()
                st.download_button(
                    label="📥 Download Future Forecast CSV",
                    data=csv_buffer,
                    file_name=f"copper_forecast_{future_days}days.csv",
                    mime="text/csv"
                )
                
            except Exception as e:
                st.error(f"❌ Future forecasting failed: {str(e)}")
        
        # Model Summary
        with st.expander("📋 View Detailed Model Summary"):
            st.markdown("### ARIMA Model Summary")
            st.text(str(arima_fitted.summary()))
            
            st.markdown("### LSTM Model Architecture")
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.text('\n'.join(model_summary))
        
    except Exception as e:
        st.error(f"❌ Error processing file: {str(e)}")
        import traceback
        st.code(traceback.format_exc())
        st.stop()

else:
    st.info("👆 Please upload a CSV file to begin forecasting.")
    
    # Show sample data format
    st.markdown("### 📝 Expected Data Format")
    st.markdown("**Date format:** `dd-mm-yyyy` (e.g., 23-12-2025 for December 23, 2025)")
    sample_df = pd.DataFrame({
        'Date': ['01-01-2023', '02-01-2023', '03-01-2023'],
        'Price': [8500.50, 8520.75, 8495.25]
    })
    st.dataframe(sample_df)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #7f8c8d;'>
        <p>Built with ❤️ using Streamlit, ARIMA, and LSTM | © 2025 Copper Price Forecasting</p>
    </div>
""", unsafe_allow_html=True)