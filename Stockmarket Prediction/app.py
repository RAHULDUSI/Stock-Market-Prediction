import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
import plotly.graph_objects as go

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="Stock Market Predictor", layout="wide")

st.title("📈 AI Stock Market Prediction (LSTM)")

ticker = st.text_input("Enter Stock Symbol (Example: AAPL, TSLA, INFY, TCS):", "AAPL")
period = st.selectbox("Select Time Period", ["1y", "2y", "5y", "10y"])
predict_days = st.slider("Predict next how many days?", 1, 30, 1)

if st.button("Predict Stock Price"):
    with st.spinner("Downloading data..."):
        data = yf.download(ticker, period=period)

    st.subheader("📊 Historical Closing Price")
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Closing Price"))
    fig.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig, use_container_width=True)

    # ----------------- DATA PREPROCESS -----------------
    df = data["Close"].values.reshape(-1, 1)

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X_train, y_train = [], []
    prediction_days = 60

    for i in range(prediction_days, len(scaled_data)):
        X_train.append(scaled_data[i - prediction_days:i, 0])
        y_train.append(scaled_data[i, 0])

    X_train, y_train = np.array(X_train), np.array(y_train)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))

    # ----------------- BUILD LSTM MODEL -----------------
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(1))

    model.compile(optimizer="adam", loss="mean_squared_error")

    with st.spinner("Training the LSTM Model (this takes a moment)..."):
        model.fit(X_train, y_train, epochs=5, batch_size=32)

    # ----------------- MAKE FUTURE PREDICTIONS -----------------
    last_60 = scaled_data[-60:]
    future_prices = []

    current_input = last_60.reshape(1, 60, 1)

    for _ in range(predict_days):
        prediction = model.predict(current_input)[0][0]
        future_prices.append(prediction)

        # Update window
        new_input = np.append(current_input.flatten()[1:], prediction)
        current_input = new_input.reshape(1, 60, 1)

    future_prices = scaler.inverse_transform(np.array(future_prices).reshape(-1, 1))

    # ----------------- DISPLAY PREDICTIONS -----------------
    st.subheader(f"📌 Predicted Prices for next {predict_days} days")

    future_dates = pd.date_range(start=data.index[-1] + pd.Timedelta(days=1), periods=predict_days)

    pred_df = pd.DataFrame({"Date": future_dates, "Predicted Price": future_prices.flatten()})

    st.dataframe(pred_df, use_container_width=True)

    # ----------------- PLOTS -----------------
    st.subheader("📈 Future Price Prediction Chart")
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(x=data.index, y=data["Close"], name="Historical"))
    fig2.add_trace(go.Scatter(x=pred_df["Date"], y=pred_df["Predicted Price"], name="Predicted", line=dict(color="red")))
    fig2.update_layout(xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig2, use_container_width=True)

    st.success("✅ Prediction Complete!")
