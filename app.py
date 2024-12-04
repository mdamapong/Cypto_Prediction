import streamlit as st
import yfinance as yf
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

def preprocess_data(data):
    data.fillna(data.mean(), inplace=True)  
    return data

def train_model(X_train, y_train):
    model = RandomForestRegressor(n_estimators=100)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mse, r2, y_pred

def plot_results(y_test, y_pred):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    ax.plot(y_test.index, y_pred, label='Predicted Prices', color='red')

    ax.set_title('Cryptocurrency Price Prediction (Actual vs Predicted)')
    ax.set_xlabel('Date')
    ax.set_ylabel('Price (USD)')
    ax.legend()

    st.pyplot(fig)

def main():
    st.title("Cryptocurrency Price Prediction with Random Forest")
    
    crypto = st.selectbox("Select Cryptocurrency", ["BTC-USD", "ETH-USD", "XRP-USD", "LTC-USD", "ADA-USD"])
    start_date = st.date_input("Start Date", pd.to_datetime("2015-01-01"))
    end_date = st.date_input("End Date", pd.to_datetime("2024-12-31"))

    if st.button("Predict"):
        data = yf.download(crypto, start=start_date, end=end_date)
        
        st.write(f"Data for {crypto} from {start_date} to {end_date}")
        st.dataframe(data.tail())  # Show the latest data

        X = data[['Open', 'High', 'Low', 'Volume']]
        y = data['Close']
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
        
        model = train_model(X_train, y_train)
        
        mse, r2, y_pred = evaluate_model(model, X_test, y_test)
        
        st.write(f"Mean Squared Error: {mse}")
        st.write(f"R-squared: {r2}")
        
        plot_results(y_test, y_pred)


if __name__ == "__main__":
    main()
