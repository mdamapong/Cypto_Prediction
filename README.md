# Cryptocurrency Price Prediction using Random Forest

## Project Overview

This project aims to predict the **closing price** of various cryptocurrencies using historical price data. A **Random Forest Regressor** model is utilized to make predictions based on historical data features like the opening price, high price, low price, and trading volume. The project also includes a **Streamlit** web interface that allows users to interactively select cryptocurrencies, choose date ranges, and visualize the predicted vs actual prices.

## Features

- Predict **closing prices** of cryptocurrencies using **Random Forest**.
- User interface created with **Streamlit**.
- Visualize the **actual vs predicted prices** over time.
- Model performance metrics, including **Mean Squared Error (MSE)** and **R-squared (R²)**.

## Technologies Used

- **Python**: Programming language used for building the model and the Streamlit app.
- **Streamlit**: Framework for building the web interface.
- **Random Forest Regressor**: Model used for predicting cryptocurrency prices.
- **yfinance**: Library to fetch historical cryptocurrency price data from Yahoo Finance.
- **Pandas**: Data manipulation and analysis.
- **Scikit-learn**: For building the Random Forest model and evaluating performance.
- **Matplotlib**: For plotting the actual vs predicted prices.

## Installation

1. Clone this repository to your local machine:
   ```bash
   git clone https://github.com/your-username/crypto-price-prediction.git
   Change into the project directory:
   ```

bash
Copy code
cd crypto-price-prediction
Create and activate a virtual environment:

bash
Copy code
python3 -m venv venv
source venv/bin/activate # On Windows, use: venv\Scripts\activate
Install the required dependencies:

bash
Copy code
pip install -r requirements.txt
Install Streamlit and other required libraries (if not included in requirements.txt):

bash
Copy code
pip install streamlit yfinance pandas scikit-learn matplotlib
Usage
Running the Streamlit Web App
Once the dependencies are installed, you can run the Streamlit app by executing the following command:

bash
Copy code
streamlit run app.py
This will start the app locally, and you can view it in your browser at http://localhost:8501

Acknowledgments
Random Forest: A versatile and robust ensemble method used for regression tasks.
Streamlit: For creating an interactive and user-friendly web interface.
yfinance: For providing historical cryptocurrency data.
