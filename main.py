# main.py

import yfinance as yf
from src.data_preprocessing import preprocess_data
from src.model_training import train_model, evaluate_model
from src.visualization import plot_actual_vs_predicted
from sklearn.model_selection import train_test_split

def main():
    print("Downloading data...")
    data = yf.download('BTC-USD', start='2015-01-01', end='2024-12-31')
    
    print("Preprocessing data...")
    data = preprocess_data(data)
    
    print("Creating features and target...")
    X = data[['Open', 'High', 'Low', 'Volume']]
    y = data['Close']

    print("Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    print("Training the model...")
    model = train_model(X_train, y_train)

    print("Evaluating the model...")
    mse = evaluate_model(model, X_test, y_test)
    print(f"Model evaluation completed with MSE: {mse}")

    print("Plotting the results...")
    y_pred = model.predict(X_test)
    plot_actual_vs_predicted(y_test, y_pred)
    print("Done!")

if __name__ == "__main__":
    main()
