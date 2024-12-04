import matplotlib.pyplot as plt

def plot_actual_vs_predicted(y_test, y_pred):
    plt.figure(figsize=(10,6))
    plt.plot(y_test.index, y_test, label='Actual Prices', color='blue')
    plt.plot(y_test.index, y_pred, label='Predicted Prices', color='red')
    plt.title('Bitcoin Price Prediction (Actual vs Predicted)')
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    plt.legend()
    plt.show()
