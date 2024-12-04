# src/model_evaluation.py

from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the performance of the model on the test data.

    Parameters:
        model: The trained machine learning model.
        X_test: Features of the test set.
        y_test: True values for the test set.
    
    Returns:
        mse: Mean Squared Error of the model's predictions.
        r2: R-squared score of the model's predictions.
    """
    y_pred = model.predict(X_test)
    
    mse = mean_squared_error(y_test, y_pred)
    
    r2 = r2_score(y_test, y_pred)
    
    print(f"Mean Squared Error (MSE): {mse}")
    print(f"R-squared (R2): {r2}")
    
    return mse, r2
