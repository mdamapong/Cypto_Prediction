import pandas as pd

def preprocess_data(data):
    data.fillna(data.mean(), inplace=True)
    return data
