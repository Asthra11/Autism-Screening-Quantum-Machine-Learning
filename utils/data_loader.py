import pandas as pd

def load_data():
    """
    Load autism dataset and return DataFrame only
    """
    return pd.read_csv("data/autism_kaggle_data.csv")
