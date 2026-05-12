import pandas as pd

#dataset from https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download

#load raw data
def load_data(csv_path):
    df = pd.read_csv(csv_path)
    return df
