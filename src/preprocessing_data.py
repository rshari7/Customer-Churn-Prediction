import pandas as pd


def preprocess(csv_path):
    # df = load_data(../data/raw/customer_data_raw.csv)
    df = pd.read_csv(csv_path)

    #----- CLEANING ------
    df = df.drop_duplicates()
    # df = df.dropna()
    # print(df['CustomerID'].nunique())
    # print(len(df))

    #CustomerID not needed
    df = df.drop(columns=['CustomerID'])

    #Check for missing values -> both returned False
    #print(df.isna().values.any())
    #print(df.isnull().values.any())
    # print(df)

    #----- ENCODING -----
    #encoding step, string to numerical
    df['Gender']= df['Gender'].map({'Male': 0,'Female': 1})
    df['Subscription Type'] = df['Subscription Type'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
    df['Contract Length'] = df['Contract Length'].map({'Monthly': 0, 'Quarterly': 1, 'Annual' : 2})

    return df






