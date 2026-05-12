from sklearn.preprocessing import StandardScaler
import pandas as pd
from src.preprocessing_data import preprocess
from src.model import train_model, save_model, category_of_risk
from sklearn.model_selection import train_test_split

def main():
    df = preprocess('../data/raw/customer_data_raw.csv')

     #----- SPLIT DATA -----
    #Proper X/y split, X = features, y = target (churn)
    X = df.drop(columns=['Churn'])
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    #----- STANDARDIZE -----
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    #----- TRAIN ------
    model = train_model(X_train, y_train)

    #------ PREDICT -----
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    category_results = [category_of_risk(p) for p in y_prob]

    results_df = pd.DataFrame(X_test)
    results_df['Probability_of_Churn'] = y_prob
    results_df['Risk_Category'] = category_results
    results_df['Actual_Churn'] = y_test.values
    results_df['Predicted_Churn'] = y_pred

    #----- SAVE -----
    save_model(model, 'model.pkl')

if __name__ == "__main__":
    main()
