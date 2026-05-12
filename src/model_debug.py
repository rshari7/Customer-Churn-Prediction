import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

#----- LOAD DATA -----
df  = pd.read_csv('../data/raw/customer_data_raw.csv')

df = df.drop(columns=['CustomerID'])

# ----- ENCODING -----
# encoding step, string to numerical
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
df['Subscription Type'] = df['Subscription Type'].map({'Basic': 0, 'Standard': 1, 'Premium': 2})
df['Contract Length'] = df['Contract Length'].map({'Monthly': 0, 'Quarterly': 1, 'Annual': 2})

# ----- SPLITTING -----
# Proper X/y split, X = features, y = target (churn)
X = df.drop(columns=['Churn'])
y = df['Churn']

# test/train split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ----- STANDARDIZING ------
# feature scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# #train model
model = LogisticRegression()
model.fit(X_train, y_train)

#make predictions (0 = non-churn, 1 = churn)
y_pred = model.predict(X_test)

#probability of churn
prob_of_churn = model.predict_proba(X_test)[:, 1]

#categorize risk as low, med, hi
def category_of_risk(prob):
    if prob < 0.3:
        return 'Low'
    elif 0.3 <= prob < 0.7:
        return 'Medium'
    else:
        return 'High'
category_results = [category_of_risk(p) for p in prob_of_churn]


#----- EVALUATING -----
results_df = pd.DataFrame(X_test)
results_df['Probability_of_Churn'] = prob_of_churn
results_df['Risk_Category'] = category_results
results_df['Actual_Churn'] = y_test.values
results_df['Predicted_Churn'] = y_pred

print(f'Accuracy Score: ' ,accuracy_score(y_test, y_pred))
print(f'Classification Report:\n' ,classification_report(y_test, y_pred))
print(f'Confusion Matrix:\n' ,confusion_matrix(y_test, y_pred))
print(f'AUC-ROC Score: ' ,roc_auc_score(y_test, y_pred))

#----- SAVING MODEL -----
#save predictions to outputs folder
# results_df.to_csv("../outputs/customer_churn_predictions.csv", index=False)

