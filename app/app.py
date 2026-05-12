from pathlib import Path
import sys
import streamlit as st
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.preprocessing_data import preprocess
from src.model import load_model, category_of_risk

st.set_page_config(layout = "wide")
st.title("Customer Churn Prediction Dashboard")
#LOAD MODEL
my_model = load_model()
BASE_DIR = Path(__file__).resolve().parent.parent
path = BASE_DIR / "data/raw/customer_data_raw.csv"

df = preprocess(path)

X = df.drop(columns= ['Churn'])
y = df['Churn']
predictions = my_model.predict(X)
df['Prediction'] = predictions

prob_of_churn = my_model.predict_proba(X)[:,1]
df['Churn Probability'] = prob_of_churn
df['Risk_Results'] = [category_of_risk(p) for p in prob_of_churn]

col1, col2 = st.columns(2)

with col1:
    #----- PIE CHART -----
    st.subheader("Churn Risk Distribution")
    risk_counts = df['Risk_Results'].value_counts().reindex(['Low', 'Medium', 'High'])
    fig, ax = plt.subplots()
    custom_colors = ['#BEC5A4', '#8A8E75', '#D3e8D4']
    ax.pie(
        risk_counts,
        labels = risk_counts.index,
        startangle = 90,
        autopct = '%1.1f%%',
        colors = custom_colors
    )
    st.pyplot(fig)
    st.caption("Low: probability of churn was < 30%  \n Medium: probability of churn was >= 30% and < 70%  \n High: probability of churn was > 70%")
with col2:
    #-----Confusion Matrix-----
    st.subheader("Confusion Matrix")

    y_pred = my_model.predict(X)

    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    ax.set_xlabel("Predicted churns")
    ax.set_ylabel("True churns")
    st.pyplot(fig)
    st.caption(
        "There was a total of 64,374 customer records. The top left square shows true positive labels, the top right square shows false positive labels and the top left square shows false negative labels. The bottom left shows false negative labels and the bottom right shows true negative labels.")

col1, col2 = st.columns(2)
with col1:
    importance = my_model.coef_[0]
    feature_names = X.columns
    fi_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance
    }).sort_values(by='Importance', ascending=False)

    #table
    st.subheader("Coefficient Rankings")
    st.write(fi_df)
    st.caption("The coefficient rankings are shown above. The larger the absolute value of the coefficient, the more of an impact they have on customer churn.")

with col2:
    #bar graph
    st.subheader("Feature Impact Visualization")
    fig, ax = plt.subplots()
    colors = ['green' if x > 0 else 'red' for x in fi_df['Importance']]
    ax.barh(fi_df['Feature'], fi_df['Importance'], color=colors)
    ax.set_xlabel("Coefficient Value")
    st.pyplot(fig)
    st.caption("Negative values (in red) indicate lower risk of churn and positive values (in green) indicate higher risk of churn.")
