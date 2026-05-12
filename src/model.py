from pathlib import Path
from sklearn.linear_model import LogisticRegression
import pickle

#----- TRAINING -----
def train_model(X_train, y_train):
    model = LogisticRegression()
    model.fit(X_train, y_train)
    return model


#----- PREDICTING -----
def predict(model, X):
    return model.predict(X)

#------ CALC PROBABILITY OF CHURN -----
def predict_proba(model, X):
    return model.predict_proba(X)[:,1]

#categorize risk as low, med, hi
def category_of_risk(prob):
    if prob < 0.3:
        return 'Low'
    elif 0.3 <= prob < 0.7:
        return 'Medium'
    else:
        return 'High'

#----- SAVE AND LOAD MODEL -----
BASE_DIR = Path(__file__).resolve().parent.parent

def save_model(model, filename = "model.pkl"):
    path = BASE_DIR / filename
    with open(path, 'wb') as file:
        pickle.dump(model, file)
def load_model(filename = "model.pkl"):
    path = BASE_DIR / filename
    with open(path, 'rb') as file:
        return pickle.load(file)
