CUSTOMER CHURN PREDICTION

INTRO: This project uses ML to predict customer churn. The model is trained using logistic regression with customer records from https://www.kaggle.com/datasets/muhammadshahidazeem/customer-churn-dataset?resource=download. This model will accept data, preprocess it, and then use it to create a Streamlit dashboard that displays a churn risk distribution chart, a confusion matrix, coefficient rankings, and a feature impact visualization. Businesses can benefit from this application by developing retention strategies to keep customers and decreasing the company costs associated with new customer acquisition.

HOW TO USE: 

STEP 1: Clone the repo in terminal (command: git clone https://github.com/rshari7/Customer-Churn-Prediction.git) and install dependencies from requirements.txt (command: pip install -r requirements.txt)

STEP 2: Insert the desired raw customer data CSV file into data/raw folder; it must contain all of the following columns: CustomerID, Age, Gender, Tenure, Usage Frequency, Support Calls, Payment Delay, Subscription Type, Contract Length, Total Spend, Last Interaction, and Churn

STEP 3: Update variable 'path' to the new CSV file path in app/app.py

STEP 4: Run src/preprocessing_data, src/model, and src/train files

STEP 5: Launch Streamlit using commmand: Streamlit run app/app.py

