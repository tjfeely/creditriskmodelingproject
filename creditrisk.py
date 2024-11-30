import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from imblearn.over_sampling import SMOTE
import streamlit as st

# Step 1: Load and Preprocess Data
def load_data(file_path):
    """ Load the dataset """
    df = pd.read_csv(file_path)
    return df

def preprocess_data(df):
    """ Handle missing values, encode categorical variables, and normalize features """
    # Drop rows with missing target values
    df.dropna(subset=['loan_status'], inplace=True)
    
    # Fill missing values with median for numerical columns
    df.fillna(df.median(numeric_only=True), inplace=True)
    
    # Encode categorical variables
    label_encoders = {}
    for column in df.select_dtypes(include=['object']).columns:
        le = LabelEncoder()
        df[column] = le.fit_transform(df[column].astype(str))
        label_encoders[column] = le

    # Feature scaling
    scaler = StandardScaler()
    numerical_features = df.select_dtypes(include=[np.number]).columns
    df[numerical_features] = scaler.fit_transform(df[numerical_features])
    
    return df, label_encoders, scaler

def balance_data(X, y):
    """ Handle class imbalance using SMOTE """
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled

# Load and preprocess dataset
file_path = 'loan_data.csv'
df = load_data(file_path)
df, label_encoders, scaler = preprocess_data(df)

# Define features and target variable
X = df.drop('loan_status', axis=1)
y = df['loan_status']

# Balance the data
X_resampled, y_resampled = balance_data(X, y)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Step 2: Model Development
def train_logistic_regression(X_train, y_train):
    """ Train a Logistic Regression model """
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr

def train_decision_tree(X_train, y_train):
    """ Train a Decision Tree model with hyperparameter tuning """
    dt = DecisionTreeClassifier(random_state=42)
    param_grid = {'max_depth': [3, 5, 7, 10], 'min_samples_split': [2, 5, 10]}
    grid_search = GridSearchCV(dt, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

# Train models
logistic_model = train_logistic_regression(X_train, y_train)
decision_tree_model = train_decision_tree(X_train, y_train)

# Step 3: Model Evaluation
def evaluate_model(model, X_test, y_test):
    """ Evaluate model performance """
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    conf_matrix = confusion_matrix(y_test, y_pred)
    
    print(f"Accuracy: {accuracy:.2f}")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 Score: {f1:.2f}")
    print(f"ROC-AUC Score: {roc_auc:.2f}")
    print("Confusion Matrix:")
    print(conf_matrix)

print("Logistic Regression Model Evaluation")
evaluate_model(logistic_model, X_test, y_test)

print("\nDecision Tree Model Evaluation")
evaluate_model(decision_tree_model, X_test, y_test)

# Step 4: User Interface using Streamlit
def predict_risk(model, user_input, scaler, label_encoders):
    """ Predict credit risk based on user input """
    input_df = pd.DataFrame([user_input])
    
    # Encode categorical variables
    for col, le in label_encoders.items():
        input_df[col] = le.transform(input_df[col])
    
    # Feature scaling
    input_df = scaler.transform(input_df)
    
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1]
    
    return "High Risk" if prediction[0] == 1 else "Low Risk", probability

st.title("Credit Risk Assessment Tool")
st.write("Enter applicant information to assess credit risk:")

# Input fields
income = st.number_input("Applicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount", min_value=0)
credit_score = st.number_input("Credit Score", min_value=300, max_value=850)
employment_length = st.number_input("Years of Employment", min_value=0)
user_input = {
    'income': income,
    'loan_amount': loan_amount,
    'credit_score': credit_score,
    'employment_length': employment_length
}

if st.button("Predict Risk"):
    risk, prob = predict_risk(logistic_model, user_input, scaler, label_encoders)
    st.write(f"Risk Category: {risk}")
    st.write(f"Probability of Default: {prob:.2f}")