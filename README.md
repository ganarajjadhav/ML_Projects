# Project Title

Diabetes Prediction using Random Forest Classifier


# Overview

Diabetes is a chronic disease that requires early detection to prevent severe health complications.
In this project, we build a machine learning model that predicts whether a patient is diabetic based on medical attributes such as glucose level, BMI, age, and insulin.

The solution uses a Random Forest Classifier to learn patterns from historical patient data and make accurate predictions on unseen data.


# ML Concepts Used

Algorithm:

Random Forest Classifier (Ensemble Learning)

Feature Engineering:

Feature-target separation

Data scaling using StandardScaler

Evaluation Metrics:

Accuracy Score

# Tech Stack

Python

Pandas

Scikit-learn

Jupyter / Google Colab


# Results

Accuracy: 74.02%

Model Performance Insight:

The Random Forest model performs well on structured medical data

Scaling features improves model stability

Suitable as a baseline model for healthcare prediction tasks


# Load the dataset
import pandas as pd
df = pd.read_csv('/content/diabetes.csv')

# Split features and target
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)

# Predictions and evaluation
from sklearn.metrics import accuracy_score
y_pred = rf.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)




Scikit-learn

Jupyter / Google Colab
