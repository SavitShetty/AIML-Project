import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
import pickle

# Load dataset
data = pd.read_csv("diabetes.csv")

# Features and target
X = data.drop("Outcome", axis=1)
y = data["Outcome"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluation
y_pred = model.predict(X_test)
print("Model Accuracy:", model.score(X_test, y_test))
print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))

# Save model + columns (IMPORTANT FIX)
with open("model.pkl", "wb") as f:
    pickle.dump((model, X.columns.tolist()), f)

print("\nModel saved as model.pkl")

import streamlit as st
import numpy as np
import pandas as pd
import pickle

# Load model and column names
model, columns = pickle.load(open("model.pkl", "rb"))

st.set_page_config(page_title="Health Predictor", layout="centered")

st.title("🧠 Smart Health Risk Predictor")
st.write("Enter your health details below:")

# Input fields
pregnancies = st.number_input("Pregnancies", 0, 20)
glucose = st.number_input("Glucose Level", 0, 200)
blood_pressure = st.number_input("Blood Pressure", 0, 150)
skin_thickness = st.number_input("Skin Thickness", 0, 100)
insulin = st.number_input("Insulin Level", 0, 900)
bmi = st.number_input("BMI", 0.0, 70.0)
diabetes_pedigree = st.number_input("Diabetes Pedigree Function", 0.0, 2.5)
age = st.number_input("Age", 1, 120)

# Prediction
if st.button("Predict"):
    input_data = pd.DataFrame([[pregnancies, glucose, blood_pressure,
                                skin_thickness, insulin, bmi,
                                diabetes_pedigree, age]],
                              columns=columns)

    prediction = model.predict(input_data)

    if prediction[0] == 1:
        st.error("⚠️ High Risk of Diabetes")
        st.write("👉 Suggestions:")
        st.write("- Reduce sugar intake")
        st.write("- Exercise regularly")
        st.write("- Consult a doctor")
    else:
        st.success("✅ Low Risk of Diabetes")
        st.write("👉 Keep maintaining a healthy lifestyle!")
