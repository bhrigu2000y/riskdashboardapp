import streamlit as st
# ✅ Must be the first Streamlit command
st.set_page_config(page_title="Project Risk Assessment Dashboard", layout="wide")

import pandas as pd
import numpy as np
import plotly.express as px
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import os

# ----------------------------
# Utility Functions
# ----------------------------
@st.cache_data
def load_sample_data():
    # Sample data for fallback
    data = pd.DataFrame({
        'Task Name': ['Task A', 'Task B'],
        'Phase': ['Planning', 'Execution'],
        'Start Date': ['2023-01-01', '2023-01-10'],
        'End Date': ['2023-01-05', '2023-01-20'],
        'Assigned To': ['Alice', 'Bob'],
        'Complexity': ['High', 'Low'],
        'Dependencies': [2, 0],
        '% Complete': [90, 50],
        'Was Delayed': ['N', 'Y']
    })
    return data

def preprocess_data(df):
    df['Duration'] = (pd.to_datetime(df['End Date']) - pd.to_datetime(df['Start Date'])).dt.days
    df['Complexity'] = LabelEncoder().fit_transform(df['Complexity'])
    df['Dependencies'] = df['Dependencies'].fillna(0).astype(int)
    df['% Complete'] = df['% Complete'].fillna(0)
    return df[['Duration', 'Complexity', 'Dependencies', '% Complete']]

def train_model(df):
    df = df.copy()
    df['Duration'] = (pd.to_datetime(df['End Date']) - pd.to_datetime(df['Start Date'])).dt.days
    df['Complexity'] = LabelEncoder().fit_transform(df['Complexity'])
    df['Dependencies'] = df['Dependencies'].fillna(0).astype(int)
    df['% Complete'] = df['% Complete'].fillna(0)
    df['Was Delayed'] = df['Was Delayed'].map({'Y': 1, 'N': 0})

    X = df[['Duration', 'Complexity', 'Dependencies', '% Complete']]
    y = df['Was Delayed']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

# ----------------------------
# Load or Train Model
# ----------------------------
model_path = "model/model.pkl"
os.makedirs("model", exist_ok=True)
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    sample_data = load_sample_data()
    model = train_model(sample_data)
    joblib.dump(model, model_path)

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Project Risk Assessment Dashboard", layout="wide")
st.title("📈 Project Risk Assessment Dashboard")

uploaded_file = st.file_uploader("Upload your project CSV file", type=["csv"])
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.subheader("📋 Uploaded Data Preview")
    st.dataframe(df.head())

    try:
        features = preprocess_data(df)
        predictions = model.predict(features)
        df['Risk Level'] = np.where(predictions == 1, 'High', 'Low')

        st.subheader("🚦 Task Risk Prediction")
        st.dataframe(df[['Task Name', 'Phase', 'Risk Level']])

        st.subheader("📊 Risk by Project Phase")
        fig = px.histogram(df, x="Phase", color="Risk Level", barmode="group",
                           title="Risk Distribution by Phase")
        st.plotly_chart(fig, use_container_width=True)
    except Exception as e:
        st.error(f"Prediction failed: {e}")
else:
    st.info("Please upload a project task CSV file to begin.")

st.markdown("---")
st.markdown("Built with ❤️ by Bhriguraj | Powered by Streamlit & ML")
