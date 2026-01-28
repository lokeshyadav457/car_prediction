import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Car Price Prediction System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CUSTOM CLASSES (if used in your pipeline) ----------------
# Only keep this if your model used it; else ignore
class MyCustomTransformer:
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X

# ---------------- LOAD MODEL FILES ----------------
@st.cache_resource
def load_model():
    model = joblib.load("car_price_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    return model, scaler, feature_names

model, scaler, feature_names = load_model()
# ---------------- LOAD DATA ----------------
@st.cache_data
def load_data():
    df = pd.read_excel("car data.xlsx")
    return df

df = load_data()
# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Navigation")
menu = st.sidebar.radio("Go to", ["Overview", "EDA", "Model Metrics", "Prediction"])
# ---------------- OVERVIEW ----------------
if menu == "Overview":
    st.title("🚗 Car Price Prediction System")

    st.markdown("""
    ### End-to-End Machine Learning Regression Project
    This system predicts the **selling price of a used car**
    using supervised machine learning.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head(20))

    st.subheader("📊 Summary Statistics")
    st.dataframe(df.describe())
# ---------------- EDA ----------------
elif menu == "EDA":
    st.title("📊 Exploratory Data Analysis")

    st.subheader("Selling Price Distribution")
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Selling_Price"], kde=True, ax=ax1)
    st.pyplot(fig1)

    st.subheader("Selling Price vs Year")
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x="Year", y="Selling_Price", data=df, ax=ax2)
    st.pyplot(fig2)

    st.subheader("Correlation Heatmap")
    num_df = df.select_dtypes(include=["int64", "float64"])
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

    st.subheader("📌 Key Insights")
    st.markdown("""
    - Newer cars generally have higher prices  
    - Price decreases as kilometers driven increase  
    - Fuel type and transmission significantly affect price  
    """)
# ---------------- MODEL METRICS ----------------
elif menu == "Model Metrics":
    st.title("📈 Model Performance")

    metrics = pd.read_csv("model_metrics.csv")
    st.subheader("Regression Metrics")
    st.dataframe(metrics)

    st.markdown("""
    **Metrics Explanation:**
    - **R² Score:** How well model explains price variation  
    - **RMSE:** Average prediction error  
    - **MAE:** Mean absolute difference  
    """)
# ---------------- PREDICTION ----------------
elif menu == "Prediction":
    st.title("🔮 Car Price Prediction")

    st.subheader("Enter Car Details")

    user_input = []
    for feature in feature_names:
        value = st.number_input(feature, value=0.0)
        user_input.append(value)

    if st.button("Predict Selling Price"):
        input_array = np.array(user_input).reshape(1, -1)
        input_scaled = scaler.transform(input_array)
        prediction = model.predict(input_scaled)[0]

        st.success(f"💰 Estimated Selling Price: ₹ {prediction:.2f} Lakhs")
