import streamlit as st
import pandas as pd
import plotly.express as px

# Load dataset
file_path = "diabetes.csv"  # Ensure this file is present when deploying
df = pd.read_csv(file_path)

# Streamlit App Title
st.title("Diabetes EDA Dashboard")

# Sidebar Dropdown for Feature Selection
feature = st.sidebar.selectbox(
    "Select Feature for Analysis:",
    ["Glucose", "BMI", "BloodPressure", "Insulin", "DiabetesPedigreeFunction", "Pregnancies", "Age"]
)

# Histogram for Feature Distribution
st.subheader(f"Distribution of {feature} by Diabetes Outcome")
fig_hist = px.histogram(df, x=feature, color='Outcome', barmode='overlay',
                        title=f'{feature} Distribution by Outcome')
st.plotly_chart(fig_hist)

# Correlation Heatmap
st.subheader("Feature Correlation Heatmap")
fig_heatmap = px.imshow(df.corr(), text_auto=True, title="Feature Correlation Heatmap")
st.plotly_chart(fig_heatmap)

# Scatter Plot: Feature vs Outcome
st.subheader(f"{feature} vs Diabetes Outcome")
fig_scatter = px.scatter(df, x=feature, y='Outcome', color='Outcome',
                          title=f'{feature} vs Diabetes Outcome',
                          trendline='ols')
st.plotly_chart(fig_scatter)

# Instructions for Running Locally
st.sidebar.markdown("### How to Run Locally")
st.sidebar.code("streamlit run diabetes_dashboard.py")
