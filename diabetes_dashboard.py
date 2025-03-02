import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import numpy as np

# Load dataset
file_path = 'diabetes.csv'
df = pd.read_csv(file_path)

# Set page title and theme
st.set_page_config(page_title='Diabetes Data Dashboard', layout='wide')
sns.set_style('whitegrid')
plt.style.use('fivethirtyeight')

# Title
st.title('🔍 Diabetes Data Analysis Dashboard')
st.markdown('A beautifully designed dashboard to explore diabetes-related patterns.')

# Sidebar for filtering
st.sidebar.header('Filters')
age_range = st.sidebar.slider('Select Age Range', int(df['Age'].min()), int(df['Age'].max()), (20, 60))
filtered_df = df[(df['Age'] >= age_range[0]) & (df['Age'] <= age_range[1])]

# Visualization 1: Diabetes Outcome Distribution
st.subheader('1️⃣ Diabetes Outcome Distribution')
outcome_counts = filtered_df['Outcome'].value_counts()
fig1 = px.pie(values=outcome_counts, names=['Non-Diabetic', 'Diabetic'], title='Proportion of Diabetic vs. Non-Diabetic')
st.plotly_chart(fig1)

# Visualization 2: Age Distribution by Outcome
st.subheader('2️⃣ Age Distribution by Diabetes Outcome')
fig2, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=filtered_df, x='Age', hue='Outcome', kde=True, bins=20, ax=ax)
ax.set_title('Age Distribution of Diabetic vs Non-Diabetic')
st.pyplot(fig2)

# Visualization 3: Correlation Heatmap
st.subheader('3️⃣ Correlation Heatmap of Features')
fig3, ax = plt.subplots(figsize=(10, 6))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig3)

# Visualization 4: Average Glucose Levels by Age Group
st.subheader('4️⃣ Average Glucose Levels by Age Group')
df['AgeGroup'] = pd.cut(df['Age'], bins=np.arange(20, 90, 10))
avg_glucose = df.groupby('AgeGroup')['Glucose'].mean()
fig4 = px.bar(avg_glucose, x=avg_glucose.index.astype(str), y=avg_glucose.values, labels={'y': 'Avg Glucose Level', 'x': 'Age Group'})
st.plotly_chart(fig4)

# Visualization 5: BMI vs. Diabetes Outcome
st.subheader('5️⃣ BMI vs Diabetes Outcome')
fig5, ax = plt.subplots(figsize=(8, 5))
sns.boxplot(x='Outcome', y='BMI', data=filtered_df, ax=ax)
ax.set_xticklabels(['Non-Diabetic', 'Diabetic'])
ax.set_title('BMI Distribution by Diabetes Outcome')
st.pyplot(fig5)

# Visualization 6: Insulin Levels Among Diabetic vs. Non-Diabetic Patients
st.subheader('6️⃣ Insulin Levels by Diabetes Outcome')
fig6, ax = plt.subplots(figsize=(8, 5))
sns.violinplot(x='Outcome', y='Insulin', data=filtered_df, ax=ax)
ax.set_xticklabels(['Non-Diabetic', 'Diabetic'])
st.pyplot(fig6)

# Visualization 7: Blood Pressure Distribution
st.subheader('7️⃣ Blood Pressure Distribution by Outcome')
fig7, ax = plt.subplots(figsize=(8, 5))
sns.histplot(data=filtered_df, x='BloodPressure', hue='Outcome', kde=True, ax=ax)
st.pyplot(fig7)

# Visualization 8: Diabetes Pedigree Function vs. Diabetes
st.subheader('8️⃣ Diabetes Pedigree Function vs. Diabetes')
fig8 = px.scatter(filtered_df, x='DiabetesPedigreeFunction', y='Outcome', color='Outcome', title='Diabetes Pedigree Influence')
st.plotly_chart(fig8)

# Visualization 9: Pregnancy Count vs Diabetes Outcome
st.subheader('9️⃣ Pregnancy Count vs Diabetes Outcome')
fig9, ax = plt.subplots(figsize=(8, 5))
sns.barplot(x='Pregnancies', y='Outcome', data=filtered_df, ax=ax)
st.pyplot(fig9)

# Visualization 10: Feature Importance (Using Correlation)
st.subheader('🔟 Feature Importance')

# Ensure only numeric columns are considered
numeric_df = df.select_dtypes(include=['number'])

# Compute feature importance
feature_importance = numeric_df.corr()['Outcome'].abs().sort_values(ascending=False).drop('Outcome')

fig10 = px.bar(feature_importance, x=feature_importance.index, y=feature_importance.values,
               title='Feature Importance for Diabetes',
               labels={'x': 'Feature', 'y': 'Correlation with Outcome'})

st.plotly_chart(fig10)


st.markdown('---')
st.markdown('**📊 Interactive Streamlit Dashboard for Diabetes Data Exploration!**')
