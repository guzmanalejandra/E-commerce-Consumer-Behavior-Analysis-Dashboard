import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.title("E-commerce Consumer Behavior Analysis Dashboard")

st.markdown("""
This dashboard performs exploratory analysis on an e-commerce consumer behavior dataset and uses machine learning to predict **Purchase_Amount** based on selected features.
""")

# ------------------------------
# 1. Load Data
# ------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("Ecommerce_Consumer_Behavior_Analysis_Data.csv")
    return df

df = load_data()


df.rename(columns={
    'PurchaseAmount': 'Purchase_Amount', 
    'AnnualIncome': 'Annual Income (k$)', 
    'SpendingScore': 'Spending Score (1-100)'
}, inplace=True)

# Clean the Purchase_Amount column by removing '$' and any commas, then convert to float
df["Purchase_Amount"] = df["Purchase_Amount"].replace({'\$': '', ',': ''}, regex=True).str.strip().astype(float)

# Create a numeric version of Income_Level
income_mapping = {"Low": 1, "Middle": 2, "High": 3}
df["Income_Numeric"] = df["Income_Level"].map(income_mapping)

# Map income levels to a numeric scale
income_mapping = {"Low": 1, "Middle": 2, "High": 3}
df["Income_Numeric"] = df["Income_Level"].map(income_mapping)

# ------------------------------
# 2. Data Overview & Filtering
# ------------------------------
st.header("Data Overview")
st.write("Dataset Shape:", df.shape)
st.write("First 5 rows of the dataset:")
st.dataframe(df.head())
st.write("Summary statistics:")
st.write(df.describe())

# Sidebar filters for exploratory analysis
st.sidebar.header("Filters")

if "Gender" in df.columns:
    genders = df["Gender"].unique().tolist()
    selected_genders = st.sidebar.multiselect("Select Gender", genders, default=genders)
    df = df[df["Gender"].isin(selected_genders)]
    
if "Age" in df.columns:
    min_age = int(df["Age"].min())
    max_age = int(df["Age"].max())
    age_range = st.sidebar.slider("Select Age Range", min_value=min_age, max_value=max_age, value=(min_age, max_age))
    df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

# ------------------------------
# 3. Exploratory Data Analysis (EDA)
# ------------------------------
st.header("Exploratory Data Analysis")

# Distribution of Age
if "Age" in df.columns:
    age_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Age:Q", bin=alt.Bin(maxbins=30)),
        y="count()"
    ).properties(title="Age Distribution")
    st.altair_chart(age_chart, use_container_width=True)

# Distribution of Annual Income
if "Annual Income (k$)" in df.columns:
    income_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Annual Income (k$):Q", bin=alt.Bin(maxbins=30)),
        y="count()"
    ).properties(title="Annual Income Distribution")
    st.altair_chart(income_chart, use_container_width=True)

# Distribution of Spending Score
if "Spending Score (1-100)" in df.columns:
    spending_chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Spending Score (1-100):Q", bin=alt.Bin(maxbins=30)),
        y="count()"
    ).properties(title="Spending Score Distribution")
    st.altair_chart(spending_chart, use_container_width=True)

# Scatter plot: Annual Income vs. Spending Score
if "Annual Income (k$)" in df.columns and "Spending Score (1-100)" in df.columns:
    scatter_chart = alt.Chart(df).mark_circle(size=60).encode(
        x="Annual Income (k$):Q",
        y="Spending Score (1-100):Q",
        tooltip=["Annual Income (k$)", "Spending Score (1-100)"]
    ).properties(title="Annual Income vs. Spending Score")
    st.altair_chart(scatter_chart, use_container_width=True)

# ------------------------------
# 4. Machine Learning: Predict Purchase_Amount
# ------------------------------
st.header("Machine Learning: Predict Purchase_Amount")

st.markdown("""
We will train a **Random Forest Regressor** to predict **Purchase_Amount** based on the following features:
- **Age**
- **Annual Income (k$)**
- **Spending Score (1-100)**
""")
target_col = "Purchase_Amount"
# Use Income_Numeric instead of Annual Income (k$)
feature_cols = ["Age", "Income_Numeric", "Frequency_of_Purchase"]

if target_col in df.columns and all(col in df.columns for col in feature_cols):
    ml_df = df[[target_col] + feature_cols].dropna()
    X = ml_df[feature_cols]
    y = ml_df[target_col]
    
    # Split into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Train the Random Forest Regressor
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Predict and evaluate
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    st.write("### Model Performance")
    st.write(f"R² Score on Test Set: {r2:.2f}")
    
    # Scatter plot: Actual vs. Predicted Purchase_Amount
    ml_test = X_test.copy()
    ml_test["Actual Purchase_Amount"] = y_test
    ml_test["Predicted Purchase_Amount"] = y_pred
    scatter_ml = alt.Chart(ml_test.reset_index()).mark_circle(size=60).encode(
        x=alt.X("Actual Purchase_Amount:Q", title="Actual Purchase_Amount"),
        y=alt.Y("Predicted Purchase_Amount:Q", title="Predicted Purchase_Amount"),
        tooltip=["Actual Purchase_Amount", "Predicted Purchase_Amount"]
    ).properties(title="Actual vs. Predicted Purchase_Amount", width=600, height=300)
    st.altair_chart(scatter_ml, use_container_width=True)
    
    st.write("---")
    st.subheader("Predict Purchase_Amount for New Input")
    st.markdown("Adjust the parameters below to predict the Purchase_Amount for a new consumer:")
    
    new_age = st.number_input("Age", min_value=int(X["Age"].min()), max_value=int(X["Age"].max()), value=int(X["Age"].mean()))
    new_income = st.number_input("Income Level (Numeric)", min_value=float(X["Income_Numeric"].min()), max_value=float(X["Income_Numeric"].max()), value=float(X["Income_Numeric"].mean()))
    new_frequency = st.number_input("Frequency of Purchase", min_value=int(X["Frequency_of_Purchase"].min()), max_value=int(X["Frequency_of_Purchase"].max()), value=int(X["Frequency_of_Purchase"].mean()))
    
    new_features = pd.DataFrame({
        "Age": [new_age],
        "Income_Numeric": [new_income],
        "Frequency_of_Purchase": [new_frequency]
    })
    
    prediction_new = model.predict(new_features)[0]
    st.write(f"**Predicted Purchase_Amount:** ${prediction_new:,.2f}")
else:
    st.error("Required columns for ML are not present in the dataset.")
