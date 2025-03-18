# E-commerce Consumer Behavior Analysis Dashboard

This project is a Streamlit dashboard that performs exploratory data analysis (EDA) on an e-commerce consumer behavior dataset and uses machine learning to predict the **Purchase_Amount** for new consumers. The dashboard includes data filtering, visualizations, and a Random Forest Regression model.


## Overview

The dashboard allows users to:
- Visualize data distributions for key variables like **Age** and **Frequency of Purchase**.
- Filter data interactively based on criteria such as gender and age range.
- Train a Random Forest Regressor to predict the **Purchase_Amount** using selected features.
- Input new consumer parameters to get a predicted **Purchase_Amount**.

> **Note:** The code originally expected columns like `Annual Income (k$)` and `Spending Score (1-100)`. However, the provided dataset contains columns such as `Income_Level` instead. The project includes guidance on mapping or modifying features as needed.

## Features

- **Data Loading & Cleaning:**  
  Loads the dataset from a CSV file and cleans the **Purchase_Amount** column (removing dollar signs, extra spaces, etc.) to ensure numeric values.

- **Exploratory Data Analysis (EDA):**  
  - Distribution charts for **Age** and other numerical features.
  - Scatter plots to visualize relationships between selected features.
  
- **Machine Learning:**  
  - A Random Forest Regressor is used to predict **Purchase_Amount**.
  - The dashboard splits the data into training and testing sets.
  - Model performance is evaluated using the R² score.
  - Users can input new consumer data to predict purchase amounts in real time.

- **Interactive Widgets:**  
  Utilizes Streamlit's sidebar filters and number input widgets for an interactive user experience.

![image](https://github.com/user-attachments/assets/2e212922-c501-412d-ba1a-aebedf2c2c81)


## Installation

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/yourusername/ecommerce-dashboard.git
   cd ecommerce-dashboard

   streamlit run main.py

