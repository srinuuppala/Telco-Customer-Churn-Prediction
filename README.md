# 📊 Telco Customer Churn Prediction — End-to-End Machine Learning Project

## 👤 Author: UPPALA VENKATA SATYA SRINIVAS  
## 📅 Year: 2025  
## 🧠 Skills Used: Python, Pandas, Scikit-Learn, XGBoost, SMOTE, Streamlit, Power BI

---

# 📝 Project Overview

This project builds a **complete end-to-end Customer Churn Prediction system** using the **Telco Customer Churn dataset**.  
It covers every major stage of a real-world data science pipeline:

✔ Data Cleaning  
✔ Exploratory Data Analysis (EDA)  
✔ Feature Engineering & Preprocessing  
✔ Handling Imbalanced Data (SMOTE)  
✔ Machine Learning Models (Logistic Regression, RandomForest, XGBoost)  
✔ Model Evaluation (Accuracy, Precision, Recall, F1, ROC-AUC)  
✔ Feature Importance  
✔ Model Deployment using **Streamlit**  
✔ Business Insights Dashboard using **Power BI**


---

# 📂 Dataset

Dataset: **Telco-Customer-Churn.csv**

Rows: **7,032 customers**  
Columns: **21 features**

Includes:

- Demographics (gender, SeniorCitizen, Partner, Dependents)
- Services (Internet, Streaming, TechSupport)
- Contract Information
- Payment Method
- Billing Charges
- Target Variable: **Churn (Yes/No)**

---

# 🧼 1. Data Cleaning

Cleaning steps performed:

- Converted `TotalCharges` from object → float  
- Removed ~11 rows with missing TotalCharges  
- Dropped `customerID` (not useful for ML)  
- Ensured no missing values remain  

```python
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)
