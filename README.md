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
```

---

🔎 2. Exploratory Data Analysis (EDA)
Key Findings:

Churn data is imbalanced (73% No, 27% Yes).

Month-to-month contracts show the highest churn.

Electronic check customers churn more often.

Fiber optic users churn more than DSL users.

Lower-tenure customers churn significantly more.

Lack of Tech Support/Online Security increases churn probability.


---


🛠️ 3. Preprocessing

Churn: Yes → 1, No → 0

One-Hot Encoding for categorical features

Train-test split

Scaling numerical columns (StandardScaler)

Prepared full feature matrix for ML models

---

⚖️ 4. Handling Imbalance — SMOTE

Because churners are fewer than non-churners, SMOTE was used to balance classes.

```python

from imblearn.over_sampling import SMOTE
sm = SMOTE(random_state=42)
X_train_sm, y_train_sm = sm.fit_resample(X_train, y_train)

```
This greatly improved Recall and F1-Score.

---

🤖 5. Machine Learning Models

Models trained:

- Logistic Regression

- RandomForest

- XGBoost (Selected Model)

---

📈 6. Model Evaluation (After SMOTE)
| Model               | Accuracy | Precision | Recall   | F1 Score | ROC-AUC  |
| ------------------- | -------- | --------- | -------- | -------- | -------- |
| Logistic Regression | 0.74     | 0.51      | 0.72     | 0.59     | 0.82     |
| Random Forest       | 0.76     | 0.54      | 0.63     | 0.58     | 0.81     |
| **XGBoost**         | **0.76** | **0.54**  | **0.67** | **0.60** | **0.82** |

✔ Final Model: XGBoost + SMOTE

Chosen for best balanced performance and highest recall (critical for churn problems).

---

🔥 7. Feature Importance

Top churn drivers:

- Month-to-Month Contract

- Fiber Optic Internet

- Electronic Check Payment

- No Tech Support

- No Online Security

- High Monthly Charges

- Low Tenure


💾 8. Saving the Model

Artifacts saved:

churn_xgb_model.pkl

scaler.pkl

feature_columns.pkl

```python
pickle.dump(model, open("churn_xgb_model.pkl","wb"))
pickle.dump(scaler, open("scaler.pkl","wb"))
pickle.dump(feature_columns, open("feature_columns.pkl","wb"))
```
These files are used in the Streamlit app for real-time predictions.

---

🌐 9. Streamlit Web App
Features:

Interactive customer input → predicts churn probability

Upload CSV → batch predictions

Download results

Clean UI with all customer fields

```bash
conda activate base
cd D:\ML-DL\Pyhton_notebooks
streamlit run app.py
```

---

📊 10. Power BI Dashboard

A full business dashboard built with Power BI includes:

Total Customers, Churn Rate, Avg Churn Probability

Churn by Contract Type

Churn by Internet Service

Churn by Payment Method

Tenure vs Churn

Customer Risk Segmentation

Service Impact Matrix

High-value customer analysis

---

```📁 Project Structure
📦 Telco-Churn-Prediction
 ┣ 📜 Telco_Churn_Prediction_End-to-End.ipynb
 ┣ 📜 churn_xgb_model.pkl
 ┣ 📜 scaler.pkl
 ┣ 📜 feature_columns.pkl
 ┣ 📜 churn_predictions_final.csv
 ┣ 📜 app.py
 ┣ 📜 Telco_Churn_PowerBI_Dashboard.pbix
 ┣ 📁 images
 ┃ ┣ churn_dashboard.png
 ┃ ┣ streamlit_app.png
 ┃ ┣ eda.png
 ┃ ┗ feature_importance.png
 ┗ 📜 README.md
```
---

🏁 Final Results

This project successfully delivers:

A full ML pipeline

Interactive web app

Business insights dashboard

High recall churn prediction model

Deployment-ready code

Clean documentation

---

📧 Contact

For project feedback or collaboration, feel free to reach out!

```contact
7981832369
srinuuppala410@gmail.com
```


