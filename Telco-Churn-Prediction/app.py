# app.py
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO


# -----------------------
# Helper: load artifacts (robust)
# -----------------------
@st.cache_resource
def load_artifacts():
    model_path = r"D:\ML-DL\Pyhton_notebooks\churn_xgb_model.pkl"
    scaler_path = r"D:\ML-DL\Pyhton_notebooks\scaler.pkl"
    cols_path = r"D:\ML-DL\Pyhton_notebooks\feature_columns.pkl"

    import os
    # check existence
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    if not os.path.exists(scaler_path):
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    if not os.path.exists(cols_path):
        raise FileNotFoundError(f"Feature columns file not found: {cols_path}")

    try:
        with open(model_path, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        # Re-raise with clearer message
        raise RuntimeError(
            f"Failed to load model from {model_path}. "
            f"This often happens when the model was pickled with a different Python version or different library versions. "
            f"Original error: {e}"
        )

    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(cols_path, "rb") as f:
        feature_cols = pickle.load(f)

    feature_cols = list(feature_cols)
    return model, scaler, feature_cols

# Try loading artifacts, but handle failure gracefully so FEATURE_COLS is always defined
try:
    model, scaler, FEATURE_COLS = load_artifacts()
except Exception as e:
    # Show a clear error in Streamlit and stop the app to avoid NameError later
    st.error("Could not load model artifacts. See details below.")
    st.error(str(e))
    st.info("Most common fixes:\n• Run Streamlit from the same Python environment that created the .pkl files (Anaconda env used for training).\n• Or re-save the .pkl files using the Python interpreter you will use for Streamlit.")
    st.stop()   # stops the app here so FEATURE_COLS won't be referenced later when missing

# -----------------------
# Prediction utilities
# -----------------------
def preprocess_single(input_df, feature_cols, scaler):
    """
    One-hot encode a single-row dataframe, add missing cols, reorder to feature_cols,
    scale numeric cols (in scaler.feature_names_in_ order if available).
    Returns preprocessed DataFrame ready for model.predict_proba.
    """
    # One-hot encode
    enc = pd.get_dummies(input_df)

    # Add missing feature columns
    missing_cols = set(feature_cols) - set(enc.columns)
    for c in missing_cols:
        enc[c] = 0

    # Reorder exactly as training
    enc = enc[feature_cols]

    # Numeric columns order the scaler expects
    if hasattr(scaler, "feature_names_in_"):
        expected_num_cols = [c for c in scaler.feature_names_in_]
    else:
        expected_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    # Ensure numeric cols exist in enc (if not, add zeros)
    for c in expected_num_cols:
        if c not in enc.columns:
            enc[c] = 0.0

    # Reorder numeric columns and scale them
    num_vals = enc[expected_num_cols]
    num_scaled = scaler.transform(num_vals)
    enc[expected_num_cols] = num_scaled

    return enc

def predict_probability_from_row(row_dict):
    """
    row_dict: single customer values as dict (raw string/numeric like original dataset)
    returns float probability of churn
    """
    input_df = pd.DataFrame([row_dict])
    X = preprocess_single(input_df, FEATURE_COLS, scaler)
    prob = model.predict_proba(X)[0][1]
    return float(prob)

def batch_predict(df_raw):
    """
    Accepts a DataFrame with raw telco columns (like original csv).
    Returns DataFrame with added 'churn_probability' and 'churn_pred' (0/1).
    """
    # Drop customerID if present (not used for prediction)
    if 'customerID' in df_raw.columns:
        df_raw = df_raw.drop(columns=['customerID'])

    # One-hot encode raw df (same process as single)
    enc = pd.get_dummies(df_raw)

    # Add missing feature columns
    missing_cols = set(FEATURE_COLS) - set(enc.columns)
    for c in missing_cols:
        enc[c] = 0

    enc = enc[FEATURE_COLS]

    # Ensure numeric order for scaler
    if hasattr(scaler, "feature_names_in_"):
        expected_num_cols = [c for c in scaler.feature_names_in_]
    else:
        expected_num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']

    for c in expected_num_cols:
        if c not in enc.columns:
            enc[c] = 0.0

    # scale numeric
    num_vals = enc[expected_num_cols]
    enc[expected_num_cols] = scaler.transform(num_vals)

    # predict
    probs = model.predict_proba(enc)[:, 1]
    preds = (probs >= 0.5).astype(int)

    result = df_raw.copy()
    result['churn_probability'] = probs
    result['churn_pred'] = preds
    return result

# -----------------------
# Streamlit UI
# -----------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="centered")

st.title("Telco Customer Churn — Streamlit Demo")
st.write("Predict probability of a customer churning using the trained XGBoost model (SMOTE).")

st.markdown("**Two ways to predict:** 1) Single customer (interactive form)  2) Batch via CSV upload")

# ---------- Single Customer Form ----------
st.header("1) Single customer prediction")

with st.form("single_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("tenure (months)", min_value=0, max_value=200, value=5)
        MonthlyCharges = st.number_input("MonthlyCharges", min_value=0.0, value=80.0, step=1.0)
        TotalCharges = st.number_input("TotalCharges", min_value=0.0, value=400.0, step=1.0)
        SeniorCitizen = st.selectbox("SeniorCitizen", [0,1], index=0)

        gender = st.selectbox("gender", ["Male","Female"])
        Partner = st.selectbox("Partner", ["Yes","No"])
        Dependents = st.selectbox("Dependents", ["Yes","No"])
        PhoneService = st.selectbox("PhoneService", ["Yes","No"])
        MultipleLines = st.selectbox("MultipleLines", ["No","Yes","No phone service"])

    with col2:
        InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
        OnlineSecurity = st.selectbox("OnlineSecurity", ["Yes","No","No internet service"])
        OnlineBackup = st.selectbox("OnlineBackup", ["Yes","No","No internet service"])
        DeviceProtection = st.selectbox("DeviceProtection", ["Yes","No","No internet service"])
        TechSupport = st.selectbox("TechSupport", ["Yes","No","No internet service"])
        StreamingTV = st.selectbox("StreamingTV", ["Yes","No","No internet service"])
        StreamingMovies = st.selectbox("StreamingMovies", ["Yes","No","No internet service"])
        Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
        PaperlessBilling = st.selectbox("PaperlessBilling", ["Yes","No"])
        PaymentMethod = st.selectbox("PaymentMethod", [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ])

    submitted = st.form_submit_button("Predict churn probability")

if submitted:
    input_dict = {
        "tenure": tenure,
        "MonthlyCharges": MonthlyCharges,
        "TotalCharges": TotalCharges,
        "SeniorCitizen": SeniorCitizen,
        "gender": gender,
        "Partner": Partner,
        "Dependents": Dependents,
        "PhoneService": PhoneService,
        "MultipleLines": MultipleLines,
        "InternetService": InternetService,
        "OnlineSecurity": OnlineSecurity,
        "OnlineBackup": OnlineBackup,
        "DeviceProtection": DeviceProtection,
        "TechSupport": TechSupport,
        "StreamingTV": StreamingTV,
        "StreamingMovies": StreamingMovies,
        "Contract": Contract,
        "PaperlessBilling": PaperlessBilling,
        "PaymentMethod": PaymentMethod
    }

    prob = predict_probability_from_row(input_dict)
    label = "CHURN" if prob >= 0.5 else "STAY"
    st.metric(label="Churn Probability", value=f"{prob:.3f}", delta=None)
    st.success(f"Model suggests: **{label}** (threshold 0.5)")

# ---------- Batch CSV upload ----------
st.header("2) Batch prediction (CSV upload)")
st.write("Upload a CSV with raw telco columns (same column names as original dataset). You can also use the sample dataset below.")

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

# small helpful link: default sample dataset path (local)
st.info("Tip: sample dataset path (local): `/mnt/data/WA_Fn-UseC_-Telco-Customer-Churn.csv`")

if uploaded_file is not None:
    try:
        df_upload = pd.read_csv(uploaded_file)
    except Exception as e:
        st.error(f"Could not read uploaded file: {e}")
        st.stop()

    st.write("Preview of uploaded data:")
    st.dataframe(df_upload.head())

    if st.button("Run batch predictions"):
        with st.spinner("Predicting..."):
            results = batch_predict(df_upload)
        st.success("Predictions complete — preview:")
        st.dataframe(results.head())

        # download link
        towrite = BytesIO()
        results.to_csv(towrite, index=False)
        towrite.seek(0)
        st.download_button(label="Download predictions CSV",
                           data=towrite,
                           file_name="churn_predictions.csv",
                           mime="text/csv")

# Option to load sample dataset from local path (useful in your environment)
st.header("Example: Load sample dataset from disk")
sample_path = r"D:\ML-DL\Pyhton_notebooks\WA_Fn_Use_Telco_Customer_Churn.csv"  # update filename if needed

st.write(f"Sample dataset path: `{sample_path}`")
if st.button("Load example sample CSV from disk"):
    try:
        sample_df = pd.read_csv(sample_path)
        st.write("Sample dataset loaded — preview:")
        st.dataframe(sample_df.head())
        if st.button("Predict on sample (first 200 rows)"):
            with st.spinner("Predicting..."):
                sample_res = batch_predict(sample_df.head(200))
            st.success("Predictions complete — preview:")
            st.dataframe(sample_res.head())
            # allow download
            buff = BytesIO()
            sample_res.to_csv(buff, index=False)
            buff.seek(0)
            st.download_button("Download sample predictions", data=buff, file_name="sample_preds.csv", mime="text/csv")
    except Exception as e:
        st.error(f"Unable to load sample dataset: {e}")

