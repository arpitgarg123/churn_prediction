# app.py
# Streamlit UI for Telco Churn model with robust categorical -> numeric conversion
# Place model.pkl in same folder.
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(page_title="Telco Churn Predictor", layout="centered")
st.title("Telco Churn Predictor")

MODEL_PATH = Path("model.pkl")

@st.cache_resource
def load_artifact(path):
    if not path.exists():
        return None, "Model file not found. Put model.pkl in the app folder."
    art = joblib.load(path)
    # If user saved a dict/artifact with model+preproc
    if isinstance(art, dict):
        model = art.get("model") or art.get("estimator") or art.get("pipeline")
        scaler = art.get("scaler")
        encoders = art.get("label_encoders", {}) or art.get("encoders", {})
        feature_columns = art.get("feature_columns")
        numeric_cols = art.get("numeric_cols") or art.get("num_cols") or []
        return {"model": model, "scaler": scaler, "encoders": encoders,
                "feature_columns": feature_columns, "numeric_cols": list(numeric_cols)}, None
    else:
        # plain model / pipeline
        return {"model": art, "scaler": None, "encoders": {}, "feature_columns": None, "numeric_cols": []}, None

artifact, err = load_artifact(MODEL_PATH)
if err:
    st.error(err)
    st.stop()

model = artifact["model"]
scaler = artifact["scaler"]
encoders = artifact["encoders"] or {}
feature_columns = artifact["feature_columns"]
numeric_cols = artifact["numeric_cols"] or []

# Fallback list of features (user had requested some removed earlier)
default_features = [
    "Partner","Dependents",
    "InternetService","OnlineSecurity","OnlineBackup",
    "DeviceProtection","TechSupport","StreamingTV","StreamingMovies",
    "Contract","PaperlessBilling","PaymentMethod"
]

cols_to_use = feature_columns if feature_columns is not None else default_features

st.markdown("Fill customer information below and click **Predict**. The app will convert common categorical values (e.g. Yes/No) to numbers before predicting.")

# Build form inputs
with st.form("input_form"):
    inputs = {}
    for col in cols_to_use:
        lower = col.lower()
        # simple widgets
        if lower in ["partner", "dependents", "paperlessbilling"]:
            inputs[col] = st.selectbox(col, ["Yes", "No"])
        elif lower in ["onlinesecurity","onlinebackup","deviceprotection","techsupport",
                       "streamingtv","streamingmovies"]:
            inputs[col] = st.selectbox(col, ["No", "Yes", "No internet service"])
        elif lower == "internetservice":
            inputs[col] = st.selectbox(col, ["DSL", "Fiber optic", "No"])
        elif lower == "contract":
            inputs[col] = st.selectbox(col, ["Month-to-month", "One year", "Two year"])
        elif lower == "paymentmethod":
            inputs[col] = st.selectbox(col, ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        else:
            inputs[col] = st.text_input(col, value="")
    submitted = st.form_submit_button("Predict")

if submitted:
    # Build DataFrame
    row = pd.DataFrame([inputs], columns=cols_to_use)

    # 1) Try applying saved encoders (LabelEncoder-like) where available
    for c, enc in encoders.items():
        if c in row.columns:
            try:
                # enc.transform expects array-like
                row[c] = enc.transform([row.at[0, c]])[0]
            except Exception:
                # ignore here; fallback mapping below will try
                pass

    # 2) Fallback mappings for common categorical values and coercion to numeric
    # Mapping helper
    def map_common_string_to_number(val):
        if val is None:
            return val
        if isinstance(val, (int, float, np.integer, np.floating)):
            return val
        s = str(val).strip().lower()
        # common yes/no
        if s in ("yes", "y", "true", "t"):
            return 1
        if s in ("no", "n", "false", "f"):
            return 0
        # specific telco strings -> map to 0 (treat as "no" / missing service)
        if s in ("no internet service", "no phone service", "no_internet", "no_phone"):
            return 0
        # keep original string for other categories (we might map later)
        return val

    # Apply mapping to all columns
    for c in row.columns:
        try:
            row.at[0, c] = map_common_string_to_number(row.at[0, c])
        except Exception:
            pass

    # 3) If after mapping some columns are still strings, try to coerce numeric; otherwise
    #    attempt to map categories using encoders if present (again), or raise helpful error.
    still_strings = []
    for c in row.columns:
        val = row.at[0, c]
        # If it's a pandas scalar type that's numeric already, skip
        if isinstance(val, (int, float, np.integer, np.floating)):
            continue
        # Try numeric conversion
        try:
            newv = pd.to_numeric(val)
            row.at[0, c] = newv
            continue
        except Exception:
            pass
        # If we have an encoder for this column, try again
        if c in encoders:
            enc = encoders[c]
            try:
                row.at[0, c] = enc.transform([val])[0]
                continue
            except Exception:
                pass
        # If we reach here, this column still holds a string
        still_strings.append(c)

    # If any columns remain string-like, show a helpful error and list them
    if still_strings:
        st.error(
            "Some inputs are still non-numeric and the model cannot accept them. "
            "Please either provide numeric values or save encoders during training so the app can map categories. "
            "Columns causing issue: " + ", ".join(still_strings)
        )
        with st.expander("Problematic values (show current row)"):
            st.write(row.T)
        st.stop()

    # 4) Ensure column ordering and presence of feature_columns if provided
    if feature_columns is not None:
        for c in feature_columns:
            if c not in row.columns:
                row[c] = 0
        row = row[feature_columns]
    else:
        # ensure consistent column order
        row = row[cols_to_use]

    # 5) Scale numeric columns if scaler available
    if scaler is not None and len(numeric_cols) > 0:
        # only scale columns that exist in row and in numeric_cols
        cols_to_scale = [c for c in numeric_cols if c in row.columns]
        if cols_to_scale:
            try:
                row[cols_to_scale] = scaler.transform(row[cols_to_scale])
            except Exception:
                # fallback: convert to DataFrame for shape compatibility
                row[cols_to_scale] = pd.DataFrame(scaler.transform(row[cols_to_scale]), columns=cols_to_scale)

    # 6) Finally predict
    try:
        pred = model.predict(row)[0]
    except Exception as e:
        st.error(f"Model prediction failed after conversions: {e}")
        with st.expander("Show input used for prediction"):
            st.write(row.T)
        st.stop()

    prob = None
    if hasattr(model, "predict_proba"):
        try:
            prob = float(model.predict_proba(row)[0][1])
        except Exception:
            prob = None

    label = "Churn (Yes)" if int(pred) == 1 else "No Churn (No)"
    st.subheader("Prediction")
    st.write("Result:", label)
    if prob is not None:
        st.write(f"Probability of churn: {prob:.3f}")

    with st.expander("Show input used for prediction (after conversions)"):
        st.dataframe(row.T)
