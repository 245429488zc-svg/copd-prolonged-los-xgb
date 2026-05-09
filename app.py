import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import streamlit.components.v1 as components

# =========================
# Load model and preprocessing files
# =========================
model = joblib.load("xgb_model.pkl")
features = pd.read_csv("feature_order.csv")["feature"].tolist()
scale_params = pd.read_csv("scale_params.csv")
caps = pd.read_csv("caps_final.csv")

# =========================
# Page title
# =========================
st.title("🩺 Prolonged Hospitalization Risk Calculator")
st.caption("XGBoost-based model for predicting LOHS ≥14 days in hospitalized AECOPD patients")

# =========================
# Input form
# =========================
st.header("📋 Enter Patient Data")

user_input = {}

user_input["CRP"] = st.number_input("C-Reactive Protein, CRP (mg/L)", min_value=0.0, step=0.1)
user_input["RDW"] = st.number_input("Red Cell Distribution Width, RDW (%)", min_value=0.0, step=0.1)
user_input["NLR"] = st.number_input("Neutrophil-to-Lymphocyte Ratio, NLR", min_value=0.0, step=0.1)
user_input["PCO2"] = st.number_input("PaCO₂ (mmHg)", min_value=0.0, step=1.0)
user_input["PLR"] = st.number_input("Platelet-to-Lymphocyte Ratio, PLR", min_value=0.0, step=0.1)
user_input["BNP"] = st.number_input("proBNP", min_value=0.0, step=10.0)
user_input["ALB"] = st.number_input("Albumin (g/L)", min_value=0.0, step=0.1)
user_input["MLR"] = st.number_input("Monocyte-to-Lymphocyte Ratio, MLR", min_value=0.0, step=0.01)
user_input["WBC"] = st.number_input("White Blood Cell Count, WBC (×10⁹/L)", min_value=0.0, step=0.1)

input_df = pd.DataFrame([user_input])[features]

# =========================
# Preprocessing
# =========================
def preprocess_input(df):
    df = df.copy()

    # Apply capping
    for _, row in caps.iterrows():
        var = row["var"]

        if var not in df.columns:
            continue

        q_low = row["q_low"]
        q_high = row["q_high"]

        if np.isfinite(q_low):
            df[var] = np.where(df[var] < q_low, q_low, df[var])

        if np.isfinite(q_high):
            df[var] = np.where(df[var] > q_high, q_high, df[var])

    # Apply Z-score normalization
    for _, row in scale_params.iterrows():
        var = row["feature"]

        if var not in df.columns:
            continue

        mean = row["mean"]
        sd = row["sd"]

        if sd != 0:
            df[var] = (df[var] - mean) / sd

    return df[features]

# =========================
# Prediction
# =========================
if st.button("🔍 Predict Risk"):

    input_processed = preprocess_input(input_df)

    prob = model.predict_proba(input_processed)[0, 1]
    result = "High Risk (LOHS ≥14 days)" if prob >= 0.5 else "Low Risk (LOHS <14 days)"

    st.subheader("📊 Prediction Result")
    st.write(f"**Predicted Probability:** `{prob:.3f}`")
    st.write(f"**Classification:** `{result}`")

    # =========================
    # Model-level plots
    # =========================
    st.subheader("📌 SHAP Feature Importance")
    st.image("shap_bar_xgb.png", width=800)

    st.subheader("🌀 SHAP Summary Plot")
    st.image("shap_beeswarm_xgb.png", width=800)

    # =========================
    # Individual-level SHAP
    # =========================
    st.subheader("🧠 SHAP Force Plot for This Patient")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(input_processed)

    expected_value = explainer.expected_value

    force_plot = shap.force_plot(
        expected_value,
        shap_values[0],
        input_processed.iloc[0],
        matplotlib=False
    )

    shap.save_html("force_plot.html", force_plot)
    components.html(open("force_plot.html", "r", encoding="utf-8").read(), height=350)

    with st.expander("Processed input used by the model"):
        st.dataframe(input_processed)

# =========================
# Footer
# =========================
st.markdown("---")
st.caption(
    "For research use only. External validation is required before clinical implementation."
)