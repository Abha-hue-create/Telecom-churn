import json

import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Retention Intelligence", page_icon="📊", layout="centered")


@st.cache_resource
def load_artifacts():
    pipe = joblib.load("model_pipeline.pkl")
    with open("model_meta.json") as f:
        meta = json.load(f)
    return pipe, meta


try:
    pipe, meta = load_artifacts()
except FileNotFoundError:
    st.error("Model files not found. Run `python train_model.py` first.")
    st.stop()

NUM = meta["numeric_features"]
CAT = meta["categorical_features"]
CAT_OPTIONS = meta["categorical_options"]
NUM_RANGES = meta["numeric_ranges"]

FRIENDLY = {
    "gender": "Gender", "SeniorCitizen": "Senior Citizen", "Partner": "Has a Partner",
    "Dependents": "Has Dependents", "tenure": "Tenure (months)", "PhoneService": "Phone Service",
    "MultipleLines": "Multiple Lines", "InternetService": "Internet Service",
    "OnlineSecurity": "Online Security", "OnlineBackup": "Online Backup",
    "DeviceProtection": "Device Protection", "TechSupport": "Tech Support",
    "StreamingTV": "Streaming TV", "StreamingMovies": "Streaming Movies",
    "Contract": "Contract Type", "PaperlessBilling": "Paperless Billing",
    "PaymentMethod": "Payment Method", "MonthlyCharges": "Monthly Charges ($)",
    "TotalCharges": "Total Charges ($)",
}

st.title("📊 BA Assignment – Group 01 | Section C")
st.subheader("Customer Retention Intelligence")
st.caption("AI-powered customer churn assessment for managerial decision-making — ABC Ltd.")

with st.expander("👥 Group Members", expanded=True):
    gm1, gm2 = st.columns(2)
    with gm1:
        st.markdown(
            "**Abha Bhatt** — P46003  \n"
            "**Kalyanbrata** — P46028  \n"
            "**Sreelekshmi S** — P46206  \n"
            "**Shubham Dikhit** — P46279"
        )
    with gm2:
        st.markdown(
            "**Smaranika Shah** — P46060  \n"
            "**Shubham Jaiswal** — ABM01027  \n"
            "**Karan Limbad** — CBF01007"
        )

with st.expander("ℹ️ About this tool", expanded=False):
    st.write(
        "This application estimates the likelihood of customer churn using patterns "
        "learned from historical customer data. The result is an additional input for "
        "managerial decision-making and should **not replace human judgment**."
    )
    st.write(
        f"Model: Logistic Regression · Test accuracy: **{meta['accuracy']:.1%}** · "
        f"ROC-AUC: **{meta['roc_auc']:.2f}** · trained on {meta['n_train']} customers, "
        f"tested on {meta['n_test']} (historical churn rate: {meta['churn_rate']:.1%})."
    )

st.divider()

st.subheader("1. Customer Profile")
c1, c2, c3 = st.columns(3)
with c1:
    gender = st.selectbox(FRIENDLY["gender"], CAT_OPTIONS["gender"])
with c2:
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])
with c3:
    tenure = st.slider(FRIENDLY["tenure"], int(NUM_RANGES["tenure"]["min"]),
                        int(NUM_RANGES["tenure"]["max"]), int(NUM_RANGES["tenure"]["median"]))

c4, c5 = st.columns(2)
with c4:
    partner = st.selectbox(FRIENDLY["Partner"], CAT_OPTIONS["Partner"])
with c5:
    dependents = st.selectbox(FRIENDLY["Dependents"], CAT_OPTIONS["Dependents"])

st.subheader("2. Services Subscribed")
c6, c7, c8 = st.columns(3)
with c6:
    phone_service = st.selectbox(FRIENDLY["PhoneService"], CAT_OPTIONS["PhoneService"])
with c7:
    multiple_lines = st.selectbox(FRIENDLY["MultipleLines"], CAT_OPTIONS["MultipleLines"])
with c8:
    internet_service = st.selectbox(FRIENDLY["InternetService"], CAT_OPTIONS["InternetService"])

c9, c10, c11 = st.columns(3)
with c9:
    online_security = st.selectbox(FRIENDLY["OnlineSecurity"], CAT_OPTIONS["OnlineSecurity"])
with c10:
    online_backup = st.selectbox(FRIENDLY["OnlineBackup"], CAT_OPTIONS["OnlineBackup"])
with c11:
    device_protection = st.selectbox(FRIENDLY["DeviceProtection"], CAT_OPTIONS["DeviceProtection"])

c12, c13, c14 = st.columns(3)
with c12:
    tech_support = st.selectbox(FRIENDLY["TechSupport"], CAT_OPTIONS["TechSupport"])
with c13:
    streaming_tv = st.selectbox(FRIENDLY["StreamingTV"], CAT_OPTIONS["StreamingTV"])
with c14:
    streaming_movies = st.selectbox(FRIENDLY["StreamingMovies"], CAT_OPTIONS["StreamingMovies"])

st.subheader("3. Account & Billing")
c15, c16 = st.columns(2)
with c15:
    contract = st.selectbox(FRIENDLY["Contract"], CAT_OPTIONS["Contract"])
with c16:
    payment_method = st.selectbox(FRIENDLY["PaymentMethod"], CAT_OPTIONS["PaymentMethod"])

c17, c18, c19 = st.columns(3)
with c17:
    paperless_billing = st.selectbox(FRIENDLY["PaperlessBilling"], CAT_OPTIONS["PaperlessBilling"])
with c18:
    monthly_charges = st.number_input(
        FRIENDLY["MonthlyCharges"], min_value=0.0,
        max_value=float(NUM_RANGES["MonthlyCharges"]["max"]),
        value=float(NUM_RANGES["MonthlyCharges"]["median"]),
    )
with c19:
    default_total = round(monthly_charges * tenure, 2)
    total_charges = st.number_input(
        FRIENDLY["TotalCharges"], min_value=0.0,
        max_value=float(NUM_RANGES["TotalCharges"]["max"]) * 1.2,
        value=default_total,
    )

st.divider()

if st.button("🔮 Predict Customer Churn", type="primary", use_container_width=True):
    row = {
        "gender": gender, "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Partner": partner, "Dependents": dependents, "tenure": tenure,
        "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security,
        "OnlineBackup": online_backup, "DeviceProtection": device_protection,
        "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract,
        "PaperlessBilling": paperless_billing, "PaymentMethod": payment_method,
        "MonthlyCharges": monthly_charges, "TotalCharges": total_charges,
    }
    X = pd.DataFrame([row])[NUM + CAT]
    proba = pipe.predict_proba(X)[0, 1]

    st.subheader("4. Churn Assessment")
    left, right = st.columns([2, 1])
    with left:
        st.metric("Estimated Churn Probability", f"{proba:.2%}")
        st.progress(min(proba, 1.0))
    with right:
        if proba < 0.3:
            st.success("🟢 LOW CHURN RISK")
        elif proba < 0.6:
            st.warning("🟡 MEDIUM CHURN RISK")
        else:
            st.error("🔴 HIGH CHURN RISK")

    st.info(
        "**Managerial interpretation:** "
        + (
            "The model does not identify this customer as a high-priority churn case. "
            "Managers can still consider other customer information before making a decision."
            if proba < 0.3
            else "The model flags this customer as a meaningful churn risk. Consider "
            "proactive retention outreach, alongside other context you have."
            if proba < 0.6
            else "The model identifies this customer as a high-priority churn case. "
            "Managers may want to prioritize this customer for retention action, "
            "while still applying their own judgment."
        )
    )

    st.subheader("5. Why did the model make this prediction?")
    st.caption(
        "Based on the logistic regression's learned coefficients for this customer's "
        "specific inputs. These are model associations, not proof of direct causation."
    )

    # Re-run the preprocessing step alone to get the transformed row,
    # then multiply each transformed value by its coefficient.
    transformed = pipe.named_steps["preprocess"].transform(X)
    transformed = transformed.toarray() if hasattr(transformed, "toarray") else transformed
    feature_names = list(pipe.named_steps["preprocess"].get_feature_names_out())
    coefs = meta["coefficients"]

    contributions = [
        (fname, coefs.get(fname, 0.0) * transformed[0, i])
        for i, fname in enumerate(feature_names)
    ]
    contributions = [c for c in contributions if abs(c[1]) > 1e-6]
    contributions.sort(key=lambda t: abs(t[1]), reverse=True)

    def pretty(fname: str) -> str:
        for raw, friendly in FRIENDLY.items():
            if fname.startswith(f"num__{raw}") or fname.startswith(f"cat__{raw}_"):
                suffix = fname.split(f"{raw}_", 1)[-1] if "cat__" in fname else ""
                return f"{friendly}" + (f" = {suffix}" if suffix and "cat__" in fname else "")
        return fname

    for fname, contrib in contributions[:6]:
        direction = "pushed churn risk **higher**" if contrib > 0 else "pushed churn risk **lower**"
        dot = "🔴" if contrib > 0 else "🟢"
        st.write(f"{dot} **{pretty(fname)}** — {direction}")

    st.subheader("6. Managerial Use")
    st.write(
        "Combine this model output with customer interactions, service history and any "
        "other context available to you — it's a decision **support** tool, not a "
        "decision-maker."
    )

    with st.expander("📄 View Customer Information Summary"):
        st.json(row)

st.divider()
st.caption("Built for BA Assignment 2 — Predictive Analytics & Managerial AI Adoption. Not a substitute for managerial judgment.")
