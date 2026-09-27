# Customer Retention Intelligence — Telco Churn (Kaggle) Edition

Built on the real **Telco Customer Churn** dataset (7,043 customers, IBM sample
data via Kaggle: https://www.kaggle.com/datasets/blastchar/telco-customer-churn).

## What's in this folder
- `telco_churn.csv` — the dataset you uploaded
- `train_model.py` — trains a Logistic Regression pipeline (one-hot encoding +
  scaling + model, all in one saved object) → **75% accuracy, 0.85 ROC-AUC** on
  held-out data
- `app.py` — the Streamlit app
- `model_pipeline.pkl`, `model_meta.json` — pre-trained, ready to run
- `requirements.txt`

## Run locally
```bash
pip install -r requirements.txt
python train_model.py      # only needed again if you change the dataset
streamlit run app.py
```

## Deploy on Streamlit Community Cloud (free)
1. Create a **public** GitHub repo and push all the files in this folder
   (including `telco_churn.csv`, `model_pipeline.pkl`, `model_meta.json` —
   commit the trained files so the app doesn't need to retrain on deploy)
2. Go to https://share.streamlit.io → sign in with GitHub → **New app**
3. Select your repo, branch `main`, main file `app.py` → **Deploy**
4. You'll get a public URL (e.g. `https://your-app.streamlit.app`) — that's
   the "app url link" your assignment asks you to submit

## For your report
- **Dataset citation:** Telco Customer Churn, IBM Sample Data, distributed via
  Kaggle (blastchar/telco-customer-churn). Refer to the company as "ABC Ltd."
  per your assignment instructions if asked to anonymize it.
- **Why Logistic Regression:** Churn is binary (Yes/No), so logistic
  regression is the appropriate model; linear regression fits continuous
  targets (e.g. sales, demand) instead.
- **Preprocessing:** `TotalCharges` had 11 blank values (all brand-new,
  0-tenure customers) — filled as `tenure × MonthlyCharges`. Categorical
  fields (contract type, payment method, services subscribed, etc.) were
  one-hot encoded; numeric fields were standardized. Worth a line in your
  methodology section.
- **Explainability:** the app's "why" section multiplies each transformed
  input by its logistic regression coefficient — a transparent, non-black-box
  explanation method, which ties directly into the "does explainability
  affect trust in AI" question in your managerial study.
- **Class balance:** the model uses `class_weight="balanced"` since only
  26.5% of customers churned — otherwise it would just predict "No churn"
  for everyone and still look falsely accurate. Good talking point on model
  choices in your write-up.
