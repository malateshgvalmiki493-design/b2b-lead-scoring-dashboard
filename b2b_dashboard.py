# b2b_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

st.set_page_config(page_title="B2B SaaS Buying Behaviour Dashboard", layout="wide")

st.title("📊 B2B SaaS — AI Buying Behaviour Dashboard")
st.markdown("Reads `D:/b2boutput.csv`, computes composite scores, shows stage classification, and explains weight contributions.")

# --------- Load CSV (adjust path if needed) ---------
CSV_PATH = "b2boutput.csv"

@st.cache_data
def load_data(path):
    try:
        df = pd.read_excel(path)
        return df
    except FileNotFoundError:
        st.error(f"File not found at {path}. Please check the path and restart the app.")
        return None
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        return None

df = load_data(CSV_PATH)
if df is None:
    st.stop()

st.success(f"Loaded {len(df)} rows from {CSV_PATH}")

# Quick preview
with st.expander("Preview dataset (first 10 rows)"):
    st.dataframe(df.head(10))

# ---------- Sidebar: Weight sliders ----------
st.sidebar.header("Scoring weights (adjust & explain)")
st.sidebar.write("Weights will be normalized automatically if they do not sum to 100%.")

w_usage = st.sidebar.slider("Product usage / Trial depth (USG) %", 0, 40, 22)
w_intent = st.sidebar.slider("Search & intent (INT) %", 0, 40, 18)
w_stake = st.sidebar.slider("Stakeholder engagement (STK) %", 0, 25, 12)
w_email = st.sidebar.slider("Email engagement (EMG) %", 0, 20, 10)
w_past = st.sidebar.slider("Past customer (PAST) %", 0, 20, 8)
w_size = st.sidebar.slider("Company size fit (CSZ) %", 0, 20, 8)
w_recency = st.sidebar.slider("Recency (REC) %", 0, 20, 10)
w_role = st.sidebar.slider("Role alignment (RAL) %", 0, 20, 7)
w_demo = st.sidebar.slider("Demo attended (DEM) %", 0, 20, 5)

weights = np.array([w_usage, w_intent, w_stake, w_email, w_past, w_size, w_recency, w_role, w_demo], dtype=float)
st.sidebar.markdown(f"**Raw total = {weights.sum():.0f}%**")
if weights.sum() != 100:
    st.sidebar.info("Weights will be normalized internally so their relative importance remains.")

# ---------- Normalize numeric features ----------
# Which columns are required (must match your CSV column names)
required_cols = [
    'usage_actions', 'trial_depth_pct', 'intent_score', 'email_engagement',
    'num_stakeholders', 'past_customer', 'company_size', 'days_since_contact', 'role_alignment', 'demo_attended'
]

missing = [c for c in required_cols if c not in df.columns]
if missing:
    st.error(f"The CSV is missing these required columns: {missing}")
    st.stop()

# Make normalized copy
df_norm = df.copy()

# Normalize continuous features to 0-1
cont_cols = ['usage_actions','trial_depth_pct','intent_score','email_engagement','days_since_contact','role_alignment']
for c in cont_cols:
    minv = df_norm[c].min()
    maxv = df_norm[c].max()
    if maxv == minv:
        df_norm[c + "_n"] = 0.0
    else:
        df_norm[c + "_n"] = (df_norm[c] - minv) / (maxv - minv)

# company size fit mapping
size_map = {'Small':0.0, 'Medium':0.5, 'Large':1.0}
df_norm['CSZ_n'] = df_norm['company_size'].map(size_map).fillna(0.0)

# other normalized variables
df_norm['USG_n'] = df_norm['trial_depth_pct_n'] = df_norm['trial_depth_pct_n'] = df_norm['trial_depth_pct_n'] if 'trial_depth_pct_n' in df_norm else df_norm['trial_depth_pct'] 
# but we'll use the normalized ones we just made
# Define normalized columns explicitly:
df_norm['USG'] = df_norm['trial_depth_pct_n'] if 'trial_depth_pct_n' in df_norm else df_norm['trial_depth_pct']
df_norm['INT'] = df_norm['intent_score_n']
df_norm['STK'] = df_norm['num_stakeholders'] / df_norm['num_stakeholders'].max()
df_norm['EMG'] = df_norm['email_engagement_n']
df_norm['PAST'] = df_norm['past_customer'].astype(int)
df_norm['CSZ'] = df_norm['CSZ_n']
df_norm['REC'] = 1.0 - df_norm['days_since_contact_n']  # invert so higher = more recent
df_norm['RAL'] = df_norm['role_alignment_n']
df_norm['DEM'] = df_norm['demo_attended'].astype(int)

feature_matrix = df_norm[['USG','INT','STK','EMG','PAST','CSZ','REC','RAL','DEM']].values

# Normalize weight vector to sum to 1
if weights.sum() == 0:
    norm_w = np.ones_like(weights) / len(weights)
else:
    norm_w = weights / weights.sum()

# Composite score 0-100
df['composite_score'] = (feature_matrix * norm_w).sum(axis=1) * 100

# Stage thresholds (adjustable)
st.sidebar.header("Stage thresholds")
t_eval = st.sidebar.slider("Evaluation threshold (>=)", 0, 100, 25)
t_trial = st.sidebar.slider("Trial threshold (>=)", 0, 100, 50)
t_neg = st.sidebar.slider("Negotiation threshold (>=)", 0, 100, 70)
t_pv = st.sidebar.slider("Purchase-imminent threshold (>=)", 0, 100, 90)

def score_to_stage(s):
    if s >= t_pv:
        return "Purchase-imminent"
    if s >= t_neg:
        return "Negotiation"
    if s >= t_trial:
        return "Trial"
    if s >= t_eval:
        return "Evaluation"
    return "Awareness"

df['pred_stage_by_score'] = df['composite_score'].apply(score_to_stage)

# --------- Show overall metrics and charts ----------
st.header("Summary Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", len(df))
# Quick model later will show AUC; for now show average composite score
col2.metric("Avg composite score", f"{df['composite_score'].mean():.1f}%")
col3.metric("Top stage share", df['pred_stage_by_score'].value_counts().idxmax())

st.subheader("Composite score distribution")
fig, ax = plt.subplots()
ax.hist(df['composite_score'], bins=30)
ax.set_xlabel("Composite score (%)")
ax.set_ylabel("Count")
st.pyplot(fig)

st.subheader("Stage counts")
st.bar_chart(df['pred_stage_by_score'].value_counts())

# ---------- Per-lead interactive inspector ----------
st.header("Per-lead analyzer (explainability)")
lead_index = st.number_input("Enter lead row index (0 to {})".format(len(df)-1), min_value=0, max_value=len(df)-1, value=0)
lead = df.iloc[int(lead_index)]
lead_norm = df_norm.iloc[int(lead_index)]

st.markdown("**Lead snapshot**")
st.json(lead.to_dict())

# compute contributions
contribs = (norm_w * feature_matrix[int(lead_index)]) * 100
contrib_df = pd.DataFrame({
    'criterion': ['USG','INT','STK','EMG','PAST','CSZ','REC','RAL','DEM'],
    'weight_pct': list(norm_w*100),
    'norm_value_0_1': feature_matrix[int(lead_index)],
    'contribution_pct_points': contribs
})
st.subheader("Score breakdown (per criterion)")
st.table(contrib_df.style.format({"weight_pct":"{:.1f}", "norm_value_0_1":"{:.3f}", "contribution_pct_points":"{:.2f}"}))

st.markdown(f"**Composite score (sum of contributions)**: {df.loc[int(lead_index),'composite_score']:.2f}%")
st.markdown(f"**Stage from composite score**: {df.loc[int(lead_index),'pred_stage_by_score']}")

# ---------- Quick ML model to predict purchase probability ----------
st.header("Model (Logistic Regression) — Purchase probability")
X = df_norm[['USG','INT','EMG','REC','RAL','DEM','PAST']]
y = df['purchased'] if 'purchased' in df.columns else None

if y is None:
    st.info("No 'purchased' column found in CSV — ML model can't be trained. If you have historical ground truth, include a 'purchased' (0/1) column.")
else:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)
    proba = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, proba)
    st.metric("Logistic Regression AUC", f"{auc:.3f}")

    # show predicted probability for selected lead
    lead_feat = X.iloc[int(lead_index)].values.reshape(1, -1)
    lead_prob = model.predict_proba(lead_feat)[0,1]
    st.write(f"Predicted purchase probability (selected lead): **{lead_prob*100:.1f}%**")

# ---------- Action recommendations ----------
st.header("Actionable recommendations (automated)")
# Identify top 2 contributing criteria for the selected lead
top_idx = np.argsort(contribs)[-2:][::-1]
recommendations = []
for idx in top_idx:
    c = contrib_df.loc[idx, 'criterion']
    if c == 'USG':
        recommendations.append("Nudge with onboarding content & in-app tips for key features.")
    elif c == 'INT':
        recommendations.append("Send pricing & ROI info; offer tailored demo calls.")
    elif c == 'STK':
        recommendations.append("Map stakeholders; schedule multi-stakeholder workshop.")
    elif c == 'EMG':
        recommendations.append("Use a higher-touch outreach: call + personalized email.")
    elif c == 'PAST':
        recommendations.append("Offer loyalty discount / easy contract renewal path.")
    elif c == 'CSZ':
        recommendations.append("Recommend appropriate pricing tier & seat-based demo.")
    elif c == 'REC':
        recommendations.append("Re-engage quickly with a time-limited offer.")
    elif c == 'RAL':
        recommendations.append("Send role-specific collateral (security for CTO, ROI for CFO).")
    elif c == 'DEM':
        recommendations.append("Prepare negotiation kit & contract template; escalate to AE.")
st.write("- " + "\n- ".join(recommendations))

st.markdown("---")
st.markdown("**Notes:**\n* The dashboard uses normalized feature contributions and slider weights to show exactly how much each criterion adds (in percentage points) to the composite score. \n* If your CSV columns use different names, edit the `required_cols` list and mapping at the top of this file accordingly.")



