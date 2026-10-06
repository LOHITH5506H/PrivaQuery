import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sklearn.tree._tree as _tree_mod
import uuid
import io

# ── Compatibility patch ──────────────────────
if not hasattr(_tree_mod, "DOUBLE"):
    _tree_mod.DOUBLE = np.float64
if not hasattr(_tree_mod, "DTYPE"):
    _tree_mod.DTYPE = np.float64

import diffprivlib.tools as dp
import diffprivlib.models as dp_models
from sklearn.linear_model import LogisticRegression as SklearnLR
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, accuracy_score

st.set_page_config(
    page_title="PrivaQuery: Differential Privacy Engine",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }
    
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }
    
    .metric-card {
        background: rgba(30, 30, 60, 0.7);
        backdrop-filter: blur(12px);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 16px;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(99, 102, 241, 0.15);
    }

    .metric-value {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8, #c084fc);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
    }
    .metric-label {
        font-size: 0.85rem;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        margin-bottom: 4px;
    }
    
    .section-header {
        font-size: 1.35rem;
        font-weight: 600;
        color: #e2e8f0;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 10px;
    }

    .hero-title {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #818cf8 0%, #c084fc 50%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 4px;
    }
    .hero-subtitle {
        font-size: 1rem;
        color: #94a3b8;
        margin-bottom: 24px;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

if "total_epsilon" not in st.session_state:
    st.session_state.total_epsilon = 5.0
if "query_results" not in st.session_state:
    st.session_state.query_results = {}

with st.sidebar:
    st.markdown('<div class="hero-title">🔒 PrivaQuery</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Differential Privacy Engine</div>', unsafe_allow_html=True)
    st.markdown("---")
    
    st.markdown("#### ⚙️ Privacy Controls")
    epsilon = st.slider(
        "Query Privacy Budget (ε)",
        min_value=0.01, max_value=2.0, value=0.5, step=0.01,
        help="Amount of budget to consume for the next query."
    )
    
    st.markdown("---")
    st.markdown("#### 🔋 Budget Tracker")
    budget_remaining = st.session_state.total_epsilon
    st.progress(max(0.0, min(1.0, budget_remaining / 5.0)))
    
    # Colour-coded privacy strength indicator
    if budget_remaining > 3.0:
        budget_color = "#34d399"
    elif budget_remaining > 1.0:
        budget_color = "#fbbf24"
    else:
        budget_color = "#f87171"
        
    st.markdown(f'<p style="color:{budget_color}; font-weight:600;">{budget_remaining:.2f} ε remaining</p>', unsafe_allow_html=True)
    
    if budget_remaining <= 0:
        st.error("Privacy Budget Exhausted — Potential Reconstruction Attack Prevented.")
        
    st.markdown("---")
    uploaded_file = st.file_uploader("Upload custom CSV", type=["csv"])
    
def consume_budget(eps):
    if st.session_state.total_epsilon - eps < 0:
        st.sidebar.error("Not enough budget left for this query! Refresh page to reset.")
        st.stop()
    st.session_state.total_epsilon -= eps

@st.cache_data
def load_default_data():
    return pd.read_csv("healthcare_data.csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = load_default_data()

# Auto-detect numeric columns and bounds
numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
if 'HeartDisease' in numeric_columns:
    numeric_columns.remove('HeartDisease')
# Also remove ID columns if they were parsed as numbers
numeric_columns = [c for c in numeric_columns if not c.lower().endswith('id')]

COLUMN_CONFIG = {}
for col in numeric_columns:
    q01 = df[col].quantile(0.01)
    q99 = df[col].quantile(0.99)
    # Handle case where q01 == q99
    if q01 == q99:
        q01 -= 1.0
        q99 += 1.0
    COLUMN_CONFIG[col] = {"bounds": (float(q01), float(q99)), "bins": 15}

st.markdown('<div class="hero-title">PrivaQuery: Differential Privacy Engine</div>', unsafe_allow_html=True)

if st.session_state.total_epsilon <= 0:
    st.error("🚨 Privacy Budget Exhausted. To protect the dataset from reconstruction attacks, all analytical interfaces have been locked. Please contact your compliance officer or refresh the environment to reset.")
    st.stop()

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Aggregate Analytics",
    "🤖 DP Machine Learning",
    "🧬 Synthetic Data Exporter",
    "🕵️ Live Linkage Attack Simulation"
])

# ─── TAB 1: Aggregate Analytics ───────────────────────────────────────────
with tab1:
    selected_column = st.selectbox("Analyze Column", numeric_columns)
    
    if st.button("Execute DP Aggregate Query", key="btn_agg"):
        consume_budget(epsilon)
        
        cfg = COLUMN_CONFIG[selected_column]
        col_data = df[selected_column].values.astype(float)
        bounds = cfg["bounds"]
        
        true_mean = float(np.mean(col_data))
        dp_mean = float(dp.mean(col_data, epsilon=epsilon, bounds=bounds))
        
        num_bins = cfg["bins"]
        bin_edges = np.linspace(bounds[0], bounds[1], num_bins + 1)
        
        true_hist, _ = np.histogram(col_data, bins=bin_edges)
        dp_hist, _ = dp.histogram(col_data, epsilon=epsilon, bins=bin_edges, range=bounds)
        dp_hist = np.clip(dp_hist, 0, None)
        
        st.session_state.query_results['agg'] = {
            'col': selected_column, 'true_mean': true_mean, 'dp_mean': dp_mean,
            'true_hist': true_hist, 'dp_hist': dp_hist, 'edges': bin_edges, 'eps': epsilon
        }
        st.rerun()

    if 'agg' in st.session_state.query_results:
        res = st.session_state.query_results['agg']
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">True Mean</div><div class="metric-value">{res["true_mean"]:,.2f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">DP Mean (ε={res["eps"]})</div><div class="metric-value">{res["dp_mean"]:,.2f}</div></div>', unsafe_allow_html=True)
        with col3:
            noise_pct = abs(res["dp_mean"] - res["true_mean"]) / res["true_mean"] * 100 if res["true_mean"]!=0 else 0
            st.markdown(f'<div class="metric-card"><div class="metric-label">Noise Introduced</div><div class="metric-value">±{noise_pct:.2f}%</div></div>', unsafe_allow_html=True)

        fig_dist = make_subplots(rows=1, cols=2, subplot_titles=(f"True Distribution", f"DP Distribution"))
        bin_centers = 0.5 * (res["edges"][:-1] + res["edges"][1:])
        fig_dist.add_trace(go.Bar(x=bin_centers, y=res["true_hist"], name="True", marker_color="#34d399"), row=1, col=1)
        fig_dist.add_trace(go.Bar(x=bin_centers, y=res["dp_hist"], name="DP", marker_color="#fbbf24"), row=1, col=2)
        fig_dist.update_layout(template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_dist, use_container_width=True)

# ─── TAB 2: DP Machine Learning ───────────────────────────────────────────
with tab2:
    st.markdown('<div class="section-header">Predicting HeartDisease using DP Logistic Regression</div>', unsafe_allow_html=True)
    st.write("Compare standard Logistic Regression with IBM diffprivlib\'s DP-compliant equivalent.")
    
    if st.button("Train ML Models", key="btn_ml"):
        if "HeartDisease" not in df.columns:
            st.error("Target column 'HeartDisease' not found in dataset!")
        else:
            consume_budget(epsilon)
            
            features = ["Age", "BloodPressure", "Cholesterol"]
            available_features = [f for f in features if f in df.columns]
            
            if not available_features:
                st.error("No compatible features found.")
            else:
                X = df[available_features].copy()
                y = df["HeartDisease"].copy()
                
                for f in available_features:
                    b_min, b_max = COLUMN_CONFIG[f]["bounds"]
                    X[f] = np.clip(X[f], b_min, b_max)
                    X[f] = (X[f] - b_min) / (b_max - b_min)
                    
                data_norm = np.sqrt(len(available_features))
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
                
                clf_true = SklearnLR()
                clf_true.fit(X_train, y_train)
                true_preds = clf_true.predict(X_test)
                true_probs = clf_true.predict_proba(X_test)[:, 1]
                true_acc = accuracy_score(y_test, true_preds)
                fpr_t, tpr_t, _ = roc_curve(y_test, true_probs)
                roc_auc_t = auc(fpr_t, tpr_t)
                
                clf_dp = dp_models.LogisticRegression(epsilon=epsilon, data_norm=data_norm)
                clf_dp.fit(X_train, y_train)
                dp_preds = clf_dp.predict(X_test)
                dp_probs = clf_dp.predict_proba(X_test)[:, 1]
                dp_acc = accuracy_score(y_test, dp_preds)
                fpr_dp, tpr_dp, _ = roc_curve(y_test, dp_probs)
                roc_auc_dp = auc(fpr_dp, tpr_dp)
                
                st.session_state.query_results['ml'] = {
                    'true_acc': true_acc, 'dp_acc': dp_acc,
                    'fpr_t': fpr_t, 'tpr_t': tpr_t, 'roc_auc_t': roc_auc_t,
                    'fpr_dp': fpr_dp, 'tpr_dp': tpr_dp, 'roc_auc_dp': roc_auc_dp,
                    'eps': epsilon
                }
                st.rerun()
                
    if 'ml' in st.session_state.query_results:
        res = st.session_state.query_results['ml']
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">True Accuracy</div><div class="metric-value">{res["true_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">DP Accuracy (ε={res["eps"]})</div><div class="metric-value">{res["dp_acc"]*100:.1f}%</div></div>', unsafe_allow_html=True)
            
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=res['fpr_t'], y=res['tpr_t'], mode='lines', name=f'True (AUC = {res["roc_auc_t"]:.2f})', line=dict(color="#34d399")))
        fig_roc.add_trace(go.Scatter(x=res['fpr_dp'], y=res['tpr_dp'], mode='lines', name=f'DP (AUC = {res["roc_auc_dp"]:.2f})', line=dict(color="#fbbf24")))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random', line=dict(dash='dash', color="#64748b")))
        fig_roc.update_layout(title="ROC Curve Comparison", template="plotly_dark", paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_roc, use_container_width=True)

# ─── TAB 3: Synthetic Data Exporter ───────────────────────────────────────────
with tab3:
    st.markdown('<div class="section-header">Generate DP Anonymized Dataset</div>', unsafe_allow_html=True)
    st.write("Generates 1,000 synthetic rows by preserving marginals and cross-column covariance using DP algorithms.")
    
    if st.button("Generate Anonymized CSV", key="btn_synth"):
        consume_budget(epsilon)
        
        features_synth = numeric_columns
        X_synth = df[features_synth].copy()
        n_samples = len(X_synth)
        
        for col in features_synth:
            b_min, b_max = COLUMN_CONFIG[col]["bounds"]
            X_synth[col] = np.clip(X_synth[col], b_min, b_max)
            X_synth[col] = (X_synth[col] - b_min) / (b_max - b_min)
            
        k = len(features_synth)
        eps_per_query = epsilon / (k + k*(k+1)/2)
        
        true_means = X_synth.mean()
        dp_means_scaled = [true_means[col] + np.random.laplace(0, 1.0 / (n_samples * eps_per_query)) for col in features_synth]
        
        true_cov = X_synth.cov()
        dp_cov_scaled = true_cov.copy()
        for i in range(k):
            for j in range(i, k):
                noise = np.random.laplace(0, 1.0 / (n_samples * eps_per_query))
                dp_cov_scaled.iloc[i, j] += noise
                if i != j:
                    dp_cov_scaled.iloc[j, i] += noise
                    
        eigvals, eigvecs = np.linalg.eigh(dp_cov_scaled)
        eigvals[eigvals < 0] = 1e-6
        dp_cov_scaled = eigvecs @ np.diag(eigvals) @ eigvecs.T
        
        synth_scaled = np.random.multivariate_normal(dp_means_scaled, dp_cov_scaled, 1000)
        synth_df = pd.DataFrame(synth_scaled, columns=features_synth)
        
        for col in features_synth:
            b_min, b_max = COLUMN_CONFIG[col]["bounds"]
            synth_df[col] = synth_df[col] * (b_max - b_min) + b_min
            synth_df[col] = np.clip(synth_df[col], b_min, b_max).round(1)
            
        st.session_state.query_results['synth'] = {
            'df': synth_df, 'eps': epsilon
        }
        st.rerun()

    if 'synth' in st.session_state.query_results:
        res = st.session_state.query_results['synth']
        st.success(f"Generated 1,000 rows of DP Synthetic Data (ε = {res['eps']})")
        st.dataframe(res['df'].head(), use_container_width=True)
        
        csv_buffer = io.StringIO()
        res['df'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Anonymized CSV",
            data=csv_buffer.getvalue(),
            file_name="dp_anonymized_dataset.csv",
            mime="text/csv",
        )

# ─── TAB 4: Live Linkage Attack Simulation ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Live Linkage Attack Simulation</div>', unsafe_allow_html=True)
    st.write("Attack Scenario: Identifying a specific individual by combining multiple semi-identifying traits.")
    
    if "Age" in df.columns and "BloodPressure" in df.columns and "HeartDisease" in df.columns:
        counts = df.groupby(['Age', 'BloodPressure']).size().reset_index(name='count')
        unique_groups = counts[counts['count'] == 1]
        
        if not unique_groups.empty:
            target = unique_groups.iloc[0]
            t_age = target['Age']
            t_bp = target['BloodPressure']
            
            st.info(f"**Target Found:** Patient with Age = {t_age} and BloodPressure = {t_bp}")
            
            if st.button("Simulate Attack Query", key="btn_attack"):
                consume_budget(epsilon)
                
                true_count = len(df[(df['Age'] == t_age) & (df['BloodPressure'] == t_bp)])
                true_hd = df[(df['Age'] == t_age) & (df['BloodPressure'] == t_bp)]['HeartDisease'].sum()
                
                dp_count = true_count + np.random.laplace(0, 1.0 / epsilon)
                dp_hd = true_hd + np.random.laplace(0, 1.0 / epsilon)
                
                st.session_state.query_results['attack'] = {
                    't_age': t_age, 't_bp': t_bp,
                    'true_count': true_count, 'true_hd': true_hd,
                    'dp_count': dp_count, 'dp_hd': dp_hd, 'eps': epsilon
                }
                st.rerun()
                
            if 'attack' in st.session_state.query_results:
                res = st.session_state.query_results['attack']
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown(f"""
                    <div style="border:1px solid #f87171; border-radius:10px; padding:15px; background:rgba(248,113,113,0.1);">
                        <h4 style="color:#f87171; margin-top:0;">🛑 Non-Private Query Result</h4>
                        <p>Query: COUNT where Age={res['t_age']} AND BloodPressure={res['t_bp']}</p>
                        <h2 style="color:#f87171;">Matched Patients: {res['true_count']}</h2>
                        <h2 style="color:#f87171;">Heart Disease Positive: {res['true_hd']}</h2>
                        <p><em>Conclusion: The analyst knows EXACTLY 1 person exists with these traits and they have Heart Disease. Privacy breached!</em></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div style="border:1px solid #34d399; border-radius:10px; padding:15px; background:rgba(52,211,153,0.1);">
                        <h4 style="color:#34d399; margin-top:0;">✅ DP Query Result (ε={res['eps']})</h4>
                        <p>Query: COUNT where Age={res['t_age']} AND BloodPressure={res['t_bp']}</p>
                        <h2 style="color:#34d399;">Matched Patients: {res['dp_count']:.2f}</h2>
                        <h2 style="color:#34d399;">Heart Disease Positive: {res['dp_hd']:.2f}</h2>
                        <p><em>Conclusion: The added Laplace noise creates plausible deniability. The attacker cannot be certain if the patient exists in the dataset or if their status is true.</em></p>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.warning("No completely unique (Age, BloodPressure) pairs found in the dataset to demonstrate the linkage attack.")
    else:
        st.warning("Dataset missing required columns for Linkage Attack Demo (Age, BloodPressure, HeartDisease).")
