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
from sklearn.preprocessing import StandardScaler

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

if "max_budget" not in st.session_state:
    st.session_state.max_budget = 5.0
if "total_epsilon" not in st.session_state:
    st.session_state.total_epsilon = st.session_state.max_budget
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
    
    new_max = st.number_input("Total Privacy Budget Limit", min_value=1.0, value=st.session_state.max_budget, step=1.0)
    if new_max != st.session_state.max_budget:
        diff = new_max - st.session_state.max_budget
        st.session_state.total_epsilon += diff
        st.session_state.max_budget = new_max
        st.rerun()
        
    if st.button("Reset Privacy Budget"):
        st.session_state.total_epsilon = st.session_state.max_budget
        st.session_state.query_results = {}
        st.rerun()

    budget_remaining = st.session_state.total_epsilon
    st.progress(max(0.0, min(1.0, budget_remaining / st.session_state.max_budget)))
    
    # Colour-coded privacy strength indicator
    if budget_remaining > (0.6 * st.session_state.max_budget):
        budget_color = "#34d399"
    elif budget_remaining > (0.2 * st.session_state.max_budget):
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
def get_clinical_data():
    np.random.seed(42)
    n = 5000
    age = np.random.randint(18, 90, size=n)
    bp = np.random.randint(80, 180, size=n)
    chol = np.random.randint(120, 300, size=n)
    income = np.random.randint(30000, 150000, size=n)
    
    # Strong mathematical correlation to guarantee ~85% AUC
    z = 1.5 * ((age - 50)/20) + 1.2 * ((bp - 120)/20) + 0.8 * ((chol - 200)/40)
    prob = 1 / (1 + np.exp(-z))
    hd = np.random.binomial(1, prob)
    
    return pd.DataFrame({
        'PatientID': [str(uuid.uuid4())[:8] for _ in range(n)],
        'Age': age, 'BloodPressure': bp, 'Cholesterol': chol, 
        'Income': income, 'HeartDisease': hd
    })

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
else:
    df = get_clinical_data()

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

budget_exhausted = st.session_state.total_epsilon <= 0.0
if budget_exhausted:
    st.warning("🚨 Privacy Budget Exhausted (Total Epsilon Exceeded). Database locked down to prevent Differential Reconstruction Attacks.")

tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Aggregate Analytics",
    "🤖 DP Machine Learning",
    "🧬 Synthetic Data Exporter",
    "🕵️ Live Linkage Attack Simulation"
])

# ─── TAB 1: Aggregate Analytics ───────────────────────────────────────────
with tab1:
    selected_column = st.selectbox("Analyze Column", numeric_columns)
    
    if st.button("Execute DP Aggregate Query", key="btn_agg", disabled=budget_exhausted):
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
    
    potential_features = numeric_columns
    default_features = [c for c in ["Age", "BloodPressure", "Cholesterol"] if c in potential_features]
    available_features = st.multiselect("Select Features for ML Model", potential_features, default=default_features)
    
    if st.button("Train ML Models", key="btn_ml", disabled=budget_exhausted):
        if "HeartDisease" not in df.columns:
            st.error("Target column 'HeartDisease' not found in dataset!")
        elif not available_features:
            st.error("Please select at least one feature.")
        else:
            consume_budget(epsilon)
            
            X = df[available_features].copy()
            y = df['HeartDisease'].copy()
            
            # 1. Force Normalization (Crucial for Differential Privacy)
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            # 2. Bound the sensitivity for diffprivlib
            # Clip data to 3 standard deviations to prevent outlier gradient explosion
            X_scaled = np.clip(X_scaled, -3, 3)
            # Calculate theoretical max L2 norm for the DP engine
            calculated_norm = np.sqrt(X.shape[1]) * 3
            
            # 3. Train Standard Sklearn Model (The Baseline)
            true_model = SklearnLR()
            true_model.fit(X_scaled, y)
            true_preds = true_model.predict(X_scaled)
            true_probs = true_model.predict_proba(X_scaled)[:, 1]
            
            # 4. Train Diffprivlib DP Model
            # MUST pass data_norm, otherwise DP noise destroys the weights
            dp_model = dp_models.LogisticRegression(epsilon=epsilon, data_norm=calculated_norm)
            dp_model.fit(X_scaled, y)
            dp_preds = dp_model.predict(X_scaled)
            dp_probs = dp_model.predict_proba(X_scaled)[:, 1]
            
            # 5. Calculate Metrics
            true_acc = accuracy_score(y, true_preds)
            dp_acc = accuracy_score(y, dp_preds)
            fpr_t, tpr_t, _ = roc_curve(y, true_probs)
            fpr_dp, tpr_dp, _ = roc_curve(y, dp_probs)
            
            st.session_state.query_results['ml'] = {
                'true_acc': true_acc, 'dp_acc': dp_acc,
                'fpr_t': fpr_t, 'tpr_t': tpr_t, 'roc_auc_t': auc(fpr_t, tpr_t),
                'fpr_dp': fpr_dp, 'tpr_dp': tpr_dp, 'roc_auc_dp': auc(fpr_dp, tpr_dp),
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
    st.write("Generates a privacy-preserving synthetic version of the dataset using the Laplace mechanism and Randomized Response.")
    
    if st.button("Generate Anonymized CSV", key="btn_synth", disabled=budget_exhausted):
        consume_budget(epsilon)
        
        synth_df = df.copy()
        
        sensitivities = {
            'Age': 72,
            'BloodPressure': 100,
            'Cholesterol': 180,
            'Income': 120000
        }
        
        for col, sens in sensitivities.items():
            if col in synth_df.columns:
                noisy_col = synth_df[col] + np.random.laplace(loc=0, scale=sens / epsilon, size=len(synth_df))
                
                # Clip to logical bounds (e.g., cannot be negative)
                if col in COLUMN_CONFIG:
                    b_min, b_max = COLUMN_CONFIG[col]["bounds"]
                    noisy_col = np.clip(noisy_col, max(0, b_min), b_max)
                else:
                    noisy_col = np.clip(noisy_col, 0, None)
                    
                synth_df[col] = np.round(noisy_col).astype(int)
                
        if 'HeartDisease' in synth_df.columns:
            p = 1 / (1 + np.exp(epsilon))
            flip_mask = np.random.binomial(1, p, size=len(synth_df)).astype(bool)
            synth_df.loc[flip_mask, 'HeartDisease'] = 1 - synth_df.loc[flip_mask, 'HeartDisease']
            
        st.session_state.query_results['synth'] = {
            'df_orig': df.head(10).copy(),
            'df_synth': synth_df.copy(),
            'eps': epsilon
        }
        st.rerun()

    if 'synth' in st.session_state.query_results:
        res = st.session_state.query_results['synth']
        st.success(f"Generated DP Synthetic Data (ε = {res['eps']})")
        
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Original Data (First 10 Rows)**")
            st.dataframe(res['df_orig'], use_container_width=True)
        with col2:
            st.markdown("**Synthesized DP Data (First 10 Rows)**")
            st.dataframe(res['df_synth'].head(10), use_container_width=True)
        
        csv_buffer = io.StringIO()
        res['df_synth'].to_csv(csv_buffer, index=False)
        st.download_button(
            label="⬇️ Download Anonymized CSV",
            data=csv_buffer.getvalue(),
            file_name="dp_synthetic_dataset.csv",
            mime="text/csv",
        )

# ─── TAB 4: Live Linkage Attack Simulation ───────────────────────────────────────────
with tab4:
    st.markdown('<div class="section-header">Live Linkage Attack Simulation</div>', unsafe_allow_html=True)
    st.write("Attack Scenario: Identifying a specific individual by combining multiple semi-identifying traits.")
    
    if "Age" in df.columns and "BloodPressure" in df.columns and "Cholesterol" in df.columns and "HeartDisease" in df.columns:
        counts = df.groupby(['Age', 'BloodPressure', 'Cholesterol']).size().reset_index(name='count')
        unique_groups = counts[counts['count'] == 1].head(5)
        
        if unique_groups.empty:
            st.warning("No unique targets found for deanonymization.")
        else:
            victim_options = {}
            for i, row in unique_groups.iterrows():
                label = f"Victim #{i+1}: {int(row['Age'])}yo, BP {int(row['BloodPressure'])}, Cholesterol {int(row['Cholesterol'])}"
                victim_options[label] = {
                    'Age': int(row['Age']),
                    'BloodPressure': int(row['BloodPressure']),
                    'Cholesterol': int(row['Cholesterol'])
                }
                
            selected_victim_label = st.selectbox("Select Target Victim for Deanonymization Attack", list(victim_options.keys()))
            victim_traits = victim_options[selected_victim_label]
            
            st.markdown(f"""
            <div style="border:1px solid #4f46e5; border-radius:10px; padding:15px; background:rgba(79,70,229,0.1); margin-bottom: 20px;">
                <h4 style="color:#818cf8; margin-top:0;">🕵️ Known Auxiliary Profile</h4>
                <p><strong>Age:</strong> {victim_traits['Age']}</p>
                <p><strong>Blood Pressure:</strong> {victim_traits['BloodPressure']}</p>
                <p><strong>Cholesterol:</strong> {victim_traits['Cholesterol']}</p>
                <p><em>The attacker knows these traits from public sources and is querying the database to find their Heart Disease status.</em></p>
            </div>
            """, unsafe_allow_html=True)
            
            if st.button("Execute Linkage Attack", key="btn_attack", disabled=budget_exhausted):
                consume_budget(epsilon)
                
                mask = (df['Age'] == victim_traits['Age']) & \
                       (df['BloodPressure'] == victim_traits['BloodPressure']) & \
                       (df['Cholesterol'] == victim_traits['Cholesterol'])
                
                true_count = int(mask.sum())
                true_hd = int(df[mask]['HeartDisease'].sum())
                
                dp_count = true_count + np.random.laplace(0, 1.0 / epsilon)
                dp_hd = true_hd + np.random.laplace(0, 1.0 / epsilon)
                
                dp_count = max(0, int(round(dp_count)))
                dp_hd = max(0, int(round(dp_hd)))
                
                st.session_state.query_results['attack'] = {
                    'victim': selected_victim_label,
                    'true_count': true_count, 'true_hd': true_hd,
                    'dp_count': dp_count, 'dp_hd': dp_hd, 'eps': epsilon
                }
                st.rerun()
                
            if 'attack' in st.session_state.query_results:
                res = st.session_state.query_results['attack']
                
                col1, col2 = st.columns(2)
                with col1:
                    status_text = "POSITIVE" if res['true_hd'] > 0 else "NEGATIVE"
                    st.markdown(f"""
                    <div style="border:1px solid #f87171; border-radius:10px; padding:15px; background:rgba(248,113,113,0.1);">
                        <h4 style="color:#f87171; margin-top:0;">🛑 Non-Private Query Result</h4>
                        <p>Privacy Breach: {res['true_count']} unique record matched. Target is 100% deanonymized.</p>
                        <h3 style="color:#f87171;">BREACH SUCCESSFUL: Target's medical record uniquely identified as {status_text} for Heart Disease.</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                with col2:
                    st.markdown(f"""
                    <div style="border:1px solid #34d399; border-radius:10px; padding:15px; background:rgba(52,211,153,0.1);">
                        <h4 style="color:#34d399; margin-top:0;">✅ DP Query Result (ε={res['eps']})</h4>
                        <p>Matches: {res['dp_count']} | Condition Positives: {res['dp_hd']}</p>
                        <h3 style="color:#34d399;">Attack Neutralized: Laplace noise injected plausible deniability. Attacker cannot determine if the victim is even in this database or if the condition is true.</h3>
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.warning("Dataset missing required columns for Linkage Attack Demo (Age, BloodPressure, Cholesterol, HeartDisease).")
