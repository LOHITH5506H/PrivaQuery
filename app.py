"""
app.py — PrivaQuery: Differential Privacy Analytics Dashboard
Powered by Streamlit, IBM diffprivlib, and Plotly
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Compatibility patch ──────────────────────
# diffprivlib 0.6.x imports DOUBLE / DTYPE constants that were removed
# in scikit-learn ≥ 1.4.  We only use diffprivlib.tools (mean, histogram),
# but its __init__ eagerly loads the models sub-package which triggers the
# missing-import error.  Patching the two constants lets the module load.
import sklearn.tree._tree as _tree_mod
if not hasattr(_tree_mod, "DOUBLE"):
    _tree_mod.DOUBLE = np.float64
if not hasattr(_tree_mod, "DTYPE"):
    _tree_mod.DTYPE = np.float64

import diffprivlib.tools as dp

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="PrivaQuery: Differential Privacy Engine",
    page_icon="🔒",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS for premium dark theme
# ──────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    /* Global font */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #1a1a2e 40%, #16213e 100%);
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a2e 0%, #0f0c29 100%);
        border-right: 1px solid rgba(99, 102, 241, 0.2);
    }

    /* Cards / containers */
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

    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 999px;
        font-size: 0.7rem;
        font-weight: 600;
        letter-spacing: 0.05em;
    }
    .badge-true   { background: rgba(52, 211, 153, 0.15); color: #6ee7b7; border: 1px solid rgba(52, 211, 153, 0.3); }
    .badge-noisy  { background: rgba(251, 146, 60, 0.15); color: #fdba74; border: 1px solid rgba(251, 146, 60, 0.3); }
    .badge-epsilon { background: rgba(129, 140, 248, 0.15); color: #a5b4fc; border: 1px solid rgba(129, 140, 248, 0.3); }

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

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* Dataframe styling */
    .stDataFrame {
        border-radius: 12px;
        overflow: hidden;
    }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Bounds & display config for each numeric column
# ──────────────────────────────────────────────
COLUMN_CONFIG = {
    "Age":            {"bounds": (18, 90),      "unit": "years",   "bins": 15},
    "Income":         {"bounds": (30000, 150000),"unit": "USD",     "bins": 20},
    "Cholesterol":    {"bounds": (120, 300),     "unit": "mg/dL",   "bins": 18},
    "BloodPressure":  {"bounds": (80, 180),      "unit": "mmHg",    "bins": 15},
}

NUMERIC_COLUMNS = list(COLUMN_CONFIG.keys())

# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hero-title">🔒 PrivaQuery</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-subtitle">Differential Privacy Engine</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("#### ⚙️ Privacy Controls")

    epsilon = st.slider(
        "Privacy Budget (ε)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.01,
        help="Lower ε → stronger privacy (more noise). Higher ε → less noise, closer to true values.",
    )

    # Colour-coded privacy strength indicator
    if epsilon <= 0.5:
        privacy_level, privacy_color = "🟢  Maximum Privacy", "#34d399"
    elif epsilon <= 1.0:
        privacy_level, privacy_color = "🔵  Strong Privacy", "#818cf8"
    elif epsilon <= 3.0:
        privacy_level, privacy_color = "🟡  Moderate Privacy", "#fbbf24"
    else:
        privacy_level, privacy_color = "🔴  Weak Privacy", "#f87171"

    st.markdown(
        f'<p style="color:{privacy_color}; font-weight:600; font-size:0.95rem;">{privacy_level}</p>',
        unsafe_allow_html=True,
    )

    st.markdown("---")
    selected_column = st.selectbox("📊 Analyze Column", NUMERIC_COLUMNS)

    st.markdown("---")
    st.markdown(
        """
        <div style="font-size:0.78rem; color:#64748b; line-height:1.5;">
        <strong>How it works:</strong> Differential privacy adds calibrated
        noise drawn from the Laplace mechanism, guaranteeing that the
        output of a query is statistically indistinguishable whether or
        not any single individual's data is included.
        </div>
        """,
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Load Data
# ──────────────────────────────────────────────
@st.cache_data
def load_data() -> pd.DataFrame:
    df = pd.read_csv("healthcare_data.csv")
    return df

df = load_data()

# ──────────────────────────────────────────────
# Hero Header
# ──────────────────────────────────────────────
st.markdown('<div class="hero-title">PrivaQuery: Differential Privacy Engine</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="hero-subtitle">Explore aggregate healthcare analytics while mathematically guaranteeing patient privacy via IBM\'s <code>diffprivlib</code>.</div>',
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────
# Dataset Preview
# ──────────────────────────────────────────────
with st.expander("📄 Dataset Preview  (PatientID hidden for anonymization)", expanded=False):
    display_df = df.drop(columns=["PatientID"])
    st.dataframe(display_df.head(50), use_container_width=True, height=320)
    st.caption(f"Showing first 50 of {len(df):,} records.  PatientID column is automatically suppressed.")

# ──────────────────────────────────────────────
# Computation
# ──────────────────────────────────────────────
cfg = COLUMN_CONFIG[selected_column]
col_data = df[selected_column].values.astype(float)
bounds = cfg["bounds"]

# --- Means ---
true_mean = float(np.mean(col_data))
dp_mean = float(dp.mean(col_data, epsilon=epsilon, bounds=bounds))

# --- Histograms ---
num_bins = cfg["bins"]
bin_edges = np.linspace(bounds[0], bounds[1], num_bins + 1)

true_hist, _ = np.histogram(col_data, bins=bin_edges)
dp_hist, dp_edges = dp.histogram(col_data, epsilon=epsilon, bins=bin_edges, range=bounds)
dp_hist = np.clip(dp_hist, 0, None)  # clip negative noisy counts

# ──────────────────────────────────────────────
# Metric Cards Row
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">📐 Aggregate Analytics</div>', unsafe_allow_html=True)

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">True Mean <span class="badge badge-true">EXACT</span></div>
        <div class="metric-value">{true_mean:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:4px;">{selected_column} ({cfg['unit']})</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">DP Mean <span class="badge badge-noisy">NOISY</span></div>
        <div class="metric-value">{dp_mean:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:4px;">{selected_column} ({cfg['unit']})</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    noise_abs = abs(dp_mean - true_mean)
    noise_pct = (noise_abs / true_mean * 100) if true_mean != 0 else 0
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">Noise Introduced <span class="badge badge-epsilon">ε = {epsilon}</span></div>
        <div class="metric-value">±{noise_abs:,.2f}</div>
        <div style="color:#64748b; font-size:0.8rem; margin-top:4px;">{noise_pct:.2f}% deviation from true mean</div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Mean Comparison Bar Chart
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">📊 Mean Comparison</div>', unsafe_allow_html=True)

fig_mean = go.Figure()
fig_mean.add_trace(go.Bar(
    x=["True Mean"],
    y=[true_mean],
    marker=dict(
        color="rgba(52, 211, 153, 0.85)",
        line=dict(color="rgba(52, 211, 153, 1)", width=1.5),
    ),
    name="True Mean",
    text=[f"{true_mean:,.2f}"],
    textposition="outside",
    textfont=dict(color="#6ee7b7", size=14, family="Inter"),
    width=0.35,
))
fig_mean.add_trace(go.Bar(
    x=["DP Mean"],
    y=[dp_mean],
    marker=dict(
        color="rgba(251, 146, 60, 0.85)",
        line=dict(color="rgba(251, 146, 60, 1)", width=1.5),
    ),
    name=f"DP Mean (ε={epsilon})",
    text=[f"{dp_mean:,.2f}"],
    textposition="outside",
    textfont=dict(color="#fdba74", size=14, family="Inter"),
    width=0.35,
))
fig_mean.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    height=380,
    margin=dict(l=40, r=40, t=30, b=40),
    showlegend=True,
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5,
        font=dict(size=12),
    ),
    yaxis=dict(
        gridcolor="rgba(148, 163, 184, 0.08)",
        title=f"{selected_column} ({cfg['unit']})",
    ),
    xaxis=dict(showgrid=False),
    bargap=0.5,
)
st.plotly_chart(fig_mean, use_container_width=True)

# ──────────────────────────────────────────────
# Distribution Comparison (side-by-side histograms)
# ──────────────────────────────────────────────
st.markdown('<div class="section-header">🔬 Distribution Comparison — True vs. Differentially Private</div>', unsafe_allow_html=True)

bin_centers = 0.5 * (bin_edges[:-1] + bin_edges[1:])
bar_width = (bin_edges[1] - bin_edges[0]) * 0.85

fig_dist = make_subplots(
    rows=1, cols=2,
    subplot_titles=(
        f"True Distribution — {selected_column}",
        f"DP Distribution (ε={epsilon}) — {selected_column}",
    ),
    horizontal_spacing=0.08,
)

# True histogram
fig_dist.add_trace(
    go.Bar(
        x=bin_centers,
        y=true_hist,
        marker=dict(
            color=[f"rgba(52, 211, 153, {0.4 + 0.6 * v / max(true_hist)})" for v in true_hist],
            line=dict(color="rgba(52, 211, 153, 0.6)", width=0.5),
        ),
        name="True",
        width=bar_width,
        showlegend=True,
    ),
    row=1, col=1,
)

# DP histogram
max_dp = max(dp_hist) if max(dp_hist) > 0 else 1
fig_dist.add_trace(
    go.Bar(
        x=bin_centers,
        y=dp_hist,
        marker=dict(
            color=[f"rgba(251, 146, 60, {0.4 + 0.6 * v / max_dp})" for v in dp_hist],
            line=dict(color="rgba(251, 146, 60, 0.6)", width=0.5),
        ),
        name=f"DP (ε={epsilon})",
        width=bar_width,
        showlegend=True,
    ),
    row=1, col=2,
)

fig_dist.update_layout(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter", color="#e2e8f0"),
    height=420,
    margin=dict(l=40, r=40, t=50, b=50),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.08, xanchor="center", x=0.5,
        font=dict(size=12),
    ),
)

for axis in ["yaxis", "yaxis2"]:
    fig_dist.update_layout(**{axis: dict(gridcolor="rgba(148, 163, 184, 0.08)", title="Count")})
for axis in ["xaxis", "xaxis2"]:
    fig_dist.update_layout(**{axis: dict(gridcolor="rgba(148, 163, 184, 0.05)", title=f"{selected_column} ({cfg['unit']})")})

st.plotly_chart(fig_dist, use_container_width=True)

# ──────────────────────────────────────────────
# Epsilon Sensitivity Explainer
# ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-header">🧠 Understanding the Privacy–Utility Trade-off</div>', unsafe_allow_html=True)

explain_col1, explain_col2 = st.columns(2)
with explain_col1:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#6ee7b7;">🔐 Low Epsilon (ε → 0)</h4>
        <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
            <li>Maximum privacy guarantee</li>
            <li>Heavy noise masks individual contributions</li>
            <li>Aggregate results may deviate significantly</li>
            <li>Ideal when data is highly sensitive</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with explain_col2:
    st.markdown("""
    <div class="metric-card">
        <h4 style="color:#fdba74;">📈 High Epsilon (ε → ∞)</h4>
        <ul style="color:#94a3b8; font-size:0.9rem; line-height:1.8;">
            <li>Minimal privacy protection</li>
            <li>Little noise, results mirror true data</li>
            <li>Higher analytical accuracy</li>
            <li>Suitable for less sensitive datasets</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────
st.markdown(
    """
    <div style="text-align:center; color:#475569; font-size:0.75rem; margin-top:48px; padding-bottom:24px;">
        PrivaQuery · Built with Streamlit & IBM diffprivlib · Differential Privacy ε-Mechanism
    </div>
    """,
    unsafe_allow_html=True,
)
