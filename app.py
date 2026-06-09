"""
app.py  ─  Movie Recommendation Engine
Streamlit dashboard with a cinema-noir editorial aesthetic.
Run with: streamlit run app.py
"""

import json
import pickle
import numpy as np
import pandas as pd
import streamlit as st
from pathlib import Path


st.set_page_config(
    page_title="CineRec · Movie Recommendation Engine",
    page_icon="🎞️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")
PLOTS_DIR  = Path("plots")
DEVICE_STR = "cpu"


st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Mono:wght@300;400;500&family=Outfit:wght@300;400;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0a0a0f !important;
    color: #e8e0d0 !important;
    font-family: 'Outfit', sans-serif;
}

[data-testid="stAppViewContainer"] > .main { padding: 0 !important; }
[data-testid="stHeader"] { display: none !important; }
[data-testid="collapsedControl"] { display: none !important; }
section[data-testid="stSidebar"] { display: none !important; }
.block-container { padding: 0 !important; max-width: 100% !important; }

::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #c9a84c; border-radius: 2px; }

.nav-bar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 18px 48px;
    border-bottom: 1px solid #1e1e2e;
    background: rgba(10,10,15,0.95);
    position: sticky;
    top: 0;
    z-index: 100;
    backdrop-filter: blur(12px);
}
.nav-logo {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    letter-spacing: -0.02em;
    color: #c9a84c;
}
.nav-logo span { color: #e8e0d0; }
.nav-tagline {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #5a5a7a;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}

.hero {
    padding: 64px 48px 40px;
    border-bottom: 1px solid #1e1e2e;
    background: linear-gradient(135deg, #0a0a0f 0%, #0f0f1a 60%, #0a0a0f 100%);
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(201,168,76,0.07) 0%, transparent 70%);
    pointer-events: none;
}
.hero-eyebrow {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-bottom: 16px;
}
.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: clamp(2.4rem, 5vw, 4rem);
    line-height: 1.05;
    color: #e8e0d0;
    margin-bottom: 12px;
}
.hero-title em { color: #c9a84c; font-style: italic; }
.hero-sub {
    font-size: 0.95rem;
    color: #6a6a8a;
    font-weight: 300;
    max-width: 560px;
    line-height: 1.6;
}

.controls-strip {
    display: flex;
    align-items: flex-end;
    gap: 32px;
    padding: 28px 48px;
    background: #0c0c14;
    border-bottom: 1px solid #1e1e2e;
    flex-wrap: wrap;
}
.control-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 6px;
}

.stat-row { display: flex; gap: 1px; margin-bottom: 40px; background: #1e1e2e; }
.stat-card {
    flex: 1;
    padding: 24px 28px;
    background: #0c0c14;
    position: relative;
}
.stat-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 28px; right: 28px;
    height: 2px;
    background: linear-gradient(90deg, #c9a84c, transparent);
    opacity: 0;
    transition: opacity 0.2s;
}
.stat-card:hover::after { opacity: 1; }
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a7a;
    margin-bottom: 10px;
}
.stat-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.2rem;
    color: #c9a84c;
    line-height: 1;
}
.stat-unit {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #3a3a5e;
    margin-top: 4px;
    letter-spacing: 0.1em;
    text-transform: uppercase;
}

.section-heading {
    display: flex;
    align-items: baseline;
    gap: 14px;
    margin-bottom: 20px;
    padding-bottom: 12px;
    border-bottom: 1px solid #1e1e2e;
}
.section-heading h2 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.5rem;
    color: #e8e0d0;
    font-weight: 400;
}
.section-heading .sh-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    color: #5a5a7a;
    text-transform: uppercase;
}

.rec-list { display: flex; flex-direction: column; gap: 2px; }
.rec-item {
    display: grid;
    grid-template-columns: 48px 1fr auto auto;
    align-items: center;
    gap: 20px;
    padding: 16px 20px;
    background: #0c0c14;
    border-left: 3px solid transparent;
    transition: all 0.15s;
    cursor: default;
}
.rec-item:hover { background: #111120; border-left-color: #c9a84c; }
.rec-rank {
    font-family: 'DM Serif Display', serif;
    font-size: 1.8rem;
    color: #2a2a3e;
    text-align: center;
    line-height: 1;
}
.rec-item:hover .rec-rank { color: #c9a84c; }
.rec-title {
    font-size: 0.95rem;
    font-weight: 600;
    color: #e8e0d0;
    margin-bottom: 3px;
    line-height: 1.3;
}
.rec-genre {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #5a5a7a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}
.rec-score {
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    color: #c9a84c;
    text-align: right;
    white-space: nowrap;
}
.rec-bar {
    width: 60px;
    height: 3px;
    background: #1e1e2e;
    border-radius: 1px;
    overflow: hidden;
}
.rec-bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #c9a84c, #e8c76a);
    border-radius: 1px;
}

.compare-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 2px; }
.compare-col { background: #0c0c14; padding: 20px; }
.compare-col-header {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #c9a84c;
    padding-bottom: 12px;
    margin-bottom: 12px;
    border-bottom: 1px solid #1e1e2e;
}
.compare-item {
    padding: 10px 0;
    border-bottom: 1px solid #13131e;
    font-size: 0.8rem;
    color: #a0a0c0;
    line-height: 1.4;
}
.compare-item strong {
    display: block;
    color: #e8e0d0;
    font-weight: 600;
    font-size: 0.85rem;
    margin-bottom: 2px;
}

.algo-box {
    background: #0c0c14;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #c9a84c;
    padding: 24px 28px;
    margin-top: 20px;
}
.algo-box h4 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.1rem;
    color: #c9a84c;
    margin-bottom: 10px;
    font-weight: 400;
}
.algo-box p {
    font-size: 0.88rem;
    color: #8080a0;
    line-height: 1.75;
}

.metrics-table { width: 100%; border-collapse: collapse; }
.metrics-table th {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #5a5a7a;
    padding: 10px 16px;
    text-align: left;
    border-bottom: 1px solid #1e1e2e;
    background: #0a0a0f;
}
.metrics-table td {
    padding: 14px 16px;
    font-size: 0.88rem;
    color: #a0a0c0;
    border-bottom: 1px solid #13131e;
    background: #0c0c14;
}
.metrics-table tr:hover td { background: #111120; }
.metrics-table td:first-child {
    font-family: 'DM Serif Display', serif;
    font-size: 1rem;
    color: #e8e0d0;
}
.metrics-table .best { color: #c9a84c; font-weight: 600; }

.info-banner {
    padding: 14px 20px;
    background: #0c0c14;
    border: 1px solid #1e1e2e;
    border-left: 3px solid #c9a84c;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #8080a0;
    letter-spacing: 0.04em;
    margin-bottom: 24px;
}
.warn-banner {
    padding: 14px 20px;
    background: rgba(201,168,76,0.06);
    border: 1px solid rgba(201,168,76,0.25);
    border-left: 3px solid #c9a84c;
    font-size: 0.82rem;
    color: #c9a84c;
    margin-bottom: 20px;
}
.err-banner {
    padding: 14px 20px;
    background: rgba(200,60,60,0.06);
    border: 1px solid rgba(200,60,60,0.25);
    border-left: 3px solid #c83c3c;
    font-size: 0.82rem;
    color: #e07070;
    margin-bottom: 20px;
}

[data-testid="stNumberInput"] input {
    background: #0c0c14 !important;
    color: #e8e0d0 !important;
    border-color: #2a2a3e !important;
    border-radius: 2px !important;
}
div[data-testid="stImage"] img { border-radius: 0 !important; }
.stSpinner > div { border-top-color: #c9a84c !important; }

/* Streamlit radio override */
[data-testid="stRadio"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #6a6a8a !important;
    text-transform: uppercase !important;
    letter-spacing: 0.06em !important;
}
[data-testid="stRadio"] [aria-checked="true"] + div {
    color: #c9a84c !important;
}

/* primary button */
[data-testid="baseButton-primary"] {
    background: #c9a84c !important;
    color: #0a0a0f !important;
    border: none !important;
    border-radius: 2px !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    font-weight: 600 !important;
}
[data-testid="baseButton-primary"]:hover {
    background: #e8c76a !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0c0c14 !important;
    border-bottom: 1px solid #1e1e2e !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.62rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #5a5a7a !important;
    background: transparent !important;
    border-radius: 0 !important;
    padding: 14px 24px !important;
}
.stTabs [aria-selected="true"] {
    color: #c9a84c !important;
    border-bottom: 2px solid #c9a84c !important;
}
[data-testid="stTabsContent"] {
    background: #0a0a0f !important;
    padding: 24px 0 !important;
}

.gold-divider {
    height: 1px;
    background: linear-gradient(90deg, #c9a84c, transparent);
    margin: 32px 0;
}

.about-block { max-width: 720px; }
.about-block h3 {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #e8e0d0;
    margin-bottom: 12px;
    margin-top: 32px;
    font-weight: 400;
}
.about-block h3:first-child { margin-top: 0; }
.about-block p {
    font-size: 0.88rem;
    color: #7070a0;
    line-height: 1.8;
    margin-bottom: 8px;
}
.about-block ul { padding-left: 20px; margin-top: 8px; }
.about-block li {
    font-size: 0.85rem;
    color: #7070a0;
    line-height: 1.8;
    margin-bottom: 4px;
}
.about-block li strong { color: #c9a84c; }
.about-pill {
    display: inline-block;
    padding: 3px 10px;
    background: rgba(201,168,76,0.1);
    border: 1px solid rgba(201,168,76,0.3);
    border-radius: 2px;
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #c9a84c;
    margin-right: 6px;
    margin-bottom: 6px;
}
</style>
""", unsafe_allow_html=True)




@st.cache_resource
def load_resources():
    resources = {}
    missing   = []

    try:
        with open(PROC_DIR / "config.json") as f:
            config = json.load(f)
        resources["n_users"] = config["n_users"]
        resources["n_items"] = config["n_items"]
    except FileNotFoundError:
        st.error("data/processed/config.json not found. Run python data_loader.py first.")
        st.stop()

    n_users = resources["n_users"]
    n_items = resources["n_items"]

    with open(PROC_DIR / "idx2movie.pkl", "rb") as f:
        resources["idx2movie"] = pickle.load(f)
    with open(PROC_DIR / "movie2idx.pkl", "rb") as f:
        resources["movie2idx"] = pickle.load(f)

    resources["train_df"]  = pd.read_csv(PROC_DIR / "train_df.csv")
    resources["movies_df"] = pd.read_csv(PROC_DIR / "movies_processed.csv")

    try:
        from baseline import PopularityRecommender
        pop = PopularityRecommender()
        pop.load(MODELS_DIR / "popularity_model.pkl")
        resources["popularity"] = pop
    except Exception:
        missing.append("popularity_model.pkl — run python baseline.py")

    try:
        from svd_scratch import SVDRecommender
        svd = SVDRecommender()
        svd.load(MODELS_DIR / "svd_model.npz")
        resources["svd"] = svd
    except Exception:
        missing.append("svd_model.npz — run python svd_scratch.py")

    try:
        import torch
        from ncf_model import NCF
        ncf = NCF(n_users, n_items, embed_dim=64)
        ncf.load_state_dict(torch.load(MODELS_DIR / "ncf_best.pt", map_location=DEVICE_STR))
        ncf.eval()
        resources["ncf"] = ncf
    except Exception:
        missing.append("ncf_best.pt — run python train_ncf.py")

    try:
        import torch
        from ncf_model import NeuMF
        neumf = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32)
        neumf.load_state_dict(torch.load(MODELS_DIR / "neumf_best.pt", map_location=DEVICE_STR))
        neumf.eval()
        resources["neumf"] = neumf
    except Exception:
        missing.append("neumf_best.pt — run python train_ncf.py")

    try:
        resources["comparison_df"] = pd.read_csv(PLOTS_DIR / "model_comparison.csv")
    except Exception:
        resources["comparison_df"] = None
        missing.append("model_comparison.csv — run python compare_models.py")

    resources["missing"] = missing
    return resources



def get_movie_info(movie_idx, movies_df, idx2movie):
    row = movies_df[movies_df["movie_idx"] == movie_idx]
    if row.empty:
        mid = idx2movie.get(movie_idx, "?")
        return {"title": f"Movie #{mid}", "genres": "Unknown"}
    return {"title": row.iloc[0]["Title"], "genres": row.iloc[0]["Genres"]}

def check_cold_start(user_idx, train_df, threshold=5):
    return int((train_df["user_idx"] == user_idx).sum()) < threshold

def predict_batch_ncf(model, user_idx, movie_idx_list):
    import torch
    with torch.no_grad():
        users  = torch.tensor([user_idx] * len(movie_idx_list), dtype=torch.long)
        items  = torch.tensor(movie_idx_list, dtype=torch.long)
        scores = model(users, items).squeeze(-1).cpu().numpy()
    return scores

def get_recommendations(model_key, user_idx, top_k, resources):
    train_df  = resources["train_df"]
    movies_df = resources["movies_df"]
    idx2movie = resources["idx2movie"]
    n_items   = resources["n_items"]

    seen      = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
    all_items = [i for i in range(n_items) if i not in seen]

    if   model_key == "popularity": scores = resources["popularity"].predict_batch(user_idx, all_items)
    elif model_key == "svd":        scores = resources["svd"].predict_batch(user_idx, all_items)
    elif model_key == "ncf":        scores = predict_batch_ncf(resources["ncf"],   user_idx, all_items)
    elif model_key == "neumf":      scores = predict_batch_ncf(resources["neumf"], user_idx, all_items)
    else: return []

    top_idx = np.argsort(scores)[::-1][:top_k]
    max_sc  = float(scores[top_idx[0]]) if len(top_idx) else 1.0
    recs = []
    for rank, idx in enumerate(top_idx, 1):
        item = all_items[idx]
        sc   = float(scores[idx])
        info = get_movie_info(item, movies_df, idx2movie)
        recs.append({
            "rank": rank, "title": info["title"], "genres": info["genres"],
            "score": sc,  "pct": sc / max_sc if max_sc > 0 else 0,
        })
    return recs

def get_metrics_for(model_name, comparison_df):
    if comparison_df is None:
        return {}
    row = comparison_df[comparison_df["Model"] == model_name]
    return row.iloc[0].to_dict() if not row.empty else {}

MODEL_KEYS = {"Popularity": "popularity", "SVD": "svd", "NCF": "ncf", "NeuMF": "neumf"}

ALGO_INFO = {
    "Popularity": ("Popularity Baseline",
        "Ranks every movie by <em>count × mean rating</em>. User-agnostic — every user "
        "receives the same globally popular list, filtered for already-seen titles. "
        "Essential as a sanity-check: any personalised model must outperform it to justify complexity."),
    "SVD": ("Biased Matrix Factorisation",
        "Predicts rating as <code>μ + b_u + b_i + P[u]·Q[i]</code>. Latent vectors P and Q "
        "(64-dim) are learned by SGD with L2 regularisation and learning-rate decay — "
        "implemented entirely in NumPy, zero sklearn SVD."),
    "NCF": ("Neural Collaborative Filtering",
        "Concatenates user and item embeddings (64-dim) and feeds them through a 4-layer MLP "
        "(128→64→32→1) with ReLU and dropout. Trained on implicit feedback via binary "
        "cross-entropy with negative sampling — captures non-linear patterns SVD cannot."),
    "NeuMF": ("Neural Matrix Factorisation",
        "Dual-path architecture: a <strong>GMF path</strong> (element-wise product, 32-dim) "
        "runs in parallel with an <strong>MLP path</strong> (64→32→16). Both outputs are "
        "concatenated and fused (48→1) — combining linear and non-linear signals simultaneously."),
}


def main():
    res = load_resources()

    # Nav
    st.markdown("""
    <div class="nav-bar">
        <div>
            <div class="nav-logo">Cine<span>Rec</span></div>
            <div class="nav-tagline">Movie Recommendation Engine &nbsp;·&nbsp; SVD + Neural CF</div>
        </div>
        <div style="font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a3a5e;letter-spacing:0.1em;text-transform:uppercase;">
            MovieLens-1M &nbsp;·&nbsp; 1,000,209 ratings
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
        <div class="hero-eyebrow">▸ Recommendation System &nbsp;·&nbsp; Amazon ML Summer School</div>
        <div class="hero-title">What should you<br>watch <em>next?</em></div>
        <div class="hero-sub">Four models. One dataset. Compare popularity, matrix factorisation,
        and deep neural approaches side by side.</div>
    </div>""", unsafe_allow_html=True)


    st.markdown('<div class="controls-strip">', unsafe_allow_html=True)
    ctrl1, ctrl2, ctrl3, ctrl4 = st.columns([3, 1.2, 1.2, 1])

    with ctrl1:
        st.markdown('<div class="control-label">Select algorithm</div>', unsafe_allow_html=True)
        model_choice = st.radio("model", ["Popularity", "SVD", "NCF", "NeuMF", "Compare All"],
                                horizontal=True, label_visibility="collapsed")
    with ctrl2:
        st.markdown('<div class="control-label">User ID</div>', unsafe_allow_html=True)
        user_id = st.number_input("uid", min_value=1, max_value=6040, value=1, step=1,
                                  label_visibility="collapsed")
    with ctrl3:
        st.markdown('<div class="control-label">Top-K results</div>', unsafe_allow_html=True)
        top_k = st.slider("topk", min_value=5, max_value=20, value=10,
                          label_visibility="collapsed")
    with ctrl4:
        st.markdown('<div class="control-label">&nbsp;</div>', unsafe_allow_html=True)
        run_btn = st.button("▶  Run", use_container_width=True, type="primary")
    st.markdown('</div>', unsafe_allow_html=True)

    if res["missing"]:
        items = " &nbsp;|&nbsp; ".join(res["missing"])
        st.markdown(f'<div class="warn-banner" style="margin:16px 48px 0;">⚠ Missing: {items}</div>',
                    unsafe_allow_html=True)

    user_idx = int(user_id) - 1
    is_cold  = check_cold_start(user_idx, res["train_df"])
    if is_cold:
        st.markdown(
            f'<div class="warn-banner" style="margin:16px 48px 0;">❄ Cold-start: user {user_id} '
            f'has &lt;5 training ratings. Falling back to Popularity.</div>',
            unsafe_allow_html=True)

    st.markdown('<div class="content-area" style="padding:40px 48px;">', unsafe_allow_html=True)

    if not run_btn:
        st.markdown(
            '<div class="info-banner">▸ Select an algorithm above and press <strong>Run</strong> '
            'to generate personalised recommendations.</div>', unsafe_allow_html=True)

    if run_btn and model_choice != "Compare All":
        eff_key  = "popularity" if is_cold else MODEL_KEYS[model_choice]
        eff_name = "Popularity" if is_cold else model_choice

        if eff_key not in res:
            st.markdown(
                f'<div class="err-banner">✕ Model "{eff_name}" not loaded. '
                'Run the training script first.</div>', unsafe_allow_html=True)
        else:
            metrics = get_metrics_for(eff_name, res.get("comparison_df"))
            if metrics:
                ndcg = metrics.get("NDCG@10", "—")
                hit  = metrics.get("Hit@10",  "—")
                rmse = metrics.get("RMSE",    "—")
                fmt  = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
                st.markdown(f"""
                <div class="stat-row">
                    <div class="stat-card">
                        <div class="stat-label">Model</div>
                        <div class="stat-value" style="font-size:1.6rem;">{eff_name}</div>
                        <div class="stat-unit">algorithm</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">NDCG @ 10</div>
                        <div class="stat-value">{fmt(ndcg)}</div>
                        <div class="stat-unit">normalised discounted CG</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">Hit Rate @ 10</div>
                        <div class="stat-value">{fmt(hit)}</div>
                        <div class="stat-unit">leave-one-out protocol</div>
                    </div>
                    <div class="stat-card">
                        <div class="stat-label">RMSE</div>
                        <div class="stat-value">{fmt(rmse)}</div>
                        <div class="stat-unit">rating scale 1–5</div>
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown(f"""
            <div class="section-heading">
                <h2>Recommendations</h2>
                <span class="sh-tag">user {user_id} &nbsp;·&nbsp; top {top_k} &nbsp;·&nbsp; {eff_name}</span>
            </div>""", unsafe_allow_html=True)

            with st.spinner(""):
                recs = get_recommendations(eff_key, user_idx, top_k, res)

            if recs:
                max_sc = recs[0]["score"]
                html = ""
                for r in recs:
                    pct = int((r["score"] / max_sc) * 100) if max_sc else 0
                    genre_s = r["genres"].replace("|", " · ")[:60]
                    html += f"""
                    <div class="rec-item">
                        <div class="rec-rank">{r['rank']:02d}</div>
                        <div>
                            <div class="rec-title">{r['title']}</div>
                            <div class="rec-genre">{genre_s}</div>
                        </div>
                        <div class="rec-bar"><div class="rec-bar-fill" style="width:{pct}%"></div></div>
                        <div class="rec-score">{r['score']:.4f}</div>
                    </div>"""
                st.markdown(f'<div class="rec-list">{html}</div>', unsafe_allow_html=True)
            else:
                st.markdown('<div class="warn-banner">No recommendations generated.</div>',
                            unsafe_allow_html=True)

            if eff_name in ALGO_INFO:
                title, body = ALGO_INFO[eff_name]
                st.markdown(f"""
                <div class="algo-box" style="margin-top:28px;">
                    <h4>How it works — {title}</h4>
                    <p>{body}</p>
                </div>""", unsafe_allow_html=True)

    if run_btn and model_choice == "Compare All":
        st.markdown(f"""
        <div class="section-heading">
            <h2>All Models</h2>
            <span class="sh-tag">user {user_id} &nbsp;·&nbsp; top 5 each</span>
        </div>""", unsafe_allow_html=True)

        col_data = {}
        for name, key in MODEL_KEYS.items():
            eff_key = "popularity" if is_cold else key
            col_data[name] = get_recommendations(eff_key, user_idx, 5, res) if eff_key in res else []

        html = '<div class="compare-grid">'
        for name, recs in col_data.items():
            html += f'<div class="compare-col"><div class="compare-col-header">{name}</div>'
            if recs:
                for r in recs:
                    html += (f'<div class="compare-item"><strong>{r["rank"]}. {r["title"]}</strong>'
                             f'{r["genres"].split("|")[0]}</div>')
            else:
                html += '<div class="compare-item" style="color:#5a5a7a;">Not trained</div>'
            html += '</div>'
        html += '</div>'
        st.markdown(html, unsafe_allow_html=True)

        bar_path = PLOTS_DIR / "comparison_bar.png"
        if bar_path.exists():
            st.markdown('<div class="gold-divider"></div>', unsafe_allow_html=True)
            st.markdown("""
            <div class="section-heading">
                <h2>Performance Comparison</h2>
                <span class="sh-tag">NDCG@10 &amp; Hit Rate@10</span>
            </div>""", unsafe_allow_html=True)
            st.image(str(bar_path), use_column_width=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="gold-divider" style="margin:0 48px;"></div>', unsafe_allow_html=True)
    tab1, tab2, tab3 = st.tabs(["MODEL METRICS", "EMBEDDING SPACE", "ABOUT"])

    with tab1:
        comp = res.get("comparison_df")
        st.markdown('<div style="padding:24px 48px;">', unsafe_allow_html=True)
        if comp is not None:
            for col in ["NDCG@10", "Hit@10"]:
                if col in comp.columns:
                    comp[col + "_best"] = comp[col] == comp[col].max()
            rows = ""
            for _, row in comp.iterrows():
                ndcg = row.get("NDCG@10", "—"); hit = row.get("Hit@10", "—"); rmse = row.get("RMSE", "—")
                nc = "best" if row.get("NDCG@10_best") else ""
                hc = "best" if row.get("Hit@10_best")  else ""
                fmt = lambda v: f"{v:.4f}" if isinstance(v, float) else str(v)
                rows += f"<tr><td>{row['Model']}</td><td class='{nc}'>{fmt(ndcg)}</td><td class='{hc}'>{fmt(hit)}</td><td>{fmt(rmse)}</td></tr>"
            st.markdown(f"""
            <table class="metrics-table">
                <thead><tr><th>Model</th><th>NDCG@10</th><th>Hit Rate@10</th><th>RMSE</th></tr></thead>
                <tbody>{rows}</tbody>
            </table>
            <p style="margin-top:12px;font-family:'DM Mono',monospace;font-size:0.6rem;color:#3a3a5e;letter-spacing:0.1em;">
                GOLD = best in column &nbsp;·&nbsp; LOO evaluation on 6,040 users
            </p>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-banner">Run python compare_models.py to populate this table.</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab2:
        st.markdown('<div style="padding:24px 48px;">', unsafe_allow_html=True)
        tsne_path = PLOTS_DIR / "embedding_tsne.png"
        if tsne_path.exists():
            st.image(str(tsne_path), use_column_width=True)
            st.markdown("""
            <p style="font-family:'DM Mono',monospace;font-size:0.65rem;color:#5a5a7a;
                       letter-spacing:0.06em;margin-top:12px;line-height:1.8;">
            Each point = one movie &nbsp;·&nbsp; 64-dim NCF embeddings → 2D via t-SNE &nbsp;·&nbsp;
            Colour = primary genre &nbsp;·&nbsp; Clusters emerge from user signals alone — no genre labels seen during training
            </p>""", unsafe_allow_html=True)
        else:
            st.markdown('<div class="info-banner">Run python compare_models.py to generate this plot.</div>',
                        unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div style="padding:24px 48px;">
        <div class="about-block">
        <div style="margin-bottom:24px;">
            <span class="about-pill">NumPy</span><span class="about-pill">PyTorch</span>
            <span class="about-pill">Streamlit</span><span class="about-pill">MovieLens-1M</span>
            <span class="about-pill">Leave-One-Out Eval</span>
        </div>
        <h3>SVD — Matrix Factorisation</h3>
        <p>Predicts rating as <code style="color:#c9a84c;">μ + b_u + b_i + P[u]·Q[i]</code>.
        All parameters learned by hand-written SGD in pure NumPy with L2 regularisation
        and learning-rate decay. No sklearn or torch used — every gradient update is explicit.</p>
        <h3>NCF — Neural Collaborative Filtering</h3>
        <p>Replaces SVD's inner product with a multi-layer perceptron. Embeddings concatenated
        and passed through 4 layers (128→64→32→1) with ReLU and dropout. Trained on implicit
        feedback: each (user, item) pair augmented with 4 random negatives, optimised with BCE loss.</p>
        <h3>NeuMF — Neural Matrix Factorisation</h3>
        <p>Dual-path architecture (He et al., 2017): GMF path (element-wise product) and MLP path
        run in parallel with separate embedding tables. Outputs fused in a final linear layer —
        capturing linear co-occurrence and non-linear interaction patterns simultaneously.</p>
        <h3>Evaluation Protocol</h3>
        <ul>
            <li><strong>Leave-One-Out:</strong> last-rated movie per user held out as positive, paired with 99 random negatives.</li>
            <li><strong>NDCG@10:</strong> discounts by rank — 1/log₂(rank+1) — rewards higher placement.</li>
            <li><strong>Hit Rate@10:</strong> binary — positive in top 10?</li>
            <li><strong>RMSE:</strong> on explicit ratings (meaningful for Popularity and SVD only).</li>
        </ul>
        </div>
        </div>""", unsafe_allow_html=True)


if __name__ == "__main__":
    main()
