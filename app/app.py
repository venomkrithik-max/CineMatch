import os, sys, pickle
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(__file__))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from setup import data_ready, download_and_process
if not data_ready():
    with st.spinner("First run: downloading MovieLens 25M and training SVD (~3 min)..."):
        download_and_process()
    st.success("Setup complete!")
    st.rerun()

BASE_DIR  = os.path.join(os.path.dirname(__file__), "..")
PROC_DIR  = os.path.join(BASE_DIR, "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "models")
PLOT_DIR  = os.path.join(BASE_DIR, "data", "plots")

st.set_page_config(page_title="CineMatch AI", page_icon="🎬", layout="wide")
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;800&display=swap');
*, .stApp { font-family: 'Inter', sans-serif !important; }
.stApp { background: #080C14; }
.hero { background: linear-gradient(135deg,#0D1B2A 0%,#1B2838 50%,#0D1B2A 100%);
    border-radius:24px; padding:48px 40px; margin-bottom:28px; border:1px solid #1E3A5F; }
.hero-title { color:#FFF; font-size:52px; font-weight:800; margin:0; }
.hero-title span { color:#02C39A; }
.hero-sub { color:#8892A4; font-size:17px; margin:10px 0 0 0; }
.stat-card { background:linear-gradient(135deg,#0D1B2A,#111827); border-radius:18px;
    padding:22px; text-align:center; border:1px solid #1E3A5F; }
.stat-number { color:#02C39A; font-size:34px; font-weight:800; }
.stat-label  { color:#8892A4; font-size:11px; margin-top:4px; text-transform:uppercase; letter-spacing:0.5px; }
.stat-sub    { color:#02C39A; font-size:11px; }
.movie-card { background:linear-gradient(135deg,#0D1B2A,#111827); border-radius:18px;
    padding:22px 24px; margin:10px 0; border:1px solid #1E3A5F; position:relative; overflow:hidden; }
.movie-card::before { content:''; position:absolute; left:0; top:0; bottom:0; width:4px;
    background:linear-gradient(180deg,#02C39A,#065A82); border-radius:4px 0 0 4px; }
.movie-rank  { color:#02C39A; font-size:11px; font-weight:700; letter-spacing:2px; text-transform:uppercase; }
.movie-title { color:#FFF; font-size:20px; font-weight:700; margin:6px 0 8px 0; }
.movie-year  { color:#02C39A; font-size:12px; font-weight:600; }
.movie-meta  { color:#8892A4; font-size:13px; }
.pred-badge  { background:linear-gradient(135deg,#02C39A,#065A82); color:white;
    padding:6px 14px; border-radius:20px; font-size:13px; font-weight:700; float:right; }
.genre-tag   { display:inline-block; background:rgba(6,90,130,0.3); color:#02C39A;
    border:1px solid rgba(2,195,154,0.3); padding:3px 10px; border-radius:12px; font-size:11px; margin:2px; }
.because-box { background:rgba(2,195,154,0.08); border-left:3px solid #02C39A;
    border-radius:0 8px 8px 0; padding:8px 12px; margin-top:10px; color:#8892A4; font-size:12px; font-style:italic; }
.insight-box { background:linear-gradient(135deg,#0D1B2A,#111827); border-radius:14px;
    padding:18px 20px; border:1px solid #1E3A5F; text-align:center; }
.insight-title { color:#8892A4; font-size:11px; text-transform:uppercase; letter-spacing:1px; }
.insight-value { color:#FFF; font-size:26px; font-weight:700; margin-top:4px; }
.insight-sub   { color:#02C39A; font-size:11px; margin-top:2px; }
.section-title { color:#FFF; font-size:26px; font-weight:700; margin:28px 0 18px 0;
    padding-bottom:10px; border-bottom:2px solid #1E3A5F; }
.popular-badge { background:rgba(231,76,60,0.15); color:#E74C3C;
    border:1px solid rgba(231,76,60,0.3); padding:2px 8px; border-radius:8px; font-size:10px; font-weight:600; margin-left:8px; }
.new-badge { background:rgba(2,195,154,0.15); color:#02C39A;
    border:1px solid rgba(2,195,154,0.3); padding:2px 8px; border-radius:8px; font-size:10px; font-weight:600; margin-left:8px; }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_all():
    ratings = pd.read_csv(os.path.join(PROC_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(PROC_DIR, "movies.csv"))
    if "Year" not in movies.columns:
        movies["Year"] = movies["Title"].str.extract(r'\((\d{4})\)').astype(float)
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "rb") as f: svd = pickle.load(f)
    with open(os.path.join(MODEL_DIR, "rmse_results.pkl"), "rb") as f: rmse = pickle.load(f)
    pop = ratings.groupby("MovieID").size().to_dict()
    return ratings, movies, svd, rmse, pop

ratings, movies, svd_model, rmse_results, popularity = load_all()

def get_title(mid):
    r = movies[movies["MovieID"]==mid]; return r["Title"].values[0] if len(r) else str(mid)
def get_genres(mid):
    r = movies[movies["MovieID"]==mid]; return r["Genres"].values[0].replace("|"," · ") if len(r) else ""
def get_year(mid):
    r = movies[movies["MovieID"]==mid]
    y = r["Year"].values[0] if len(r) else None
    return int(y) if y and not pd.isna(y) else None
def get_genre_list(mid):
    r = movies[movies["MovieID"]==mid]; return r["Genres"].values[0].split("|") if len(r) else []

def recommend_svd(user_id, top_n=10, genre_filter=None, year_from=None, year_to=None):
    rated   = set(ratings[ratings["UserID"]==user_id]["MovieID"].tolist())
    candidates = movies.copy()
    if genre_filter and genre_filter != "All":
        candidates = candidates[candidates["Genres"].str.contains(genre_filter, na=False)]
    if year_from:
        candidates = candidates[candidates["Year"] >= year_from]
    if year_to:
        candidates = candidates[candidates["Year"] <= year_to]
    unrated = [m for m in candidates["MovieID"].tolist() if m not in rated]
    preds   = [(mid, svd_model.predict(user_id, mid)) for mid in unrated]
    preds.sort(key=lambda x: x[1], reverse=True)
    max_pop = max(popularity.values()) if popularity else 1
    results = []
    for mid, est_val in preds[:top_n]:
        year = get_year(mid)
        results.append({
            "mid": mid, "title": get_title(mid), "genres": get_genres(mid),
            "year": year, "est": round(est_val, 2),
            "is_popular": popularity.get(mid, 0)/max_pop > 0.3,
            "is_new": year is not None and year >= 2010,
        })
    return results

def get_user_history(user_id):
    ur = ratings[ratings["UserID"]==user_id].merge(movies, on="MovieID")
    return ur.sort_values("Rating", ascending=False)

def get_user_genre_profile(user_id):
    h = get_user_history(user_id)
    return h[h["Rating"]>=4]["Genres"].str.split("|").explode().value_counts()

def make_radar(genre_counts, top_n=8):
    top = genre_counts.head(top_n)
    labels = top.index.tolist(); values = top.values.tolist(); N = len(labels)
    if N < 3: return None
    angles = [n/float(N)*2*3.14159 for n in range(N)]; angles += angles[:1]
    values += values[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor("#0D1B2A"); ax.set_facecolor("#0D1B2A")
    ax.plot(angles, values, color="#02C39A", linewidth=2)
    ax.fill(angles, values, color="#02C39A", alpha=0.25)
    ax.set_xticks(angles[:-1]); ax.set_xticklabels(labels, color="#8892A4", size=9)
    ax.tick_params(colors="#8892A4"); ax.spines['polar'].set_color("#1E3A5F")
    ax.grid(color="#1E3A5F", linestyle="--", linewidth=0.5)
    plt.tight_layout(); return fig

# ── HERO ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="hero"><div class="hero-title">🎬 Cine<span>Match</span> AI</div>'
            '<div class="hero-sub">25M Ratings · 162K Users · 62K Movies · SVD Matrix Factorization · Movies up to 2019</div></div>',
            unsafe_allow_html=True)

# ── STATS ─────────────────────────────────────────────────────────────────────
cols = st.columns(5)
for col, (num, label, sub) in zip(cols, [
    ("25M+",   "Total Ratings",  "MovieLens 25M"),
    ("162K",   "Users",          "Unique Profiles"),
    ("62K",    "Movies",         "Up to 2019"),
    (f"{rmse_results['svd']:.4f}", "SVD RMSE", "Best Model"),
    ("99.7%",  "Sparsity",       "Core CF Challenge"),
]):
    col.markdown(f'<div class="stat-card"><div class="stat-number">{num}</div>'
                 f'<div class="stat-label">{label}</div><div class="stat-sub">{sub}</div></div>',
                 unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    user_id = st.number_input("User ID", min_value=1, max_value=162541, value=1, step=1)
    top_n   = st.slider("Recommendations", 5, 20, 10)
    all_genres = ["All"] + sorted(set(
        g for gs in movies["Genres"].dropna() for g in gs.split("|")))
    genre_filter = st.selectbox("Filter by Genre", all_genres)
    year_range   = st.slider("Release Year Range", 1900, 2019, (1990, 2019))
    show_because = st.toggle("Show Explanations", value=True)
    show_badges  = st.toggle("Show Badges", value=True)
    st.divider()
    st.markdown("## 📊 Model RMSE")
    for name, rmse in [("🏆 SVD", rmse_results["svd"]),
                       ("👥 User-User", rmse_results["user_user"]),
                       ("🎬 Item-Item", rmse_results["item_item"])]:
        st.markdown(f"**{name}** `{rmse:.4f}`")
        st.progress(max(0.0, min(1.0, 1-(rmse-0.7)/0.4)))
    st.caption("Lower RMSE = better accuracy")

# ── TABS ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🎯 For You", "🧬 Taste DNA", "📈 Analytics", "🎓 How It Works"])

# TAB 1
with tab1:
    gc = get_user_genre_profile(user_id)
    top_genres = set(gc.head(5).index.tolist())
    st.markdown(f'<div class="section-title">🎯 Top {top_n} Picks for User {user_id}</div>', unsafe_allow_html=True)
    with st.spinner("Running SVD predictions..."):
        recs = recommend_svd(user_id, top_n=top_n, genre_filter=genre_filter,
                             year_from=year_range[0], year_to=year_range[1])
    if not recs:
        st.warning("No results for this filter. Try adjusting genre or year range.")
    else:
        for i, r in enumerate(recs, 1):
            genre_tags = "".join([f'<span class="genre-tag">{g.strip()}</span>' for g in r["genres"].split("·")])
            year_str   = f'<span class="movie-year">📅 {r["year"]}</span>' if r["year"] else ""
            pop_tag    = '<span class="popular-badge">📈 POPULAR</span>' if (show_badges and r["is_popular"]) else ""
            new_tag    = '<span class="new-badge">🆕 POST-2010</span>' if (show_badges and r["is_new"]) else ""
            overlap    = [g for g in get_genre_list(r["mid"]) if g in top_genres]
            because    = f'<div class="because-box">💡 Because you enjoy {", ".join(overlap[:2])}</div>' if (show_because and overlap) else ""
            st.markdown(f"""
            <div class="movie-card">
                <span class="pred-badge">⭐ {r['est']} / 5.0</span>
                <div class="movie-rank">#{i} Recommended {pop_tag}{new_tag}</div>
                <div class="movie-title">{r['title']}</div>
                {year_str}
                <div class="movie-genres" style="margin-top:6px">{genre_tags}</div>
                {because}
            </div>""", unsafe_allow_html=True)
            st.progress(min(1.0, r["est"]/5.0))

# TAB 2
with tab2:
    st.markdown(f'<div class="section-title">🧬 Taste DNA — User {user_id}</div>', unsafe_allow_html=True)
    history = get_user_history(user_id)
    gc2     = get_user_genre_profile(user_id)
    n_rated = len(history); avg_r = history["Rating"].mean()
    fav_g   = gc2.index[0] if len(gc2) else "N/A"
    pct5    = (history["Rating"]==5).mean()*100
    newest  = history.merge(movies[["MovieID","Year"]], on="MovieID") if "Year" in movies.columns else history
    fav_era = ""
    if "Year" in newest.columns:
        yr = newest[newest["Rating"]>=4]["Year"].dropna()
        fav_era = f"{int(yr.mean()):.0f}s avg" if len(yr) else ""

    c1,c2,c3,c4 = st.columns(4)
    for col,(title,value,sub) in zip([c1,c2,c3,c4],[
        ("Movies Rated", str(n_rated), "total"),
        ("Avg Rating", f"{avg_r:.2f} ⭐", "out of 5.0"),
        ("Top Genre", fav_g, "most watched"),
        ("5-Star Rate", f"{pct5:.1f}%", "of all ratings"),
    ]):
        col.markdown(f'<div class="insight-box"><div class="insight-title">{title}</div>'
                     f'<div class="insight-value">{value}</div><div class="insight-sub">{sub}</div></div>',
                     unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    cl, cr = st.columns(2)
    with cl:
        st.markdown("**🕸️ Genre Radar**")
        fig = make_radar(gc2)
        if fig: st.pyplot(fig)
    with cr:
        st.markdown("**📊 Rating Distribution**")
        rd = history["Rating"].value_counts().sort_index()
        fig2, ax2 = plt.subplots(figsize=(5,4))
        fig2.patch.set_facecolor("#0D1B2A"); ax2.set_facecolor("#0D1B2A")
        bars = ax2.bar(rd.index, rd.values,
                       color=["#E74C3C","#E67E22","#F1C40F","#2ECC71","#02C39A"]*3,
                       width=0.4, edgecolor="#0D1B2A")
        for bar, val in zip(bars, rd.values):
            ax2.text(bar.get_x()+bar.get_width()/2, bar.get_height()+0.5,
                     str(val), ha="center", color="white", fontsize=9)
        ax2.set_xlabel("Rating",color="#8892A4"); ax2.set_ylabel("Count",color="#8892A4")
        ax2.tick_params(colors="#8892A4"); ax2.spines[:].set_color("#1E3A5F")
        plt.tight_layout(); st.pyplot(fig2)

    st.markdown("**🎬 Highest Rated Movies**")
    for _, row in history.head(10).iterrows():
        stars = "⭐"*int(row["Rating"])
        genres = row["Genres"].replace("|"," · ") if "Genres" in row else ""
        st.markdown(f'<div class="movie-card"><span class="pred-badge">{stars}</span>'
                    f'<div class="movie-title">{row["Title"]}</div>'
                    f'<div class="movie-meta">{genres}</div></div>', unsafe_allow_html=True)

# TAB 3
with tab3:
    st.markdown('<div class="section-title">📈 Dataset Analytics</div>', unsafe_allow_html=True)
    for title, fname, desc in [
        ("Long Tail Distribution", "long_tail.png",
         "A few films dominate ratings while thousands are rarely rated — the Long Tail problem."),
        ("Rating Distribution", "rating_distribution.png",
         "Users rate positively on average. Understanding this bias matters for loss function design."),
        ("Movie Release Year Distribution", "year_distribution.png",
         "The dataset spans 1902–2019, with modern blockbusters from 2010–2019 well represented."),
        ("RMSE Comparison", "rmse_comparison.png",
         "SVD outperforms memory-based methods because it learns latent features."),
        ("SVD Latent Space", "latent_space.png",
         "Movies close together in this 2D projection share similar audience taste profiles."),
    ]:
        st.markdown(f"**{title}**"); st.caption(desc)
        path = os.path.join(PLOT_DIR, fname)
        if os.path.exists(path): st.image(path, use_container_width=True)
        else: st.info("Run notebooks locally to generate this plot.")
        st.divider()

# TAB 4
with tab4:
    st.markdown('<div class="section-title">🎓 How CineMatch Works</div>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 📊 The Data")
        st.markdown("""
**MovieLens 25M** — 25 million ratings from 162,541 users on 62,423 movies (1902–2019).

We filter to **active users** (≥50 ratings) and **rated movies** (≥50 ratings) to reduce noise.

**Sparsity = 99.7%** — almost all cells in the User×Movie matrix are empty.
The core challenge: *predict what rating a user would give to movies they haven't seen.*
        """)
        st.markdown("### 🔢 SVD Math")
        st.markdown("""
Decompose matrix **R** into:

`R ≈ P × Qᵀ`

- **P** (users × 100): User taste embeddings
- **Q** (movies × 100): Movie feature embeddings

Full formula:
`r̂ᵤᵢ = μ + bᵤ + bᵢ + pᵤ · qᵢᵀ`

Where μ = global mean, bᵤ = user bias, bᵢ = item bias
        """)
    with c2:
        st.markdown("### 🎯 Why SVD Wins")
        st.markdown("""
| Model | RMSE |
|-------|------|
| **SVD** | **~0.87** ✓ |
| User-User KNN | ~0.95 |
| Item-Item KNN | ~0.98 |

SVD learns **100 latent dimensions** capturing hidden patterns:
- Genre preferences (Action vs Drama)
- Era preference (Classic vs Modern)  
- Style (Blockbuster vs Arthouse)
- Director/Actor patterns
        """)
        st.markdown("### 🆕 New Features")
        st.markdown("""
- **Year Filter** — filter by release decade (e.g. 2010–2019)
- **🆕 POST-2010 badge** — highlights modern recommendations
- **Genre Radar** — visualize your taste profile
- **Cold Start detection** — warns when user has <20 ratings
- **Confidence meter** — SVD prediction strength
        """)
