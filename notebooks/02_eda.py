"""
Step 2: EDA + data preparation
Run: python3 02_eda.py
"""
import os, pickle
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PLOT_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "plots")
for d in [PROC_DIR, PLOT_DIR]: os.makedirs(d, exist_ok=True)

print("Loading data...")
ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
movies  = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
ratings.rename(columns={"userId":"UserID","movieId":"MovieID","rating":"Rating","timestamp":"Timestamp"}, inplace=True)
movies.rename(columns={"movieId":"MovieID","title":"Title","genres":"Genres"}, inplace=True)
print(f"  {len(ratings):,} ratings | {ratings['UserID'].nunique():,} users | {len(movies):,} movies")

# Filter: users with >=50 ratings, movies with >=50 ratings
print("Filtering active users and rated movies...")
u_counts = ratings.groupby("UserID").size()
m_counts = ratings.groupby("MovieID").size()
ratings  = ratings[ratings["UserID"].isin(u_counts[u_counts>=50].index) &
                   ratings["MovieID"].isin(m_counts[m_counts>=50].index)]
print(f"  After filter: {len(ratings):,} ratings | {ratings['UserID'].nunique():,} users | {ratings['MovieID'].nunique():,} movies")

# Extract year from title
movies["Year"] = movies["Title"].str.extract(r'\((\d{4})\)').astype(float)

# Show newer movies in dataset
print("\nSample of movies from 2010-2019 in dataset:")
newer = movies[(movies["Year"]>=2010) & (movies["MovieID"].isin(ratings["MovieID"].unique()))]
print(newer[["Title","Year"]].dropna().sort_values("Year", ascending=False).head(15).to_string(index=False))

# Long tail
rpm = ratings.groupby("MovieID").size().sort_values(ascending=False).reset_index()
rpm.columns = ["MovieID","RatingCount"]
fig, ax = plt.subplots(figsize=(12,5))
ax.bar(range(len(rpm)), rpm["RatingCount"], color="#065A82", width=1.0)
ax.set_xlabel("Movies (ranked by popularity)", fontsize=12)
ax.set_ylabel("Number of Ratings", fontsize=12)
ax.set_title("Long Tail Distribution — MovieLens 25M", fontsize=13)
ax.axvline(x=100, color="#02C39A", linestyle="--", linewidth=1.5, label="Top 100 movies")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "long_tail.png"), dpi=150); plt.close()

# Rating distribution
fig, axes = plt.subplots(1, 2, figsize=(12,5))
rc = ratings["Rating"].value_counts().sort_index()
axes[0].bar(rc.index, rc.values, color="#1C7293", edgecolor="white", width=0.4)
axes[0].set_title("Rating Distribution", fontsize=13)
axes[0].set_xlabel("Rating"); axes[0].set_ylabel("Count")
rpu = ratings.groupby("UserID").size()
axes[1].hist(rpu, bins=50, color="#02C39A", edgecolor="white")
axes[1].set_title(f"Ratings per User (avg: {rpu.mean():.0f})", fontsize=13)
axes[1].set_xlabel("Ratings per User"); axes[1].set_ylabel("Users")
plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "rating_distribution.png"), dpi=150); plt.close()

# Year distribution of rated movies
year_counts = movies[movies["MovieID"].isin(ratings["MovieID"].unique())]["Year"].dropna()
fig, ax = plt.subplots(figsize=(12,5))
ax.hist(year_counts, bins=50, color="#065A82", edgecolor="white")
ax.set_xlabel("Release Year", fontsize=12); ax.set_ylabel("Number of Movies", fontsize=12)
ax.set_title("Movie Release Year Distribution — Includes Films Up to 2019", fontsize=13)
ax.axvline(x=2010, color="#02C39A", linestyle="--", linewidth=1.5, label="Post-2010 movies")
ax.legend(); plt.tight_layout()
plt.savefig(os.path.join(PLOT_DIR, "year_distribution.png"), dpi=150); plt.close()

# Sparsity
n_users  = ratings["UserID"].nunique()
n_movies = ratings["MovieID"].nunique()
sparsity = (1 - len(ratings) / (n_users * n_movies)) * 100
print(f"\nSparsity: {sparsity:.2f}%")
print(f"Saved 3 plots to {os.path.abspath(PLOT_DIR)}")

ratings.to_csv(os.path.join(PROC_DIR, "ratings.csv"), index=False)
movies.to_csv(os.path.join(PROC_DIR,  "movies.csv"),  index=False)
print("Saved processed data. Week 1 done!")
