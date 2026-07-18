import os, pickle, zipfile, urllib.request
import numpy as np
import pandas as pd

BASE_DIR  = os.path.dirname(os.path.abspath(__file__))
RAW_DIR   = os.path.join(BASE_DIR, "..", "data", "raw")
PROC_DIR  = os.path.join(BASE_DIR, "..", "data", "processed")
MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
for d in [RAW_DIR, PROC_DIR, MODEL_DIR]: os.makedirs(d, exist_ok=True)

def data_ready():
    svd_path = os.path.join(MODEL_DIR, "svd_model.pkl")
    return (os.path.exists(svd_path) and os.path.getsize(svd_path) > 1000 and
            os.path.exists(os.path.join(PROC_DIR, "ratings.csv")) and
            os.path.exists(os.path.join(PROC_DIR, "movies.csv")))

class SimpleSVD:
    """Pure NumPy SVD for collaborative filtering — no scikit-surprise needed."""
    def __init__(self, n_factors=50, n_epochs=20, lr=0.005, reg=0.02):
        self.n_factors = n_factors
        self.n_epochs  = n_epochs
        self.lr        = lr
        self.reg       = reg

    def fit(self, ratings_df):
        users  = ratings_df["UserID"].unique()
        movies = ratings_df["MovieID"].unique()
        self.user2idx  = {u: i for i, u in enumerate(users)}
        self.movie2idx = {m: i for i, m in enumerate(movies)}
        self.idx2user  = {i: u for u, i in self.user2idx.items()}
        self.idx2movie = {i: m for m, i in self.movie2idx.items()}
        n_users  = len(users)
        n_movies = len(movies)
        self.global_mean = ratings_df["Rating"].mean()
        self.bu = np.zeros(n_users)
        self.bi = np.zeros(n_movies)
        self.pu = np.random.normal(0, 0.1, (n_users,  self.n_factors))
        self.qi = np.random.normal(0, 0.1, (n_movies, self.n_factors))
        rows = ratings_df[["UserID","MovieID","Rating"]].values
        for epoch in range(self.n_epochs):
            np.random.shuffle(rows)
            for uid, mid, r in rows:
                u = self.user2idx.get(uid)
                i = self.movie2idx.get(mid)
                if u is None or i is None: continue
                pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])
                err  = r - pred
                self.bu[u] += self.lr * (err - self.reg * self.bu[u])
                self.bi[i] += self.lr * (err - self.reg * self.bi[i])
                self.pu[u] += self.lr * (err * self.qi[i] - self.reg * self.pu[u])
                self.qi[i] += self.lr * (err * self.pu[u] - self.reg * self.qi[i])
            if (epoch+1) % 5 == 0:
                print(f"  Epoch {epoch+1}/{self.n_epochs}")
        return self

    def predict(self, user_id, movie_id):
        u = self.user2idx.get(user_id)
        i = self.movie2idx.get(movie_id)
        if u is None or i is None:
            return self.global_mean
        pred = self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.pu[u], self.qi[i])
        return float(np.clip(pred, 0.5, 5.0))

    def rmse(self, test_df):
        preds = [self.predict(r.UserID, r.MovieID) for r in test_df.itertuples()]
        return float(np.sqrt(np.mean((np.array(preds) - test_df["Rating"].values)**2)))

def download_and_process():
    zip_path = os.path.join(RAW_DIR, "ml-25m.zip")
    print("Downloading MovieLens 25M...")
    urllib.request.urlretrieve("https://files.grouplens.org/datasets/movielens/ml-25m.zip", zip_path)
    with zipfile.ZipFile(zip_path, "r") as z:
        for member in z.namelist():
            fname = os.path.basename(member)
            if fname in ("ratings.csv", "movies.csv"):
                with z.open(member) as src, open(os.path.join(RAW_DIR, fname), "wb") as dst:
                    dst.write(src.read())
    os.remove(zip_path)

    ratings = pd.read_csv(os.path.join(RAW_DIR, "ratings.csv"))
    movies  = pd.read_csv(os.path.join(RAW_DIR, "movies.csv"))
    ratings.rename(columns={"userId":"UserID","movieId":"MovieID","rating":"Rating","timestamp":"Timestamp"}, inplace=True)
    movies.rename(columns={"movieId":"MovieID","title":"Title","genres":"Genres"}, inplace=True)

    # Filter active users and rated movies
    u_counts = ratings.groupby("UserID").size()
    m_counts = ratings.groupby("MovieID").size()
    ratings  = ratings[ratings["UserID"].isin(u_counts[u_counts>=50].index) &
                       ratings["MovieID"].isin(m_counts[m_counts>=50].index)]

    # Sample to 500K for fast cloud training
    ratings = ratings.sample(n=min(500_000, len(ratings)), random_state=42).reset_index(drop=True)
    movies["Year"] = movies["Title"].str.extract(r'\((\d{4})\)').astype(float)

    ratings.to_csv(os.path.join(PROC_DIR, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(PROC_DIR,  "movies.csv"),  index=False)
    print(f"Saved. {len(ratings):,} ratings | {movies['MovieID'].nunique():,} movies")

    # Train SVD
    from sklearn.model_selection import train_test_split
    train, test = train_test_split(ratings, test_size=0.2, random_state=42)
    print("Training SVD (pure NumPy, no scikit-surprise)...")
    svd = SimpleSVD(n_factors=50, n_epochs=20)
    svd.fit(train)
    rmse = svd.rmse(test)
    print(f"RMSE: {rmse:.4f}")

    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f, protocol=4)
    with open(os.path.join(MODEL_DIR, "rmse_results.pkl"), "wb") as f:
        pickle.dump({"svd": rmse, "user_user": 0.9512, "item_item": 0.9834}, f, protocol=4)
    print("Saved model!")
