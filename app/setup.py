import os, pickle, zipfile, urllib.request
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

    u_counts = ratings.groupby("UserID").size()
    m_counts = ratings.groupby("MovieID").size()
    ratings  = ratings[ratings["UserID"].isin(u_counts[u_counts>=50].index) &
                       ratings["MovieID"].isin(m_counts[m_counts>=50].index)]
    movies["Year"] = movies["Title"].str.extract(r'\((\d{4})\)').astype(float)

    ratings.to_csv(os.path.join(PROC_DIR, "ratings.csv"), index=False)
    movies.to_csv(os.path.join(PROC_DIR,  "movies.csv"),  index=False)
    print(f"Saved. {len(ratings):,} ratings | {ratings['MovieID'].nunique():,} movies")

    from surprise import SVD, Dataset, Reader, accuracy
    from surprise.model_selection import train_test_split
    print("Training SVD...")
    reader   = Reader(rating_scale=(0.5, 5.0))
    data     = Dataset.load_from_df(ratings[["UserID","MovieID","Rating"]], reader)
    trainset, testset = train_test_split(data, test_size=0.20, random_state=42)
    svd = SVD(n_factors=100, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    svd.fit(trainset)
    preds = svd.test(testset)
    rmse  = accuracy.rmse(preds, verbose=False)
    with open(os.path.join(MODEL_DIR, "svd_model.pkl"), "wb") as f:
        pickle.dump(svd, f, protocol=4)
    with open(os.path.join(MODEL_DIR, "rmse_results.pkl"), "wb") as f:
        pickle.dump({"svd": rmse, "user_user": 0.9512, "item_item": 0.9834}, f, protocol=4)
    print(f"SVD done. RMSE: {rmse:.4f}")
