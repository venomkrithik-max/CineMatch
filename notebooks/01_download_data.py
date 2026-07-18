"""
Step 1: Download MovieLens 25M
Run: python3 01_download_data.py
"""
import os, sys, zipfile, urllib.request

RAW_DIR  = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
os.makedirs(RAW_DIR, exist_ok=True)

if os.path.exists(os.path.join(RAW_DIR, "ratings.csv")):
    print("Already downloaded. Skipping.")
    sys.exit(0)

URL      = "https://files.grouplens.org/datasets/movielens/ml-25m.zip"
ZIP_PATH = os.path.join(RAW_DIR, "ml-25m.zip")

def progress(count, block_size, total_size):
    if total_size > 0:
        pct = min(100, count * block_size * 100 // total_size)
        print(f"\r  {pct}%", end="", flush=True)

print("Downloading MovieLens 25M (~250 MB)...")
urllib.request.urlretrieve(URL, ZIP_PATH, reporthook=progress)
print("\nExtracting...")
with zipfile.ZipFile(ZIP_PATH, "r") as z:
    for member in z.namelist():
        fname = os.path.basename(member)
        if fname in ("ratings.csv", "movies.csv"):
            with z.open(member) as src, open(os.path.join(RAW_DIR, fname), "wb") as dst:
                dst.write(src.read())
os.remove(ZIP_PATH)
print("Done!")
for f in ["ratings.csv", "movies.csv"]:
    size = os.path.getsize(os.path.join(RAW_DIR, f)) // (1024*1024)
    print(f"  {f}: {size} MB")
