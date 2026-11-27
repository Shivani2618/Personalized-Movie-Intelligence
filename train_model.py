"""Train a memory-efficient sparse TF-IDF recommendation model."""
from pathlib import Path
import pickle
import numpy as np
import pandas as pd
from scipy.sparse import save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"


def main():
    movies_path = DATA / "combined_movies.csv"
    if not movies_path.exists():
        raise FileNotFoundError("Run: python prepare_dataset.py")
    movies = pd.read_csv(movies_path, low_memory=False).fillna("")
    tags = movies["tags"].astype(str)

    vectorizer = TfidfVectorizer(
        max_features=60000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.98,
        stop_words="english",
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(tags).tocsr()

    save_npz(DATA / "tfidf_matrix.npz", matrix, compressed=True)
    with open(DATA / "tfidf_vectorizer.pkl", "wb") as file:
        pickle.dump(vectorizer, file, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Model trained for {len(movies):,} movies.")
    print(f"Sparse matrix: {matrix.shape[0]:,} rows × {matrix.shape[1]:,} features")
    print(f"Non-zero values: {matrix.nnz:,}")
    print(f"Saved sparse model to {DATA}")


if __name__ == "__main__":
    main()
