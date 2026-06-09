"""
data_loader.py
Loads, processes, and splits the MovieLens-1M dataset.
Saves processed files to data/processed/.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

RAW_DIR  = Path("data/raw")
PROC_DIR = Path("data/processed")
PROC_DIR.mkdir(parents=True, exist_ok=True)


class MovieLensLoader:
    """Loads and preprocesses the MovieLens-1M dataset."""

    def __init__(self):
        self.ratings  = None
        self.movies   = None
        self.users    = None
        self.user2idx = {}
        self.movie2idx = {}
        self.idx2movie = {}
        self.n_users  = 0
        self.n_items  = 0


    def load_raw(self):
        print("=" * 60)
        print("Loading raw MovieLens-1M data …")

        self.ratings = pd.read_csv(
            RAW_DIR / "ratings.dat",
            sep="::",
            engine="python",
            header=None,
            names=["UserID", "MovieID", "Rating", "Timestamp"],
            encoding="latin-1",
        )

        self.movies = pd.read_csv(
            RAW_DIR / "movies.dat",
            sep="::",
            engine="python",
            header=None,
            names=["MovieID", "Title", "Genres"],
            encoding="latin-1",
        )

        self.users = pd.read_csv(
            RAW_DIR / "users.dat",
            sep="::",
            engine="python",
            header=None,
            names=["UserID", "Gender", "Age", "Occupation", "Zip"],
            encoding="latin-1",
        )

        print(f"  Ratings : {len(self.ratings):,}")
        print(f"  Movies  : {len(self.movies):,}")
        print(f"  Users   : {len(self.users):,}")
        return self


    def remap_ids(self):
        print("\nRemapping UserID / MovieID to 0-indexed integers …")

        unique_users  = sorted(self.ratings["UserID"].unique())
        unique_movies = sorted(self.ratings["MovieID"].unique())

        self.user2idx  = {uid: idx for idx, uid in enumerate(unique_users)}
        self.movie2idx = {mid: idx for idx, mid in enumerate(unique_movies)}
        self.idx2movie = {idx: mid for mid, idx in self.movie2idx.items()}

        self.ratings["user_idx"]  = self.ratings["UserID"].map(self.user2idx)
        self.ratings["movie_idx"] = self.ratings["MovieID"].map(self.movie2idx)

        self.n_users = len(unique_users)
        self.n_items = len(unique_movies)

        print(f"  n_users = {self.n_users}, n_items = {self.n_items}")
        return self


    def split_data(self, train_ratio=0.8, val_ratio=0.1):
        """Stratified split: every user has ratings in all three partitions."""
        print("\nSplitting data (stratified by user) 80/10/10 …")

        train_rows, val_rows, test_rows = [], [], []

        for _, group in self.ratings.groupby("user_idx"):
            group = group.sample(frac=1, random_state=42)  # shuffle within user
            n = len(group)
            n_train = max(1, int(n * train_ratio))
            n_val   = max(1, int(n * val_ratio))

            train_rows.append(group.iloc[:n_train])
            val_rows.append(group.iloc[n_train : n_train + n_val])
            test_rows.append(group.iloc[n_train + n_val :])

        train_df = pd.concat(train_rows).reset_index(drop=True)
        val_df   = pd.concat(val_rows).reset_index(drop=True)
        test_df  = pd.concat(test_rows).reset_index(drop=True)

        # drop users with no test ratings
        test_df = test_df[test_df.groupby("user_idx")["user_idx"].transform("count") >= 1]

        print(f"  Train : {len(train_df):,}  Val : {len(val_df):,}  Test : {len(test_df):,}")
        return train_df, val_df, test_df


    def build_loo_test(self, train_df, n_negatives=99):
        """
        For each user: last rated item (by timestamp) = positive.
        Sample n_negatives random items the user has NOT rated.
        """
        print(f"\nBuilding leave-one-out test set (1 pos + {n_negatives} neg per user) …")

        sorted_ratings = self.ratings.sort_values(["user_idx", "Timestamp"])
        last_items = sorted_ratings.groupby("user_idx")["movie_idx"].last().reset_index()
        last_items.columns = ["user_idx", "pos_item"]

        user_seen = train_df.groupby("user_idx")["movie_idx"].apply(set).to_dict()

        all_items = set(range(self.n_items))
        rows = []
        rng  = np.random.default_rng(42)

        for _, row in last_items.iterrows():
            u    = int(row["user_idx"])
            pos  = int(row["pos_item"])
            seen = user_seen.get(u, set())
            neg_pool = list(all_items - seen - {pos})

            if len(neg_pool) < n_negatives:
                neg_items = neg_pool  # fallback if not enough negatives
            else:
                neg_items = rng.choice(neg_pool, size=n_negatives, replace=False).tolist()

            rows.append({"user_idx": u, "pos_item": pos, "neg_items": json.dumps(neg_items)})

        loo_df = pd.DataFrame(rows)
        print(f"  LOO test users : {len(loo_df):,}")
        return loo_df


    def save(self, train_df, val_df, test_df, loo_df):
        print("\nSaving processed files to data/processed/ …")

        train_df.to_csv(PROC_DIR / "train_df.csv",   index=False)
        val_df.to_csv(  PROC_DIR / "val_df.csv",     index=False)
        test_df.to_csv( PROC_DIR / "test_df.csv",    index=False)
        loo_df.to_csv(  PROC_DIR / "loo_test.csv",   index=False)

        movies_proc = self.movies.copy()
        movies_proc["movie_idx"] = movies_proc["MovieID"].map(self.movie2idx)
        movies_proc.dropna(subset=["movie_idx"], inplace=True)
        movies_proc["movie_idx"] = movies_proc["movie_idx"].astype(int)
        movies_proc.to_csv(PROC_DIR / "movies_processed.csv", index=False)

        with open(PROC_DIR / "user2idx.pkl",  "wb") as f:
            pickle.dump(self.user2idx, f)
        with open(PROC_DIR / "movie2idx.pkl", "wb") as f:
            pickle.dump(self.movie2idx, f)
        with open(PROC_DIR / "idx2movie.pkl", "wb") as f:
            pickle.dump(self.idx2movie, f)

        config = {"n_users": self.n_users, "n_items": self.n_items}
        with open(PROC_DIR / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        print("  Saved: train_df.csv, val_df.csv, test_df.csv, loo_test.csv")
        print("  Saved: movies_processed.csv")
        print("  Saved: user2idx.pkl, movie2idx.pkl, idx2movie.pkl, config.json")


    def print_stats(self):
        n_ratings = len(self.ratings)
        sparsity  = 1.0 - n_ratings / (self.n_users * self.n_items)
        avg_per_user = n_ratings / self.n_users

        print("\n" + "=" * 60)
        print("DATASET STATISTICS")
        print("=" * 60)
        print(f"  Total ratings      : {n_ratings:,}")
        print(f"  Unique users       : {self.n_users:,}")
        print(f"  Unique movies      : {self.n_items:,}")
        print(f"  Sparsity           : {sparsity * 100:.2f}%")
        print(f"  Avg ratings/user   : {avg_per_user:.1f}")
        print(f"  Rating scale       : {self.ratings['Rating'].min()} – {self.ratings['Rating'].max()}")
        print("=" * 60)


    def run(self):
        self.load_raw()
        self.remap_ids()
        self.print_stats()
        train_df, val_df, test_df = self.split_data()
        loo_df = self.build_loo_test(train_df)
        self.save(train_df, val_df, test_df, loo_df)
        print("\n✓ Data pipeline complete.\n")
        return train_df, val_df, test_df, loo_df


if __name__ == "__main__":
    loader = MovieLensLoader()
    loader.run()
