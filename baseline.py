"""
baseline.py
Popularity-based recommendation baseline.
Score = count(ratings) * mean(rating) per movie.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

np.random.seed(42)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(parents=True, exist_ok=True)


class PopularityRecommender:
    """
    Non-personalised recommender.
    Ranks movies by: popularity_score = n_ratings * mean_rating.
    """

    def __init__(self):
        self.scores: dict  = {}      # movie_idx → popularity score
        self.ranked: list  = []      # movie indices sorted by score (desc)
        self.is_fitted: bool = False

    

    def fit(self, train_df: pd.DataFrame) -> "PopularityRecommender":
        print("Fitting PopularityRecommender …")

        stats = (
            train_df.groupby("movie_idx")["Rating"]
            .agg(count="count", mean="mean")
            .reset_index()
        )
        stats["score"] = stats["count"] * stats["mean"]

        self.scores  = dict(zip(stats["movie_idx"].astype(int), stats["score"]))
        self.ranked  = sorted(self.scores, key=self.scores.get, reverse=True)
        self.is_fitted = True

        print(f"  Fitted on {len(train_df):,} ratings, {len(self.scores):,} unique movies.")
        return self

    

    def predict(self, user_idx: int, movie_idx_list: list) -> list:
        """Return popularity scores for a list of movie indices (user-agnostic)."""
        return [self.scores.get(int(m), 0.0) for m in movie_idx_list]

    def predict_batch(self, user_idx: int, movie_idx_list: list) -> np.ndarray:
        """Same as predict but returns numpy array (for evaluate_loo compatibility)."""
        return np.array(self.predict(user_idx, movie_idx_list))

    def predict_rating(self, user_idx: int, movie_idx: int) -> float:
        """
        Approximate rating prediction: map popularity score to [1, 5] range.
        Used for RMSE computation.
        """
        if not self.scores:
            return 3.0
        max_score = max(self.scores.values()) if self.scores else 1.0
        score = self.scores.get(int(movie_idx), 0.0)
        # Linear mapping: [0, max_score] → [1, 5]
        return 1.0 + 4.0 * (score / max_score) if max_score > 0 else 3.0

    # alias for evaluate_rmse compatibility
    def predict(self, user_idx: int, movie_idx: int) -> float:  # noqa: F811
        if isinstance(movie_idx, (list, np.ndarray)):
            return self.predict_batch(user_idx, movie_idx)
        return self.predict_rating(user_idx, movie_idx)

    def predict_batch(self, user_idx: int, movie_idx_list) -> np.ndarray:  # noqa: F811
        return np.array([self.scores.get(int(m), 0.0) for m in movie_idx_list])

    
    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        train_df: pd.DataFrame = None,
    ) -> list:
        """Return top-n movie indices for a user."""
        if not self.is_fitted:
            raise RuntimeError("Call fit() first.")

        seen = set()
        if exclude_seen and train_df is not None:
            seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())

        recommendations = [m for m in self.ranked if m not in seen]
        return recommendations[:n]

    

    def save(self, path: str = "models/popularity_model.pkl"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scores": self.scores, "ranked": self.ranked}, f)
        print(f"  Saved popularity model → {path}")

    def load(self, path: str = "models/popularity_model.pkl") -> "PopularityRecommender":
        with open(path, "rb") as f:
            data = pickle.load(f)
        self.scores    = data["scores"]
        self.ranked    = data["ranked"]
        self.is_fitted = True
        print(f"  Loaded popularity model ← {path}")
        return self


if __name__ == "__main__":
    import json

    print("=" * 60)
    print("Training Popularity Recommender")
    print("=" * 60)

    # Load processed data
    train_df = pd.read_csv("data/processed/train_df.csv")
    with open("data/processed/config.json") as f:
        config = json.load(f)

    model = PopularityRecommender()
    model.fit(train_df)
    model.save("models/popularity_model.pkl")

    # Quick sanity check
    recs = model.recommend(user_idx=0, n=10, exclude_seen=True, train_df=train_df)
    print(f"\nTop-10 recommendations for user 0: {recs}")
    print("\n✓ Popularity baseline complete.\n")
