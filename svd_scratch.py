"""
svd_scratch.py
Matrix-Factorisation recommender built from scratch with NumPy.
Model: r̂(u,i) = μ + b_u + b_i + P[u] · Q[i]
Trained with SGD + L2 regularisation + learning-rate decay.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm

np.random.seed(42)

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("plots")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)


class SVDRecommender:
    """
    Biased Matrix Factorisation via SGD.

    Parameters
    ----------
    n_factors : int    – latent dimension for P and Q
    lr        : float  – initial learning rate
    reg       : float  – L2 regularisation coefficient
    n_epochs  : int    – number of full passes over training data
    lr_decay  : float  – multiplicative LR decay applied each epoch
    """

    def __init__(
        self,
        n_factors: int   = 64,
        lr: float        = 0.005,
        reg: float       = 0.02,
        n_epochs: int    = 25,
        lr_decay: float  = 0.96,
    ):
        self.n_factors = n_factors
        self.lr        = lr
        self.reg       = reg
        self.n_epochs  = n_epochs
        self.lr_decay  = lr_decay

        # Learned parameters 
        self.global_mean: float = 0.0
        self.b_u: np.ndarray   = None   # (n_users,)
        self.b_i: np.ndarray   = None   # (n_items,)
        self.P:   np.ndarray   = None   # (n_users, n_factors)
        self.Q:   np.ndarray   = None   # (n_items, n_factors)

        # Best-model snapshots
        self._best_val_rmse: float = float("inf")
        self._best_weights: dict   = {}

        self.history: dict = {"train_rmse": [], "val_rmse": []}


    def _init_params(self, n_users: int, n_items: int, global_mean: float):
        self.global_mean = global_mean
        self.b_u = np.zeros(n_users)
        self.b_i = np.zeros(n_items)
        self.P   = np.random.normal(0, 0.01, (n_users, self.n_factors))
        self.Q   = np.random.normal(0, 0.01, (n_items, self.n_factors))


    def _predict_one(self, u: int, i: int) -> float:
        return self.global_mean + self.b_u[u] + self.b_i[i] + self.P[u] @ self.Q[i]


    def _run_epoch(self, users: np.ndarray, items: np.ndarray,
                   ratings: np.ndarray, lr: float) -> float:
        """One SGD pass over all triplets. Returns train RMSE."""
        sq_errors = []
        idx = np.random.permutation(len(ratings))

        for k in idx:
            u, i, r = users[k], items[k], ratings[k]
            pred  = self._predict_one(u, i)
            err   = r - pred

            # Gradient updates
            self.b_u[u] += lr * (err - self.reg * self.b_u[u])
            self.b_i[i] += lr * (err - self.reg * self.b_i[i])
            pu = self.P[u].copy()
            qi = self.Q[i].copy()
            self.P[u] += lr * (err * qi - self.reg * pu)
            self.Q[i] += lr * (err * pu - self.reg * qi)

            sq_errors.append(err ** 2)

        return float(np.sqrt(np.mean(sq_errors)))


    def _val_rmse(self, val_df: pd.DataFrame) -> float:
        preds  = self._predict_batch_df(val_df)
        truths = val_df["Rating"].values.astype(float)
        return float(np.sqrt(np.mean((truths - preds) ** 2)))

    def _predict_batch_df(self, df: pd.DataFrame) -> np.ndarray:
        u = df["user_idx"].values
        i = df["movie_idx"].values
        return (
            self.global_mean
            + self.b_u[u]
            + self.b_i[i]
            + np.sum(self.P[u] * self.Q[i], axis=1)
        )


    def fit(self, train_df: pd.DataFrame, val_df: pd.DataFrame) -> "SVDRecommender":
        print("=" * 60)
        print(f"Training SVD (factors={self.n_factors}, epochs={self.n_epochs})")
        print("=" * 60)

        n_users = int(train_df["user_idx"].max()) + 1
        n_items = int(train_df["movie_idx"].max()) + 1
        global_mean = float(train_df["Rating"].mean())

        self._init_params(n_users, n_items, global_mean)

        users   = train_df["user_idx"].values.astype(int)
        items   = train_df["movie_idx"].values.astype(int)
        ratings = train_df["Rating"].values.astype(float)

        current_lr = self.lr

        for epoch in tqdm(range(1, self.n_epochs + 1), desc="SVD epochs"):
            train_rmse = self._run_epoch(users, items, ratings, current_lr)
            val_rmse   = self._val_rmse(val_df)

            self.history["train_rmse"].append(train_rmse)
            self.history["val_rmse"].append(val_rmse)

            # Save best weights
            if val_rmse < self._best_val_rmse:
                self._best_val_rmse = val_rmse
                self._best_weights  = {
                    "b_u": self.b_u.copy(),
                    "b_i": self.b_i.copy(),
                    "P":   self.P.copy(),
                    "Q":   self.Q.copy(),
                }

            tqdm.write(
                f"  Epoch {epoch:02d}/{self.n_epochs} | "
                f"LR={current_lr:.5f} | "
                f"Train RMSE={train_rmse:.4f} | "
                f"Val RMSE={val_rmse:.4f}"
                + ("  ✓ best" if val_rmse == self._best_val_rmse else "")
            )

            current_lr *= self.lr_decay

        # Restore best weights
        self.b_u = self._best_weights["b_u"]
        self.b_i = self._best_weights["b_i"]
        self.P   = self._best_weights["P"]
        self.Q   = self._best_weights["Q"]
        print(f"\nBest Val RMSE : {self._best_val_rmse:.4f}")
        return self


    def predict(self, user_idx: int, movie_idx: int) -> float:
        """Predict rating for a single (user, item) pair."""
        return float(self._predict_one(user_idx, movie_idx))

    def predict_batch(self, user_idx: int, movie_idx_list) -> np.ndarray:
        """Predict scores for one user and a list of items."""
        idxs = np.asarray(movie_idx_list, dtype=int)
        return (
            self.global_mean
            + self.b_u[user_idx]
            + self.b_i[idxs]
            + self.P[user_idx] @ self.Q[idxs].T
        )

    def recommend(
        self,
        user_idx: int,
        n: int = 10,
        exclude_seen: bool = True,
        train_df: pd.DataFrame = None,
    ) -> list:
        """Return top-n movie indices for a user."""
        all_items = np.arange(len(self.b_i))

        if exclude_seen and train_df is not None:
            seen = set(train_df[train_df["user_idx"] == user_idx]["movie_idx"].tolist())
            all_items = np.array([m for m in all_items if m not in seen])

        scores = self.predict_batch(user_idx, all_items)
        top_n_idx = np.argsort(scores)[::-1][:n]
        return all_items[top_n_idx].tolist()


    def save(self, path: str = "models/svd_model.npz"):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            global_mean=np.array([self.global_mean]),
            b_u=self.b_u,
            b_i=self.b_i,
            P=self.P,
            Q=self.Q,
            train_rmse=np.array(self.history["train_rmse"]),
            val_rmse=np.array(self.history["val_rmse"]),
        )
        print(f"  Saved SVD model → {path}")

    def load(self, path: str = "models/svd_model.npz") -> "SVDRecommender":
        data = np.load(path)
        self.global_mean = float(data["global_mean"][0])
        self.b_u  = data["b_u"]
        self.b_i  = data["b_i"]
        self.P    = data["P"]
        self.Q    = data["Q"]
        self.history = {
            "train_rmse": data["train_rmse"].tolist(),
            "val_rmse":   data["val_rmse"].tolist(),
        }
        print(f"  Loaded SVD model ← {path}")
        return self


    def plot_training_curve(self, save_path: str = "plots/svd_training_curve.png"):
        fig, ax = plt.subplots(figsize=(8, 5))
        epochs = range(1, len(self.history["train_rmse"]) + 1)
        ax.plot(epochs, self.history["train_rmse"], "b-o", markersize=4, label="Train RMSE")
        ax.plot(epochs, self.history["val_rmse"],   "r-o", markersize=4, label="Val RMSE")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("RMSE")
        ax.set_title("SVD Training Curve")
        ax.legend()
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Saved training curve → {save_path}")



if __name__ == "__main__":
    print("=" * 60)
    print("SVD Recommender — Training")
    print("=" * 60)

    train_df = pd.read_csv("data/processed/train_df.csv")
    val_df   = pd.read_csv("data/processed/val_df.csv")

    model = SVDRecommender(
        n_factors=64,
        lr=0.005,
        reg=0.02,
        n_epochs=25,
        lr_decay=0.96,
    )
    model.fit(train_df, val_df)
    model.save("models/svd_model.npz")
    model.plot_training_curve("plots/svd_training_curve.png")

    # Quick sanity check
    recs = model.recommend(user_idx=0, n=10, exclude_seen=True, train_df=train_df)
    print(f"\nTop-10 for user 0: {recs}")
    print("\n✓ SVD training complete.\n")
