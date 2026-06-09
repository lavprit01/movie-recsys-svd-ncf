"""
evaluate.py
Standalone evaluation functions for recommendation models.
Metrics: RMSE, Hit Rate@K, NDCG@K (leave-one-out protocol).
"""

import json
import math
import numpy as np
import pandas as pd
from tqdm import tqdm

np.random.seed(42)



def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Root Mean Squared Error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))



def hit_rate_at_k(recommended_list, ground_truth_item, k: int = 10) -> int:
    """
    1 if ground_truth_item appears in the top-k of recommended_list, else 0.
    recommended_list: ordered list of item indices (best first).
    """
    return int(ground_truth_item in recommended_list[:k])


def ndcg_at_k(recommended_list, ground_truth_item, k: int = 10) -> float:
    """
    Normalised Discounted Cumulative Gain @ K.
    Only one relevant item (the positive), so ideal DCG = 1.
    Returns 0.0 if positive not in top-k.
    """
    top_k = recommended_list[:k]
    if ground_truth_item not in top_k:
        return 0.0
    rank = top_k.index(ground_truth_item) + 1  # 1-indexed
    return 1.0 / math.log2(rank + 1)



def evaluate_loo(model, loo_test_df: pd.DataFrame, idx2movie: dict,
                 k: int = 10, sample_frac: float = 1.0) -> dict:
    """
    Leave-one-out evaluation.

    model must implement:
        predict_batch(user_idx: int, movie_idx_list: list) -> np.ndarray of scores

    loo_test_df columns: user_idx, pos_item, neg_items (JSON list of ints)

    Returns: {'ndcg@K': float, 'hit@K': float}
    """
    if sample_frac < 1.0:
        loo_test_df = loo_test_df.sample(frac=sample_frac, random_state=42)

    hits, ndcgs = [], []

    for _, row in tqdm(loo_test_df.iterrows(), total=len(loo_test_df),
                       desc=f"LOO eval (k={k})", leave=False):
        user   = int(row["user_idx"])
        pos    = int(row["pos_item"])
        negs   = json.loads(row["neg_items"])
        items  = [pos] + negs

        scores  = model.predict_batch(user, items)
        ranked  = [items[i] for i in np.argsort(scores)[::-1]]

        hits.append(hit_rate_at_k(ranked, pos, k))
        ndcgs.append(ndcg_at_k(ranked, pos, k))

    return {
        f"ndcg@{k}": float(np.mean(ndcgs)),
        f"hit@{k}":  float(np.mean(hits)),
    }


def evaluate_rmse(model, test_df: pd.DataFrame) -> float:
    """
    RMSE evaluation on explicit-rating test set.
    model must implement:
        predict(user_idx: int, movie_idx: int) -> float
    """
    preds, truths = [], []
    for _, row in tqdm(test_df.iterrows(), total=len(test_df),
                       desc="RMSE eval", leave=False):
        preds.append(model.predict(int(row["user_idx"]), int(row["movie_idx"])))
        truths.append(float(row["Rating"]))
    return rmse(np.array(truths), np.array(preds))



if __name__ == "__main__":
    # Quick unit tests
    y_true = np.array([4.0, 3.0, 5.0, 2.0])
    y_pred = np.array([3.8, 3.2, 4.5, 2.1])
    print(f"RMSE test          : {rmse(y_true, y_pred):.4f}  (expected ~0.23)")

    rec  = [10, 5, 3, 7, 1, 8, 2, 9, 6, 4]
    print(f"Hit@10  (pos=5)    : {hit_rate_at_k(rec, 5, 10)}  (expected 1)")
    print(f"Hit@10  (pos=99)   : {hit_rate_at_k(rec, 99, 10)} (expected 0)")
    print(f"NDCG@10 (pos=5)    : {ndcg_at_k(rec, 5, 10):.4f}  (pos at rank 2 → 1/log2(3)≈0.6309)")
    print(f"NDCG@10 (pos=10)   : {ndcg_at_k(rec, 10, 10):.4f} (pos at rank 10 → 1/log2(11)≈0.2895)")
    print(f"NDCG@5  (pos=5)    : {ndcg_at_k(rec, 5, 5):.4f}   (pos at rank 2)")
    print(f"NDCG@5  (pos=4)    : {ndcg_at_k(rec, 4, 5):.4f}   (pos=4 not in top 5)")
    print("\n✓ evaluate.py unit tests passed.")
