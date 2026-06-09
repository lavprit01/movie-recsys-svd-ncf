"""
compare_models.py
Loads all saved models, evaluates on the full LOO test set,
prints a comparison table, and generates comparison plots.

Outputs:
  plots/model_comparison.csv
  plots/comparison_bar.png
  plots/embedding_tsne.png
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import torch
from pathlib import Path
from sklearn.manifold import TSNE
from tqdm import tqdm

from baseline    import PopularityRecommender
from svd_scratch import SVDRecommender
from ncf_model   import NCF, NeuMF
from evaluate    import evaluate_loo, evaluate_rmse

np.random.seed(42)
torch.manual_seed(42)

PROC_DIR   = Path("data/processed")
MODELS_DIR = Path("models")
PLOTS_DIR  = Path("plots")
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
K      = 10

TOP_GENRES = ["Action", "Comedy", "Drama", "Thriller", "Romance",
              "Sci-Fi", "Adventure", "Animation"]



class NCFWrapper:
    def __init__(self, model, device):
        self.model  = model.to(device)
        self.device = device
        self.model.eval()

    def predict_batch(self, user_idx: int, movie_idx_list) -> np.ndarray:
        users  = torch.tensor([user_idx] * len(movie_idx_list), dtype=torch.long).to(self.device)
        items  = torch.tensor(list(movie_idx_list),             dtype=torch.long).to(self.device)
        with torch.no_grad():
            return self.model(users, items).squeeze(-1).cpu().numpy()

    def predict(self, user_idx: int, movie_idx: int) -> float:
        return float(self.predict_batch(user_idx, [movie_idx])[0])



def load_all_models(n_users: int, n_items: int):
    print("Loading all models …")
    models = {}

    # Popularity
    pop = PopularityRecommender()
    pop.load("models/popularity_model.pkl")
    models["Popularity"] = pop

    # SVD
    svd = SVDRecommender()
    svd.load("models/svd_model.npz")
    models["SVD"] = svd

    # NCF
    ncf = NCF(n_users, n_items, embed_dim=64)
    ncf.load_state_dict(torch.load("models/ncf_best.pt", map_location=DEVICE))
    models["NCF"] = NCFWrapper(ncf, DEVICE)

    # NeuMF
    neumf = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32)
    neumf.load_state_dict(torch.load("models/neumf_best.pt", map_location=DEVICE))
    models["NeuMF"] = NCFWrapper(neumf, DEVICE)

    print("  All models loaded.\n")
    return models


def run_evaluation(models: dict, loo_test_df: pd.DataFrame, test_df: pd.DataFrame) -> pd.DataFrame:
    results = []

    for name, model in models.items():
        print(f"\nEvaluating {name} …")

        loo_metrics = evaluate_loo(model, loo_test_df, idx2movie={}, k=K, sample_frac=1.0)
        test_rmse   = evaluate_rmse(model, test_df)

        row = {
            "Model":   name,
            f"NDCG@{K}": round(loo_metrics[f"ndcg@{K}"], 4),
            f"Hit@{K}":  round(loo_metrics[f"hit@{K}"],  4),
            "RMSE":    round(test_rmse, 4),
        }
        results.append(row)
        print(f"  {name}: NDCG@{K}={row[f'NDCG@{K}']:.4f}  "
              f"Hit@{K}={row[f'Hit@{K}']:.4f}  RMSE={row['RMSE']:.4f}")

    return pd.DataFrame(results)



def plot_comparison_bar(results_df: pd.DataFrame, save_path: str):
    models    = results_df["Model"].tolist()
    ndcg_vals = results_df[f"NDCG@{K}"].tolist()
    hit_vals  = results_df[f"Hit@{K}"].tolist()

    x      = np.arange(len(models))
    width  = 0.35
    colors = ["#4C72B0", "#DD8452"]

    fig, ax = plt.subplots(figsize=(10, 6))
    bars1 = ax.bar(x - width / 2, ndcg_vals, width, label=f"NDCG@{K}", color=colors[0])
    bars2 = ax.bar(x + width / 2, hit_vals,  width, label=f"Hit@{K}",  color=colors[1])

    ax.bar_label(bars1, fmt="%.3f", padding=3, fontsize=9)
    ax.bar_label(bars2, fmt="%.3f", padding=3, fontsize=9)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score",  fontsize=12)
    ax.set_title(f"Model Comparison — NDCG@{K} and Hit Rate@{K}", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=11)
    ax.set_ylim(0, max(max(ndcg_vals), max(hit_vals)) * 1.20)
    ax.legend(fontsize=11)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved bar chart → {save_path}")



def _get_first_genre(genres_str: str, top_genres: list) -> str:
    for g in genres_str.split("|"):
        if g in top_genres:
            return g
    return "Other"


def plot_embedding_tsne(ncf_model_raw: NCF, movies_df: pd.DataFrame,
                        idx2movie: dict, save_path: str):
    print("\nComputing t-SNE of NCF item embeddings …")
    ncf_model_raw.eval()

    # Extract embeddings
    with torch.no_grad():
        embeddings = ncf_model_raw.item_embedding.weight.cpu().numpy()   # (n_items, embed_dim)

    # Map movie_idx → genre
    genre_map = {}
    for _, row in movies_df.iterrows():
        midx = int(row["movie_idx"]) if not pd.isna(row.get("movie_idx", float("nan"))) else -1
        if midx >= 0:
            genre_map[midx] = _get_first_genre(str(row["Genres"]), TOP_GENRES)

    genres = [genre_map.get(i, "Other") for i in range(len(embeddings))]

    # t-SNE 
    n_total = len(embeddings)
    max_pts = 3000
    if n_total > max_pts:
        idx = np.random.choice(n_total, max_pts, replace=False)
        embeddings_sub = embeddings[idx]
        genres_sub     = [genres[i] for i in idx]
    else:
        embeddings_sub = embeddings
        genres_sub     = genres

    print(f"  Running t-SNE on {len(embeddings_sub)} item embeddings …")
    tsne = TSNE(n_components=2, perplexity=40, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(embeddings_sub)

    # Plot
    all_genres  = TOP_GENRES + ["Other"]
    palette     = plt.cm.tab10.colors
    genre2color = {g: palette[i % len(palette)] for i, g in enumerate(all_genres)}

    fig, ax = plt.subplots(figsize=(12, 8))

    for genre in all_genres:
        mask = [g == genre for g in genres_sub]
        if not any(mask):
            continue
        pts = coords[np.array(mask)]
        ax.scatter(pts[:, 0], pts[:, 1], s=5, alpha=0.5,
                   color=genre2color[genre], label=genre)

    legend_handles = [
        mpatches.Patch(color=genre2color[g], label=g)
        for g in all_genres
        if any(g == gg for gg in genres_sub)
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right",
              framealpha=0.8, markerscale=2)
    ax.set_title("t-SNE of NCF Item Embeddings (coloured by first genre)",
                 fontsize=13, fontweight="bold")
    ax.set_xlabel("t-SNE dim 1")
    ax.set_ylabel("t-SNE dim 2")
    ax.grid(alpha=0.2)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved t-SNE plot → {save_path}")


if __name__ == "__main__":
    print("=" * 60)
    print("Model Comparison")
    print("=" * 60)

    # Load config
    with open(PROC_DIR / "config.json") as f:
        config = json.load(f)
    n_users = config["n_users"]
    n_items = config["n_items"]

    # Load data
    loo_test_df = pd.read_csv(PROC_DIR / "loo_test.csv")
    test_df     = pd.read_csv(PROC_DIR / "test_df.csv")
    movies_df   = pd.read_csv(PROC_DIR / "movies_processed.csv")

    with open(PROC_DIR / "idx2movie.pkl", "rb") as f:
        idx2movie = pickle.load(f)

    # Load models
    models = load_all_models(n_users, n_items)

    # Evaluate
    results_df = run_evaluation(models, loo_test_df, test_df)

    # Save CSV
    csv_path = PLOTS_DIR / "model_comparison.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\n  Saved results → {csv_path}")

    # Print table
    print("\n" + "=" * 60)
    print("FINAL COMPARISON TABLE")
    print("=" * 60)
    print(results_df.to_string(index=False))
    print("=" * 60)

    # Bar chart
    plot_comparison_bar(results_df, "plots/comparison_bar.png")

    # t-SNE — needs raw NCF model (not wrapper)
    ncf_raw = NCF(n_users, n_items, embed_dim=64)
    ncf_raw.load_state_dict(torch.load("models/ncf_best.pt", map_location="cpu"))
    plot_embedding_tsne(ncf_raw, movies_df, idx2movie, "plots/embedding_tsne.png")

    print("\n✓ Model comparison complete.\n")
