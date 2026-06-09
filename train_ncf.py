"""
train_ncf.py
Trains both NCF and NeuMF on the MovieLens-1M dataset.
Uses binary cross-entropy on implicit feedback with negative sampling.
Early stopping on validation Hit Rate@10.
Saves best checkpoints to models/ and plots to plots/.
"""

import json
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path
from torch.utils.data import DataLoader
from tqdm import tqdm

from ncf_model   import NCF, NeuMF
from ncf_dataset import NCFDataset
from evaluate    import evaluate_loo

np.random.seed(42)
torch.manual_seed(42)

MODELS_DIR = Path("models")
PLOTS_DIR  = Path("plots")
PROC_DIR   = Path("data/processed")
MODELS_DIR.mkdir(parents=True, exist_ok=True)
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EMBED_DIM   = 64
LR          = 0.001
BATCH_SIZE  = 256
N_EPOCHS    = 20
N_NEGATIVES = 4
PATIENCE    = 5
EVAL_FRAC   = 0.10    # fraction of LOO test used for per-epoch evaluation (speed)
K           = 10



class NCFWrapper:
    """Thin wrapper so PyTorch models expose the predict_batch interface."""

    def __init__(self, model: nn.Module, device: torch.device):
        self.model  = model
        self.device = device
        self.model.eval()

    def predict_batch(self, user_idx: int, movie_idx_list: list) -> np.ndarray:
        users  = torch.tensor([user_idx] * len(movie_idx_list), dtype=torch.long).to(self.device)
        items  = torch.tensor(movie_idx_list,                   dtype=torch.long).to(self.device)
        with torch.no_grad():
            scores = self.model(users, items).squeeze(-1).cpu().numpy()
        return scores



def train_one_model(
    model_name: str,
    model: nn.Module,
    train_df: pd.DataFrame,
    loo_test_df: pd.DataFrame,
    n_items: int,
    save_path: str,
) -> dict:
    """
    Full training loop for one model.
    Returns history dict with lists of train_loss, val_hit, val_ndcg.
    """
    print("\n" + "=" * 60)
    print(f"Training {model_name}")
    print("=" * 60)
    print(f"  Device     : {DEVICE}")
    print(f"  Parameters : {sum(p.numel() for p in model.parameters()):,}")

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.BCELoss()

    dataset = NCFDataset(train_df, n_items=n_items, n_negatives=N_NEGATIVES)

    history = {"train_loss": [], "val_hit": [], "val_ndcg": []}
    best_hit = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, N_EPOCHS + 1):
        dataset.refresh()
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                            num_workers=0, pin_memory=(DEVICE.type == "cuda"))

        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for users, items, labels in tqdm(loader, desc=f"  Epoch {epoch:02d}/{N_EPOCHS}",
                                         leave=False):
            users  = users.to(DEVICE)
            items  = items.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad()
            preds = model(users, items).squeeze(-1)
            loss  = criterion(preds, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            n_batches  += 1

        avg_loss = epoch_loss / max(n_batches, 1)

        model.eval()
        wrapper = NCFWrapper(model, DEVICE)
        metrics = evaluate_loo(wrapper, loo_test_df, idx2movie={},
                               k=K, sample_frac=EVAL_FRAC)
        hit  = metrics[f"hit@{K}"]
        ndcg = metrics[f"ndcg@{K}"]

        history["train_loss"].append(avg_loss)
        history["val_hit"].append(hit)
        history["val_ndcg"].append(ndcg)

        improved = hit > best_hit
        if improved:
            best_hit = hit
            torch.save(model.state_dict(), save_path)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"  Epoch {epoch:02d}/{N_EPOCHS} | "
            f"Loss={avg_loss:.4f} | "
            f"Hit@{K}={hit:.4f} | "
            f"NDCG@{K}={ndcg:.4f}"
            + ("  ✓ saved" if improved else f"  (patience {epochs_without_improvement}/{PATIENCE})")
        )

        if epochs_without_improvement >= PATIENCE:
            print(f"  Early stopping triggered at epoch {epoch}.")
            break

    print(f"\n  Best Val Hit@{K}: {best_hit:.4f}   checkpoint → {save_path}")
    return history



def plot_ncf_curves(ncf_hist: dict, neumf_hist: dict, save_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, metric, title in zip(
        axes,
        ["val_hit", "val_ndcg"],
        [f"Hit Rate@{K}", f"NDCG@{K}"],
    ):
        for hist, label, color in [
            (ncf_hist,   "NCF",   "steelblue"),
            (neumf_hist, "NeuMF", "darkorange"),
        ]:
            epochs = range(1, len(hist[metric]) + 1)
            ax.plot(epochs, hist[metric], "o-", color=color, markersize=4, label=label)

        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.set_title(f"Validation {title}")
        ax.legend()
        ax.grid(alpha=0.3)

    plt.suptitle("NCF vs NeuMF Training", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Saved training curves → {save_path}")



def full_eval(model: nn.Module, loo_test_df: pd.DataFrame, name: str) -> dict:
    """Run full LOO evaluation on all users."""
    print(f"\n  Full LOO eval for {name} …")
    model.eval()
    wrapper = NCFWrapper(model, DEVICE)
    metrics = evaluate_loo(wrapper, loo_test_df, idx2movie={}, k=K, sample_frac=1.0)
    print(f"  {name}: NDCG@{K}={metrics[f'ndcg@{K}']:.4f}  Hit@{K}={metrics[f'hit@{K}']:.4f}")
    return metrics



if __name__ == "__main__":
    print("Loading processed data …")
    train_df    = pd.read_csv(PROC_DIR / "train_df.csv")
    loo_test_df = pd.read_csv(PROC_DIR / "loo_test.csv")

    with open(PROC_DIR / "config.json") as f:
        config = json.load(f)
    n_users = config["n_users"]
    n_items = config["n_items"]

    print(f"  n_users={n_users}, n_items={n_items}")
    print(f"  Train interactions: {len(train_df):,}")
    print(f"  LOO test users    : {len(loo_test_df):,}")

    ncf_model = NCF(n_users, n_items, embed_dim=EMBED_DIM)
    ncf_hist  = train_one_model(
        "NCF",
        ncf_model,
        train_df,
        loo_test_df,
        n_items,
        save_path="models/ncf_best.pt",
    )

    neumf_model = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32)
    neumf_hist  = train_one_model(
        "NeuMF",
        neumf_model,
        train_df,
        loo_test_df,
        n_items,
        save_path="models/neumf_best.pt",
    )

    ncf_model.load_state_dict(torch.load("models/ncf_best.pt", map_location=DEVICE))
    neumf_model.load_state_dict(torch.load("models/neumf_best.pt", map_location=DEVICE))
    ncf_model.to(DEVICE)
    neumf_model.to(DEVICE)

    ncf_metrics   = full_eval(ncf_model,   loo_test_df, "NCF")
    neumf_metrics = full_eval(neumf_model, loo_test_df, "NeuMF")

    plot_ncf_curves(ncf_hist, neumf_hist, "plots/ncf_training_curve.png")

    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    print(f"{'Model':<10} | {'NDCG@10':>8} | {'Hit@10':>8}")
    print("-" * 35)
    for name, m in [("NCF", ncf_metrics), ("NeuMF", neumf_metrics)]:
        print(f"{name:<10} | {m[f'ndcg@{K}']:>8.4f} | {m[f'hit@{K}']:>8.4f}")
    print("=" * 60)
    print("\n✓ NCF training complete.\n")
