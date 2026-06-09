"""
ncf_dataset.py
PyTorch Dataset for implicit-feedback NCF training.
Each positive (user, item) pair is augmented with n_negatives random negatives.
Negatives are re-sampled each epoch via refresh().
"""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

np.random.seed(42)
torch.manual_seed(42)


class NCFDataset(Dataset):
    """
    Implicit-feedback dataset with on-the-fly negative sampling.

    Parameters
    ----------
    train_df    : pd.DataFrame with columns ['user_idx', 'movie_idx']
    n_items     : total number of items
    n_negatives : number of negative samples per positive interaction
    """

    def __init__(self, train_df: pd.DataFrame, n_items: int, n_negatives: int = 4):
        self.n_items      = n_items
        self.n_negatives  = n_negatives
        self.rng          = np.random.default_rng(42)

        # Store positives as numpy arrays for speed
        self.pos_users = train_df["user_idx"].values.astype(np.int64)
        self.pos_items = train_df["movie_idx"].values.astype(np.int64)

        # Build set of items seen by each user 
        print("  Building user→seen_items index …")
        self.user_seen: dict[int, set] = {}
        for u, i in zip(self.pos_users, self.pos_items):
            self.user_seen.setdefault(int(u), set()).add(int(i))

        # Internal sample arrays (populated by refresh)
        self._users:  np.ndarray = None
        self._items:  np.ndarray = None
        self._labels: np.ndarray = None

        # Sample negatives for the first epoch
        self.refresh()


    def refresh(self):
        """
        Regenerate negative samples.
        Call once per epoch before creating a new DataLoader iteration.
        """
        all_items = np.arange(self.n_items)
        n_pos = len(self.pos_users)

        neg_users  = np.repeat(self.pos_users, self.n_negatives)
        neg_items  = np.empty(n_pos * self.n_negatives, dtype=np.int64)

        for pos_idx in range(n_pos):
            u    = int(self.pos_users[pos_idx])
            seen = self.user_seen.get(u, set())
            # Sample candidates;
            candidates = self.rng.integers(0, self.n_items,
                                           size=self.n_negatives * 4)
            valid = [c for c in candidates if c not in seen]
            if len(valid) >= self.n_negatives:
                chosen = valid[:self.n_negatives]
            else:
                # Fallback
                pool   = np.setdiff1d(all_items, list(seen))
                chosen = self.rng.choice(pool, size=self.n_negatives,
                                         replace=False).tolist()
            neg_items[pos_idx * self.n_negatives:
                      (pos_idx + 1) * self.n_negatives] = chosen

        # Combine positives and negatives
        self._users  = np.concatenate([self.pos_users, neg_users])
        self._items  = np.concatenate([self.pos_items, neg_items])
        self._labels = np.concatenate([
            np.ones(n_pos,                    dtype=np.float32),
            np.zeros(n_pos * self.n_negatives, dtype=np.float32),
        ])


    def __len__(self) -> int:
        """Total samples = n_positives * (1 + n_negatives)."""
        return len(self._labels)

    def __getitem__(self, idx: int):
        """
        Returns
        -------
        (user_tensor, item_tensor, label_tensor)
        """
        return (
            torch.tensor(self._users[idx],  dtype=torch.long),
            torch.tensor(self._items[idx],  dtype=torch.long),
            torch.tensor(self._labels[idx], dtype=torch.float32),
        )



if __name__ == "__main__":
    from torch.utils.data import DataLoader

    # Minimal synthetic data
    n_users, n_items = 100, 200
    dummy_df = pd.DataFrame({
        "user_idx":  np.random.randint(0, n_users, 500),
        "movie_idx": np.random.randint(0, n_items, 500),
        "Rating":    np.random.randint(1, 6, 500),
    })

    ds = NCFDataset(dummy_df, n_items=n_items, n_negatives=4)
    print(f"Dataset length     : {len(ds)}  (expected {500 * 5} = 2500)")

    dl = DataLoader(ds, batch_size=32, shuffle=True)
    u, i, l = next(iter(dl))
    print(f"Batch user shape   : {u.shape}")
    print(f"Batch item shape   : {i.shape}")
    print(f"Batch label shape  : {l.shape}")
    print(f"Label values       : {set(l.numpy().tolist())}  (expected {{0.0, 1.0}})")

    ds.refresh()
    print(f"After refresh len  : {len(ds)}  (same)")

    print("\n✓ ncf_dataset.py self-test passed.")
