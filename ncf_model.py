"""
ncf_model.py
PyTorch implementations of:
  - NCF  : Neural Collaborative Filtering (concat-embed → MLP → sigmoid)
  - NeuMF: Neural Matrix Factorisation (GMF + MLP fusion, He et al. 2017)
"""

import torch
import torch.nn as nn

torch.manual_seed(42)



class NCF(nn.Module):
    """
    Basic Neural Collaborative Filtering.

    Architecture:
        user_emb(u) ++ item_emb(i)
        → Linear(2*embed_dim, 128) → ReLU → Dropout(0.2)
        → Linear(128, 64)          → ReLU → Dropout(0.2)
        → Linear(64, 32)           → ReLU
        → Linear(32, 1)            → Sigmoid
    """

    def __init__(self, n_users: int, n_items: int, embed_dim: int = 64):
        super().__init__()
        self.user_embedding = nn.Embedding(n_users, embed_dim)
        self.item_embedding = nn.Embedding(n_items, embed_dim)

        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.normal_(self.user_embedding.weight, std=0.01)
        nn.init.normal_(self.item_embedding.weight, std=0.01)
        for layer in self.mlp:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        user_idx : LongTensor (batch,)
        item_idx : LongTensor (batch,)

        Returns
        -------
        Tensor (batch, 1) – interaction probability in [0, 1]
        """
        u = self.user_embedding(user_idx)   # (batch, embed_dim)
        i = self.item_embedding(item_idx)   # (batch, embed_dim)
        x = torch.cat([u, i], dim=-1)       # (batch, 2*embed_dim)
        return self.mlp(x)                  # (batch, 1)



class NeuMF(nn.Module):
    """
    Neural Matrix Factorisation (He et al., 2017).

    Two parallel paths:
      GMF path  – element-wise product of GMF embeddings
      MLP path  – concatenated MLP embeddings through a deep MLP

    Fusion:
      concat(gmf_out, mlp_out) → Linear(48, 1) → Sigmoid
    """

    def __init__(self, n_users: int, n_items: int,
                 gmf_dim: int = 32, mlp_dim: int = 32):
        super().__init__()

        self.user_emb_gmf = nn.Embedding(n_users, gmf_dim)
        self.item_emb_gmf = nn.Embedding(n_items, gmf_dim)

        self.user_emb_mlp = nn.Embedding(n_users, mlp_dim)
        self.item_emb_mlp = nn.Embedding(n_items, mlp_dim)

        self.mlp = nn.Sequential(
            nn.Linear(mlp_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
        )

        fusion_in = gmf_dim + 16   # = 48
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, 1),
            nn.Sigmoid(),
        )

        self._init_weights()

    def _init_weights(self):
        for emb in [self.user_emb_gmf, self.item_emb_gmf,
                    self.user_emb_mlp, self.item_emb_mlp]:
            nn.init.normal_(emb.weight, std=0.01)

        for layer in list(self.mlp) + list(self.fusion):
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)

    def forward(self, user_idx: torch.Tensor, item_idx: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        user_idx : LongTensor (batch,)
        item_idx : LongTensor (batch,)

        Returns
        -------
        Tensor (batch, 1) – interaction probability in [0, 1]
        """
        # GMF path
        u_gmf = self.user_emb_gmf(user_idx)       # (batch, gmf_dim)
        i_gmf = self.item_emb_gmf(item_idx)       # (batch, gmf_dim)
        gmf_out = u_gmf * i_gmf                   # element-wise, (batch, gmf_dim)

        # MLP path
        u_mlp = self.user_emb_mlp(user_idx)       # (batch, mlp_dim)
        i_mlp = self.item_emb_mlp(item_idx)       # (batch, mlp_dim)
        mlp_in  = torch.cat([u_mlp, i_mlp], dim=-1)   # (batch, 2*mlp_dim)
        mlp_out = self.mlp(mlp_in)                     # (batch, 16)

        # Fusion
        fused = torch.cat([gmf_out, mlp_out], dim=-1)  # (batch, 48)
        return self.fusion(fused)                       # (batch, 1)



if __name__ == "__main__":
    n_users, n_items = 6040, 3706
    batch = 8

    users = torch.randint(0, n_users, (batch,))
    items = torch.randint(0, n_items, (batch,))

    # NCF
    ncf = NCF(n_users, n_items, embed_dim=64)
    out_ncf = ncf(users, items)
    print(f"NCF output shape   : {out_ncf.shape}   (expected: [{batch}, 1])")
    ncf_params = sum(p.numel() for p in ncf.parameters())
    print(f"NCF total params   : {ncf_params:,}")

    # NeuMF
    neumf = NeuMF(n_users, n_items, gmf_dim=32, mlp_dim=32)
    out_neumf = neumf(users, items)
    print(f"NeuMF output shape : {out_neumf.shape}  (expected: [{batch}, 1])")
    neumf_params = sum(p.numel() for p in neumf.parameters())
    print(f"NeuMF total params : {neumf_params:,}")

    print("\n✓ ncf_model.py self-test passed.")
