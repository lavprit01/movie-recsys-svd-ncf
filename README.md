# Movie Recommendation Engine: SVD + Neural Collaborative Filtering

A production-quality movie recommendation system implementing and comparing three approaches
— Popularity Baseline, Matrix Factorisation (SVD from scratch in NumPy), and Neural CF
(NCF & NeuMF in PyTorch) — on the MovieLens-1M dataset, with an interactive Streamlit dashboard.

---

## Execution Order

```bash
python data_loader.py        # 1. Process raw data → data/processed/
python baseline.py           # 2. Train & save popularity model
python svd_scratch.py        # 3. Train SVD (NumPy SGD), save model + plots
python train_ncf.py          # 4. Train NCF + NeuMF, save models + plots
python compare_models.py     # 5. Full evaluation, comparison table + plots
streamlit run app.py         # 6. Launch interactive dashboard
```

---

## Results

| Model          | NDCG@10 | Hit Rate@10 | RMSE |
|----------------|---------|-------------|------|
| Popularity     | -       | -           | -    |
| SVD (scratch)  | -       | -           | -    |
| NCF (scratch)  | -       | -           | -    |
| NeuMF          | -       | -           | -    |

*Fill in real numbers after training.*

---

## Architecture

### SVD — Matrix Factorisation from Scratch

Predicts rating as **r̂(u,i) = μ + b_u + b_i + P[u]·Q[i]**, where μ is the global mean,
b_u / b_i are user and item biases, and P[u] / Q[i] are 64-dimensional latent vectors
learned by SGD over all observed (user, item, rating) triplets. L2 regularisation and an
epoch-wise learning-rate decay (factor 0.96) prevent overfitting. The entire optimiser is
implemented in raw NumPy — no external SVD library is used.

### NCF & NeuMF — Neural Collaborative Filtering

NCF replaces the inner-product of SVD with a multi-layer perceptron: user and item indices
are mapped to dense embeddings, concatenated, and passed through four linear layers
(128→64→32→1) with ReLU activations and dropout, outputting an interaction probability via
sigmoid. NeuMF (He et al., 2017) extends this with a parallel GMF path (element-wise product
of 32-dim embeddings) fused with the MLP path output before the final prediction layer,
combining linear and non-linear interaction modelling. Both models are trained on implicit
feedback using binary cross-entropy with four negative samples per positive interaction.

---

## Setup

### 1. Create and activate environment

```bash
conda create -n recsys python=3.10 -y
conda activate recsys
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Download the dataset

Download MovieLens-1M from the official GroupLens page:
https://grouplens.org/datasets/movielens/1m/

Extract the archive and place the three `.dat` files in `data/raw/`:

```
data/raw/
├── ratings.dat
├── movies.dat
└── users.dat
```

### 4. Run the pipeline

```bash
# From the project root (movie-recsys-svd-ncf/)
python data_loader.py
python baseline.py
python svd_scratch.py
python train_ncf.py
python compare_models.py
```

---

## Usage

### Streamlit App

```bash
streamlit run app.py
```

Open http://localhost:8501 in your browser.

**Features:**
- Select any model (Popularity / SVD / NCF / NeuMF) or "Compare All"
- Enter a User ID (1–6040) and choose Top-K (5–20)
- View ranked recommendations with movie title, genres, and model score
- Explore model metrics, embedding t-SNE visualisation, and algorithm explanations
- Cold-start detection: users with < 5 training ratings fall back to Popularity

### Notebook

```bash
cd notebooks
jupyter notebook analysis.ipynb
```

---

## Tech Stack

- **Python 3.10**
- **NumPy** — SVD from scratch (SGD, L2 regularisation)
- **PyTorch** — NCF and NeuMF neural models
- **Pandas** — data loading and preprocessing
- **scikit-learn** — t-SNE for embedding visualisation
- **Matplotlib** — training curves and comparison plots
- **Streamlit** — interactive web dashboard
- **tqdm** — progress bars

---

## Key Implementation Details

### Custom SGD for SVD

Rather than calling `np.linalg.svd`, the model iterates over every (user, item, rating)
triplet and applies the gradient update rule: for error `e = r - r̂`, the update is
`P[u] += lr * (e * Q[i] - reg * P[u])` and symmetrically for `Q[i]` and the biases.
This produces a Funk-SVD / biased MF model that converges efficiently to a low-RMSE solution.

### Negative Sampling for NCF

For each positive (user, item) interaction, four items the user has **not** rated are
sampled uniformly at random and labelled 0. Negatives are regenerated at the start of every
epoch to prevent the model from memorising specific negative examples. The full training set
therefore contains 5× the number of observed interactions.

### Leave-One-Out Evaluation

Following the standard protocol (He et al., 2017), the most recent rated item per user
(by timestamp) is held out as the test positive. 99 random unseen items are sampled as
negatives. All 100 candidates are scored, ranked, and the positive's rank in the top-10
determines Hit Rate@10 and NDCG@10. This protocol is deterministic and reproducible with
`random_seed=42`.

### NDCG Metric

Normalised Discounted Cumulative Gain discounts the rank of the positive item
logarithmically: **NDCG = 1 / log₂(rank + 1)** if the positive appears in the top-K, else 0.
The ideal DCG is 1 (positive at rank 1), so no normalisation constant is needed with a
single relevant item.

---

## File Structure

```
movie-recsys-svd-ncf/
├── data/
│   ├── raw/            ← original .dat files (download separately)
│   └── processed/      ← generated by data_loader.py
├── models/             ← saved model checkpoints
├── plots/              ← training curves, comparison charts, t-SNE
├── notebooks/
│   └── analysis.ipynb  ← end-to-end analysis notebook
├── data_loader.py      ← data pipeline
├── evaluate.py         ← metric functions
├── baseline.py         ← popularity recommender
├── svd_scratch.py      ← SVD via NumPy SGD
├── ncf_model.py        ← NCF and NeuMF PyTorch models
├── ncf_dataset.py      ← implicit-feedback dataset with negative sampling
├── train_ncf.py        ← NCF/NeuMF training script
├── compare_models.py   ← full evaluation + plots
├── app.py              ← Streamlit dashboard
└── requirements.txt
```
