import torch, torch.nn as nn, torch.nn.functional as F
import pandas as pd
from collections import defaultdict
from sklearn.model_selection import train_test_split
import os, urllib.request, zipfile
torch.manual_seed(0)

# 0. Download MovieLens (ml-latest-small) if not already present
DATA_DIR = "ml-latest-small"
RATINGS_PATH = os.path.join(DATA_DIR, "ratings.csv")
if not os.path.exists(RATINGS_PATH):
    url = "https://files.grouplens.org/datasets/movielens/ml-latest-small.zip"
    print("Downloading MovieLens ml-latest-small...")
    urllib.request.urlretrieve(url, "ml-latest-small.zip")
    with zipfile.ZipFile("ml-latest-small.zip", "r") as z:
        z.extractall(".")
    print("Done.")

# 1. Load MovieLens
ratings = pd.read_csv(RATINGS_PATH)
print(f"{len(ratings):,} ratings, {ratings.userId.nunique():,} users, {ratings.movieId.nunique():,} movies")
ratings = ratings[ratings.rating >= 4.0]                  # link = liked it


u_ids, m_ids = ratings.userId.unique(), ratings.movieId.unique()
num_users, num_movies = len(u_ids), len(m_ids)
N = num_users + num_movies
u_map = {u: i for i, u in enumerate(u_ids)}
m_map = {m: i + num_users for i, m in enumerate(m_ids)}    # movies after users
ratings["u"] = ratings.userId.map(u_map)
ratings["m"] = ratings.movieId.map(m_map)
train_df, test_df = train_test_split(ratings, test_size=0.1, random_state=0)

u = torch.tensor(train_df.u.values); m = torch.tensor(train_df.m.values)
edge_index = torch.stack([torch.cat([u, m]), torch.cat([m, u])])   # [2, 2E]

in_dim = 32
node_emb = nn.Embedding(N, in_dim)

# 2. Mean aggregator, vectorized
def aggregate_mean(H, edge_index, N):
    src, dst = edge_index
    agg = torch.zeros_like(H).index_add_(0, dst, H[src])
    deg = torch.zeros(N, device=H.device).index_add_(0, dst, torch.ones_like(dst, dtype=H.dtype))
    return agg / deg.clamp(min=1).unsqueeze(-1)

class SageLayer(nn.Module):
    def __init__(self, i, o, act=True):
        super().__init__(); self.W = nn.Linear(2*i, o); self.act = act
    def forward(self, H, edge_index, N):
        out = self.W(torch.cat([H, aggregate_mean(H, edge_index, N)], dim=1))
        return F.relu(out) if self.act else out

class GraphSAGE(nn.Module):
    def __init__(self, i, h, o):
        super().__init__(); self.l1 = SageLayer(i, h); self.l2 = SageLayer(h, o, act=False)
    def forward(self, X, edge_index, N):
        return self.l2(self.l1(X, edge_index, N), edge_index, N)

model = GraphSAGE(in_dim, 64, 64)
opt = torch.optim.Adam(list(model.parameters()) + list(node_emb.parameters()), lr=0.01)

# 3. Train
pos_u = torch.tensor(train_df.u.values); pos_m = torch.tensor(train_df.m.values)
for epoch in range(100):
    Z = model(node_emb.weight, edge_index, N)
    neg_m = torch.randint(num_users, N, (pos_u.size(0),))
    pos_score = (Z[pos_u] * Z[pos_m]).sum(-1)
    neg_score = (Z[pos_u] * Z[neg_m]).sum(-1)
    loss = -(F.logsigmoid(pos_score) + F.logsigmoid(-neg_score)).mean()
    acc = (pos_score > neg_score).float().mean()
    opt.zero_grad(); loss.backward(); opt.step()
    print(f"epoch {epoch:3d} loss {loss.item():.4f} acc {acc.item():.4f}")

# 4. Recall@K 
model.eval()
with torch.no_grad():
    Z = model(node_emb.weight, edge_index, N)
    movie_emb = Z[num_users:]
    test_pos, train_pos = defaultdict(set), defaultdict(set)
    for r in test_df.itertuples(): test_pos[r.u].add(r.m)
    for r in train_df.itertuples(): train_pos[r.u].add(r.m)
    K, recalls, precisions = 10, [], []
    for user, truth in test_pos.items():
        scores = movie_emb @ Z[user]
        for seen in train_pos[user]:
            scores[seen - num_users] = -1e9
        topk = (torch.topk(scores, K).indices + num_users).tolist()
        hits = len(set(topk) & truth)
        recalls.append(hits / len(truth)); precisions.append(hits / K)
print(f"Recall@{K}    {sum(recalls)/len(recalls):.4f}")
print(f"Precision@{K} {sum(precisions)/len(precisions):.4f}")