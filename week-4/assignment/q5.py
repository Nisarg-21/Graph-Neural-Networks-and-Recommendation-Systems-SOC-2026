

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph, NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from torch_geometric.data import Data
from torch_geometric.nn import GATConv
from torch_geometric.utils import to_undirected

CSV = r"IITB_Student_Dataset.xlsx - Sheet1.csv"
k = 8

raw = pd.read_csv(CSV).drop(columns=["Student_ID"])
n = len(raw)
rng = np.random.default_rng(0)
perm = rng.permutation(n)
train_idx, test_idx = perm[: int(0.8 * n)], perm[int(0.8 * n):]
raw_train = raw.iloc[train_idx].reset_index(drop=True)
raw_test = raw.iloc[test_idx].reset_index(drop=True)

g_train = raw_train["Cumulative_Grade"]
floor = g_train.min()
mid_cut = g_train[g_train > floor].median()

def to_class(g):
    return np.where(g <= floor, 0, np.where(g <= mid_cut, 1, 2))

y_train = to_class(raw_train["Cumulative_Grade"].values)
y_test = to_class(raw_test["Cumulative_Grade"].values)

diet_map = {"Poor": 0, "Average": 1, "Good": 2, "Excellent": 3}

def encode(df):
    df = df.drop(columns=["Cumulative_Grade"]).copy()
    df["Diet_Quality"] = df["Diet_Quality"].map(diet_map)
    return pd.get_dummies(df, columns=["Gender", "Department", "Living_Situation"])

enc_train = encode(raw_train)
enc_test = encode(raw_test).reindex(columns=enc_train.columns, fill_value=0)

medians = enc_train.median(numeric_only=True)          
enc_train = enc_train.fillna(medians)
enc_test = enc_test.fillna(medians)

scaler = StandardScaler().fit(enc_train.values.astype(float))   
X_train = scaler.transform(enc_train.values.astype(float))
X_test = scaler.transform(enc_test.values.astype(float))

coo = kneighbors_graph(X_train, n_neighbors=k, mode="connectivity").tocoo()
ei_train = to_undirected(torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long))
data_train = Data(
    x=torch.tensor(X_train, dtype=torch.float),
    edge_index=ei_train,
    y=torch.tensor(y_train, dtype=torch.long),
)

class GAT(nn.Module):
    def __init__(self, in_dim, hidden, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(in_dim, hidden, heads=heads, dropout=0.5)
        self.conv2 = GATConv(hidden * heads, num_classes, heads=1, dropout=0.5)

    def forward(self, x, edge_index):
        x = F.elu(self.conv1(x, edge_index))
        return self.conv2(x, edge_index)

model = GAT(X_train.shape[1], hidden=8, num_classes=3)
opt = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
for epoch in range(1, 201):
    model.train()
    opt.zero_grad()
    out = model(data_train.x, data_train.edge_index)
    loss = F.cross_entropy(out, data_train.y)
    loss.backward()
    opt.step()

n_tr, n_te = X_train.shape[0], X_test.shape[0]
_, nbr_idx = NearestNeighbors(n_neighbors=k).fit(X_train).kneighbors(X_test)

x_all = torch.tensor(np.vstack([X_train, X_test]), dtype=torch.float)
test_src = np.repeat(np.arange(n_te) + n_tr, k)   
test_dst = nbr_idx.reshape(-1)                    
extra = torch.tensor(np.vstack([test_src, test_dst]), dtype=torch.long)
ei_all = to_undirected(torch.cat([ei_train, extra], dim=1))

model.eval()
with torch.no_grad():
    gat_pred = model(x_all, ei_all)[n_tr:].argmax(dim=1).numpy()   
gat_acc = accuracy_score(y_test, gat_pred)

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(X_train, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test))

print(f"Train nodes: {n_tr}  Test nodes: {n_te}  Features: {X_train.shape[1]}  k={k}")
print(f"GAT (inductive) test accuracy: {gat_acc:.3f}")
print(f"Random Forest test accuracy:   {rf_acc:.3f}")