

import torch
import torch.nn as nn


def message_passing(A, X, num_rounds=1, aggregation="sum", add_self_loops=True):
   
    N = A.size(0)

   
    A_hat = A + torch.eye(N) if add_self_loops else A

    if aggregation == "mean":
        
        deg = A_hat.sum(dim=1, keepdim=True)   
        deg[deg == 0] = 1                     
        A_hat = A_hat / deg

    H = X
    for r in range(num_rounds):
        H = A_hat @ H
        print(f"After round {r + 1}:\n{H}\n")

    return H


class GCNLayer(nn.Module):
   

    def __init__(self, in_features, out_features):
        super().__init__()
     
        self.W = nn.Parameter(torch.randn(in_features, out_features))

    def forward(self, A, X):
        N = A.size(0)
        A_hat = A + torch.eye(N)        
        H = A_hat @ X @ self.W          
        return torch.relu(H)            


if __name__ == "__main__":
   
    A = torch.tensor(
        [
            [0, 1, 1, 0],
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=torch.float,
    )

    
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])

    print("Adjacency matrix A:")
    print(A, "\n")
    print("Initial node features X:")
    print(X, "\n")

    print("=== SUM aggregation, 2 rounds (with self-loops) ===")
    message_passing(A, X, num_rounds=2, aggregation="sum")

    print("=== MEAN aggregation, 2 rounds (with self-loops) ===")
    message_passing(A, X, num_rounds=2, aggregation="mean")

    print("=== GCNLayer: relu((A + I) @ X @ W) ===")
    torch.manual_seed(0)            
    layer = GCNLayer(in_features=1, out_features=2)
    out = layer(A, X)
    print("Learnable W:")
    print(layer.W.data, "\n")
    print("Layer output (N x 2):")
    print(out)