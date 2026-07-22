# Results Summary
### GraphSAGE Link Prediction on MovieLens (Week 7)

---

## 1. Dataset

| Item | Value |
| --- | --- |
| Dataset | MovieLens `ml-latest-small` |
| Total ratings | 100,836 |
| Users | 610 |
| Movies | 9,724 |
| Edge rule | rating >= 4.0 counts as "liked" |
| Liked edges after filter | ~48,000 |
| Train / test split | 90% / 10% of liked edges |

The graph is bipartite: users on one side, movies on the other, an edge only where a user liked a movie. Users are node indices `0 .. 609`, movies are stacked after them. The graph is stored as an edge list of shape `[2, 2E]` (both directions) rather than an adjacency matrix, since only ~48K of ~5.9M possible pairs actually exist.

---

## 2. Model

| Item | Value |
| --- | --- |
| Architecture | 2-layer GraphSAGE |
| Aggregator | mean over neighbours |
| Layer operation | concat(self, neighbour-mean) then a linear transform |
| Input dimension | 32 (learnable node embeddings) |
| Hidden / output dimension | 64 / 64 |
| Final layer activation | none (raw scores for ranking) |
| Optimizer | Adam, lr = 0.01 |
| Epochs | 100 |
| Loss | BCE on positive edges vs one random negative movie per positive |
| Scoring | dot product of user vector and movie vector |

Note: the aggregator half of GraphSAGE is implemented. Neighbour sampling is not used, since the graph is small enough to process full-batch.

---

## 3. Training Progress

Accuracy here means: for a real (user, liked movie) pair and a random (user, movie) pair, how often the real pair scores higher.

| Epoch | Loss | Accuracy |
| --- | --- | --- |
| 0 | 1.5357 | 0.4685 |
| 20 | 1.3336 | 0.7730 |
| 40 | 0.9381 | 0.8522 |
| 60 | 0.7046 | 0.9123 |
| 80 | 0.5402 | 0.9446 |
| 99 | 0.4610 | 0.9576 |

Accuracy starts at 0.4685, which is essentially random guessing, and rises to 0.9576 by the last epoch. Loss falls steadily from 1.5357 to 0.4610 with no sign of divergence. This confirms the model is learning structure from the graph and not just memorising noise.

---

## 4. Final Results

| Metric | Value |
| --- | --- |
| Recall@10 | 0.0893 |
| Precision@10 | 0.0530 |
| Final training accuracy | 0.9576 |

---




