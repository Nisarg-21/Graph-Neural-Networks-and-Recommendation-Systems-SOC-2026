

import networkx as nx
import numpy as np

G = nx.karate_club_graph()

print("Number of nodes:", G.number_of_nodes())
print("Number of edges:", G.number_of_edges())
print()


A = nx.to_numpy_array(G, dtype=int)

print("Adjacency matrix (shape", A.shape, "):")
print(A)
print()

edges = list(G.edges(data="weight"))

print("Edge list (", len(edges), "edges ):")
for u, v, w in edges:
    print(u, "--", v, " weight:", w)