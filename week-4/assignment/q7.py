


class Graph:

    def __init__(self, n):
        self.n = n
        self.A = [[0] * n for _ in range(n)]
        self.features = [None] * n

    def add_edge(self, u, v, weight=1):
        self.A[u][v] = weight
        self.A[v][u] = weight

    def set_node_feature(self, node, feature):
    
        self.features[node] = feature

    def neighbors(self, node):
    
        return [j for j in range(self.n) if self.A[node][j] != 0]

    def summary(self):
        
        edge_count = sum(
            1 for i in range(self.n) for j in range(i + 1, self.n) if self.A[i][j] != 0
        )
        print("Nodes:", self.n, " Edges:", edge_count)
        print("Node features:", self.features)
        print("Adjacency matrix:")
        for row in self.A:
            print(row)


if __name__ == "__main__":
    #  graph  demo:
    #   0 --- 1
    #   |     |
    #   2 --- 3
    g = Graph(4)

    g.add_edge(0, 1, weight=4)
    g.add_edge(0, 2, weight=5)
    g.add_edge(1, 3, weight=2)
    g.add_edge(2, 3, weight=3)

    g.set_node_feature(0, "instructor")
    g.set_node_feature(1, "student")
    g.set_node_feature(2, "student")
    g.set_node_feature(3, "president")

    g.summary()
    print()
    print("Neighbors of node 0:", g.neighbors(0))
    print("Neighbors of node 3:", g.neighbors(3))