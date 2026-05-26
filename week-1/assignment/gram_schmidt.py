import numpy as np

def gram_schmidt(A):
    n_cols = A.shape[1]
    U = []

    for k in range(n_cols):
        u_k = A[:, k].astype(float)

        for j in range(k):
            u_j = U[j]
            u_k = u_k - (np.dot(u_k, u_j) / np.dot(u_j, u_j)) * u_j

        U.append(u_k)

    Q = np.column_stack([u / np.linalg.norm(u) for u in U])
    return Q


A = np.array([
    [1, 2, 0],
    [0, 3, 1],
    [1, 1, 4]
], dtype=float)

Q = gram_schmidt(A)
print(Q)
print(np.allclose(Q.T @ Q, np.eye(A.shape[1])))