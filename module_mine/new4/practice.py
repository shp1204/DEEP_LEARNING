import numpy as np

def scale(X, x_min, x_max):
    nom = (X-X.min(axis=0))*(x_max-x_min)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom==0] = 1
    return x_min + nom/denom

X = np.array([
    [ 0,  1],
    [ 2,  3],
    [ 4,  5],
    [ 6,  7],
    [ 8,  9],
    [10, 11],
    [12, 13],
    [14, 15]
])


X = np.array([[1,2,3],[2,3,5],[1,3,7],[4,5,7],[0,2,3],[5,6,7],[8,3,1],[9,5,7]])

X_scaled = scale(X, 0, 1)
#print(X_scaled)

edges = np.array([[0,1], [1,2], [2,4], [3,4], [4,5], [5,6], [5,7]])
#features = sp.csr_matrix([[1,2,3],[2,3,5],[1,3,7],[4,5,7],[0,2,3],[5,6,7],[8,3,1],[9,5,7]])
edge_features = [[0],[0],[1],[2],[6],[4],[10]]
labels = np.array([2, 3, 5, 0, 7, 4, 2, 10])
#print(edges.shape[0])
#print(labels.shape[0])
###########################################################################
import torch
i = torch.LongTensor([[2, 4]])
v = torch.FloatTensor([[1, 3],[5, 7]])
print(torch.sparse.FloatTensor(i, v).to_dense())
###########################################################################
