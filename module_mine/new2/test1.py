import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

# data
#노드 연결 정보
edges = np.array([[0 ,1],[2 ,3],[1 ,4],[3 ,4],[4 ,5],[4 ,6]])
# 각 노드 특성 정보(H) = 7 X 4
features = sp.csr_matrix([[1, 0, 0, 0],[0, 1, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 1]])
# edge 특성 정보
edge_features = [[3],[5],[1],[10],[6],[8]]
# labels
labels = np.array([1,4,5,2,6,3,0])

# 단위 행렬 더해주기
# direction matrix 생성
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
             shape = (labels.shape[0], labels.shape[0]),
             dtype = np.float32)
adj = adj + sp.eye(adj.shape[0])
print(adj)
print(features)