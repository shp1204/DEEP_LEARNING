import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
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


# normalize
def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드 정보 개수
    print('=====row별 feature 특성 합=====')
    print(rowsum)

    # r_inv
    # 역행렬로 np.power 수행
    # 0, 1, # power : 0, 1, 8, 27, ,,, / 0, 1, 4, 9, ,,, / 0, 1, 0.5, 0.333, 0.25
    r_inv = np.power(rowsum, 0.5).flatten()
    print(r_inv)

    print('===== 역행렬로 np.power 수행 =====')
    r_inv[np.isinf(r_inv)] = 0
    print(r_inv)

    # r_mat_inv
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어줌
    print(r_mat_inv.toarray())

    # 노드 adj 와 노드 feature 정보 행렬연산
    print('=====adj, feature 행렬곱=====')
    mx = r_mat_inv.dot(mx)

    print(mx.toarray())
    return mx

print('features : ')
print(features)
print('adj  :')
print(adj)

features = normalize(features)
adj = normalize(adj) # 대각행렬


print('features : ')
print(features)
print('adj  :')
print(adj)