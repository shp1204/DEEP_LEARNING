import numpy as np
import scipy.sparse as sp
from concatenating import concatenating
import torch
from torch.nn.parameter import Parameter
from model import *

# 데이터 로드

import pickle
graph_all = {}
path = './data/graph'
for idx in range(1, 5):
    with open(path + str(idx) + '.txt', 'rb') as rf:
        graph_all['graph'+str(idx)] = pickle.load(rf)



test = graph_all['graph'+str(4)]

edges = np.array(test['edges'])
features = sp.csr_matrix(test['features'])
edge_features = test['edge_features']
labels = test['labels']


model(1, edges, features, edge_features,'mean', 'mean')
#print(edges, features, edge_features, labels)
# print(edges)
# print(features)
# print(edge_features)
# print(labels)


# preprocessing
'''
1 ) 단위행렬
단위행렬 matrix 생성
단위행렬 matrix x F

2 ) 인접행렬 x F
인접행렬 matrix 생성
인접행렬 matrix x F

3 ) edge_feature 행렬 x edgeF
edge_features 크기 맞추고
인접행렬 matrix x edgeF

4 ) concat
concat matrix 생성 x weight
- hop에 따라 다르게
-- hop == 1 : concat 후 weight 바로 진행
-- hop >= 2 : concat 후 F -> 마지막 layer에서 weight 진행 
'''



        # weight 가 변해야하는데 ?

        # features로 GCNConv # features랑 크기 똑같게 변경
        # sp.csr_matrix로 변경해서 features에 다시 넣어줘야함




