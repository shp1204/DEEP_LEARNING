import numpy as np
import scipy.sparse as sp
from concatenating import concatenating

# 데이터 로드

import pickle
graph_all = {}
path = './data/graph'
for idx in range(1, 4):
    with open(path + str(idx) + '.txt', 'rb') as rf:
        graph_all['graph'+str(idx)] = pickle.load(rf)



test = graph_all['graph'+str(2)]

edges = np.array(test['edges'])
features = sp.csr_matrix(test['features'])
edge_features = test['edge_features']
labels = test['labels']

#print(edges, features, edge_features, labels)



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
#print(features)

hop = 5
hoppi = hop
for turn in range(1, hop+1):
    if hoppi == 1 :
        df = concatenating(edges, features, edge_features,
                           'mean', 'mean').concat()
        print('here')
        # features로 GCNConv # 1, 1 로 변경
    else:
        features = concatenating(edges, features, edge_features,
                   'mean', 'mean').concat()

        # features로 GCNConv # features랑 크기 똑같게 변경
        # sp.csr_matrix로 변경해서 features에 다시 넣어줘야함


        hoppi -= 1

