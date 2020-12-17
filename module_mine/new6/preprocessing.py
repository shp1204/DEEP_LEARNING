import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from calculate import *

# input : rawA, self.new_features, edge_F, method
class preprocessing(nn.Module):
    def __init__(self, A, F, edge_F, method):
        super(preprocessing, self).__init__()

        self.A = A
        self.F = F
        self.edge_F = edge_F
        self.method = method

    ## 단위행렬 더하는 과정
    def coo(self):
        self.adj = sp.coo_matrix((np.ones(self.A.shape[0]),
                                  (self.A[:, 0], self.A[:, 1])),
                            shape=(self.F.shape[0], self.F.shape[0]),
                            dtype=np.float32)
        self.adj = self.adj + sp.eye(self.adj.shape[0])
        return self.adj

    ## features와 edge_features를 한 matrix로 결합하기 위한 전처리
    def convert(self):

        ### 현재 feature 사이즈의 array 생성
        self.new_features = [[0]] * self.F.shape[0]

        for idx, A_idx in enumerate(self.A):

            # 해당 edge(3, 4) 중 3에 edge_feature를 할당
            edge_info = A_idx[0]

            # 정보를 여러개 갖고 있을 경우, 기존 정보에 update 해준다
            if self.new_features[edge_info] != [0]:
                self.new_features[edge_info].append(self.edge_F[idx][0])
            # 정보가 없는 경우는 0 대신 채워준다
            else:
                self.new_features[self.A[idx][0]] = self.edge_F[idx]

        # 입력받은 calculate 방법에 따라 중복된 값 처리
        self.new_features = calculate(self.method, self.new_features).cal()
        return self.new_features