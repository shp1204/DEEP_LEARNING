import torch.nn as nn
import scipy.sparse as sp
import numpy as np
from calculate import *

# input : rawA, self.new_features, edge_F, method
class preprocessing(nn.Module):
    def __init__(self, A, F): # , F, EF, F_aggr_method, EF_aggr_method
        super(preprocessing, self).__init__()

        self.A = A
        self.F = F
        # self.EF = EF
        # self.Fmethod = F_aggr_method
        # self.EFmethod = EF_aggr_method

        self.spA = sp.coo_matrix((np.ones(self.A.shape[0]),
                                (self.A[:,0], self.A[:,1])),
                               shape = (self.F.shape[0], self.F.shape[0]),
                               dtype = np.float32)



    def aggregate(self, features, aggr_method):

        # calculate 2개로 나눠야함

        if aggr_method == 'sum':
            return np.matmul(self.spA.toarray(), self.F.toarray())
        if aggr_method == 'mean':
            self.divide = self.spA.toarray().sum(axis=1)
            # 아래에서 Runtime Warning이 발생함. 추후에 수정 필요
            self.tmpF = np.matmul(self.spA.toarray(), features.toarray())/self.divide[:,None]
            return np.nan_to_num(self.tmpF)



    ## features와 edge_features를 한 matrix로 결합하기 위한 전처리
    ## 제일 처음에 edge 합하는 과정
    def convert(self, edge_features, EFmethod):


        self.EF = edge_features
        self.EFmethod = EFmethod


        ### 현재 feature 사이즈의 array 생성
        self.temp_EF = [[0]] * self.F.shape[0]

        for idx, A_idx in enumerate(self.A):

            # 해당 edge(3, 4) 중 3에 edge_features를 할당
            edge_info = A_idx[0]

            # 정보를 여러개 갖고 있을 경우, 기존 정보에 update 해준다

            if self.temp_EF[edge_info] != [0]:
                self.temp_EF[edge_info].append(self.EF[idx][0])
            # 정보가 없는 경우는 0 대신 채워준다
            else:
                self.temp_EF[self.A[idx][0]] = self.EF[idx]


        # 입력받은 calculate 방법에 따라 중복된 값 처리
        self.temp_EF = calculate(self.EFmethod, self.temp_EF).cal()
        return self.temp_EF