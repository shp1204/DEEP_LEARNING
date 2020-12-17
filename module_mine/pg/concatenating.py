import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from calculate import *
from preprocessing import *

class concatenating(nn.Module):
    def __init__(self, A, F, EF, F_aggr_method, EF_aggr_method):
        super(concatenating, self).__init__()

        self.A = A # np.array
        self.F = F # sp.csr_matrix
        self.EF = EF # list
        self.Fmethod = F_aggr_method
        self.EFmethod = EF_aggr_method

        # 1. F
        # 1 ) 단위행렬
        ## F 와 myA_F가 같음 ( 나중에 연산이 필요한지 한번 더 생각해보기 )
        self.myA = sp.eye(self.F.shape[0])
        self.myA_F = np.matmul(self.myA.toarray(), self.F.toarray())

        # ppmodule = preprocessing(self.A, self.F, self.EF,
        #                          self.Fmethod, self.EFmethod)

        ppmodule = preprocessing(self.A, self.F)

        # 2 ) 인접행렬
        self.A_F = ppmodule.aggregate(self.F, self.Fmethod)


        # 2. eF 생성
        ## 주의 : edge_features가 여러개일 경우에는 평균낸 값부터 시작한다
        ## edge_featurs = [4, 7, 6] -> [5.333] 부터 aggr 시작함
        self.EF = ppmodule.convert(self.EF, self.EFmethod) # 제일 처음 edge 합하고
        # 1 ) 단위행렬 ## 처음 edge 합한거랑 # self.EF랑 똑같음
        self.myA_EF = np.matmul(self.myA.toarray(), self.EF)
        # 2 ) 인접행렬 ## 인접행렬 edge 합한거랑
        self.A_EF = ppmodule.aggregate(sp.csr_matrix(self.EF), self.EFmethod)

    def result(self):
        return sp.csr_matrix(np.concatenate((self.myA_F, self.A_F), axis=1)),\
               np.concatenate((self.myA_EF, self.A_EF), axis=1)