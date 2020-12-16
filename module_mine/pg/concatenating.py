import torch.nn as nn
import numpy as np
import scipy.sparse as sp
from calculate import *
from preprocessing import *

class concatenating(nn.Module):
    def __init__(self, A, F, EF, F_aggr_method, EF_aggr_method):
        super(concatenating, self).__init__()

        self.A = A
        self.F = F
        self.EF = EF
        self.Fmethod = F_aggr_method
        self.EFmethod = EF_aggr_method

        # 1. F
        # 1 ) 단위행렬
        ## F 와 myA_F가 같음 ( 나중에 연산이 필요한지 한번 더 생각해보기 )
        self.myA = sp.eye(self.F.shape[0])
        self.myA_F = np.matmul(self.myA.toarray(), self.F.toarray())

        ppmodule = preprocessing(self.A, self.F, self.EF, self.Fmethod, self.EFmethod)

        # 2 ) 인접행렬
        self.A_F = ppmodule.aggregate()


        # 2. eF 생성
        self.EF = ppmodule.convert()
        # 1 ) 단위행렬
        self.myA_EF = np.matmul(self.myA.toarray(), self.EF)
        # 2 ) 인접행렬
        self.A_EF = ppmodule.convert()


    def result(self):
        return np.concatenate(self.myA_F, self.A_F), np.concatenate(self.myA_EF, self.A_EF)