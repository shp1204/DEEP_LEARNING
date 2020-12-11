import torch.nn as nn
from layer1 import *
from calculate import *
import scipy.sparse as sp
from preprocessing import *
from torch.nn import Linear
import torch.nn.functional as Func

def normalize(X):
    nom = X - X.min(axis=0)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return nom / denom


class mymodel(nn.Module):
    def __init__(self, A, F, edge_F, method):
        super(mymodel, self).__init__()

        self.rawA = A # 단위행렬 더하기 전
        self.F = normalize(F.toarray()) # node_feature
        self.edge_F = edge_F # edge_feature

        # 전처리
        ## INPUT : A, F, eF, F와 eF를 결합할 방법
        pp = preprocessing(self.rawA, self.F, self.edge_F, method)

        ## 단위행렬 더하는 과정
        self.A = pp.coo().toarray()
        ## features와 edge_features를 한 matrix로 결합하기 위한 전처리

        if method == 'linear':
            self.edge_encoder = Linear(len(self.edge_F), self.F.shape[0])

            self.edge_encoder.reset_parameters()

            self.edge_F = torch.Tensor(np.array(self.edge_F))
            self.edge_F = Func.relu(self.edge_encoder(self.edge_F.T).T).detach().numpy()

        else:
            self.edge_F = pp.convert()



    def forward(self)-> torch.Tensor:

        # hop 2회 계산
        for hop in range(2):
            # AFW -> F, AWeF -> eF
            self.F = layer_1(self.A, self.F).forward()
            self.edge_F = layer_1(self.A, self.edge_F).forward()

        # 최종 F, eF를 concat
        self.newF = torch.cat([torch.Tensor(self.F),
                               torch.Tensor(self.edge_F)], dim=-1)

        return self.newF # 최종 학습할 정보 생성