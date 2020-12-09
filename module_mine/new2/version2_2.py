import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as Func
import scipy.sparse as sp
from torch.nn import Linear, ReLU
from torch import Tensor


# normalize
def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드가 갖고 있는 정보 개수
    r_inv = np.power(rowsum, 0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어 준다
    mx_update = r_mat_inv.dot(mx)
    return mx_update


class mymodel10(torch.nn.Module):

    def __init__(self, A, F, edge_F, rawA):
        super(mymodel10, self).__init__()
        self.rawA = A
        self.A = normalize(A).toarray()  # 인접행렬
        self.F = normalize(F).toarray()  # feature정보
        self.edge_F = torch.Tensor(edge_F) # edge_feature 정보

        # A, F 사이즈 확인
        print('A size : {}, F size : {}, edge_F size : {}'.format(A.shape, F.shape, len(edge_F)))
        print('=' * 50)

        # 0으로 구성된 feature 생성
        self.new_features = [[]]* F.shape[0]
        print('self.new_features : {}'.format(self.new_features))

        for idx, A_idx in enumerate(rawA):
            edge_info = self.new_features[rawA[idx][0]]

            if edge_info:
                self.new_features[rawA[idx][0]].append(edge_F[idx][0])
            else:
                self.new_features[rawA[idx][0]] = edge_F[idx]

        self.new_features.fillna(0)
        print(self.new_features)


    def aggregate(self, method) -> Tensor:
        # new_features에 빈 곳 있으면 0으로 채워주고
        self.new_features.fillna(0)
        
        # sum, mean, min 이런 method 입력받아서
        # 2개 이상일 때 위에서 선택한 연산 수행하고
        
        # transpose 해서 feature로 concat해주기
    
        self.newF = torch.cat([torch.Tensor(self.F), torch.Tensor(self.new_features)], dim=-1)
        print(self.newF)
        self.newF = self.F




    def AFWlayer(self):
        print('=' * 50)
        print('start AFWlayer')
        # A, F 연산

        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.newF.detach().numpy())))# AF
        # AF 크기의 weight 생성
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))
        # weight 초기화
        torch.nn.init.xavier_uniform_(self.weight)

        print('=' * 50)
        print(self.AF)

        # AF와 weight 연산
        return torch.matmul(self.AF, self.weight).detach().numpy()  # AF X Weight


    def forward(self) -> torch.Tensor:
        # AFW = newF가 되어서 새로운 AFW 생성
        for hop in range(2):

            self.newF = Func.relu(torch.Tensor(self.AFWlayer()))
            print('=' * 15, 'relu 취한 후', '=' * 15)
            print(self.newF)

        return self.newF

