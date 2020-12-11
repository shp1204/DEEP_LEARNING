import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as Func
import scipy.sparse as sp


# normalize
def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드가 갖고 있는 정보 개수
    r_inv = np.power(rowsum, 0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어 준다
    mx_update = r_mat_inv.dot(mx)
    return mx_update


class mymodel8(torch.nn.Module):

    # torch.manual_seed(111)

    def __init__(self, A, F):
        super(mymodel8, self).__init__()
        self.A = normalize(A).toarray()  # 인접행렬
        self.F = normalize(F).toarray()  # feature정보

        # A, F 사이즈 확인
        print('A size : {}, F size : {}'.format(A.shape, F.shape))
        print('=' * 50)


    def AFWlayer(self):
        print('=' * 50)
        print('start AFWlayer')
        # A, F 연산
        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.F)))  # AF
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

            self.F = Func.relu(torch.Tensor(self.AFWlayer()))
            print('=' * 15, 'relu 취한 후', '=' * 15)
            print(self.F)

        return self.F

