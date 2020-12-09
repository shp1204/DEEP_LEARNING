import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import scipy.sparse as sp


def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드 정보 개수
    r_inv = np.power(rowsum, 0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어줌
    mx_update = r_mat_inv.dot(mx)
    return mx_update



class mymodel6(torch.nn.Module):

    torch.manual_seed(111)

    def __init__(self, A, F, labels):
        super(mymodel6, self).__init__()

        print('=================init 시작=====================')
        # self.tempA = A.tocoo()
        self.A = normalize(A).toarray()
        self.F = normalize(F).toarray()

        #self.A = torch.FloatTensor(normalize(A).toarray())
        #self.F = torch.FloatTensor(normalize(F).toarray())
        self.labels = torch.LongTensor(labels)

        #self.AF = torch.FloatTensor(self.AF)

        # AF -> float tensor
        print('A size : {}, F size : {}'.format(A.shape , F.shape))
        print('==============================================')
        #print(self.AF) # 행렬


    def AFWlayer(self):
        print('start AFWlayer')
        self.AF = np.array(np.matmul(self.A, self.F))

        self.AF = torch.FloatTensor(self.AF)
        self.temp2 = torch.sparse.Tensor([self.tempA.row.tolist(),
                                          self.tempA.col.tolist()])
        print(self.temp2) # 누구누구 연결인지
        print(self.temp2.size())
        print(self.AF) # 그 안에 담겨있는 정보
        print(self.AF.size())


        # self.weight = Parameter(torch.Tensor(np.random.rand(self.temp2.size()[1],
        #                                                     self.temp2.size()[1])))

        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))

        torch.nn.init.xavier_uniform_(self.weight)

        print(self.temp2.size(), self.weight.size())
        print('end AFWlayer')
        return torch.matmul(self.AF, self.weight).detach().numpy()


    def forward(self) -> torch.Tensor:
        # 1번
        for hop in range(2):
            print('hop{} forward 시작'.format(hop+1))
            self.F = F.relu(torch.Tensor(self.AFWlayer()))
            print(self.F)
            print('hop{} forward 완료'.format(hop+1))
        return self.F