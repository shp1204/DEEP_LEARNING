import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import  ReLU
import torch.nn.functional as Func

def normalize(X):
    nom = X - X.min(axis=0)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1 # 최대값 최소값이 같은 경우
    return nom / denom

class layer_1(torch.nn.Module):
    def __init__(self, Adj, matrix):
        super(layer_1, self).__init__()
        # Adj 행렬과 함께 곱하고자 하는 행렬 입력 받음
        self.A = Adj
        self.matrix = matrix

    def forward(self) -> torch.Tensor:

        # AF
        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.matrix)))
        self.AF = self.AF/self.AF.sum(axis=1)[:,None]
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))

        # weight 초기화
        torch.nn.init.xavier_uniform_(self.weight)

        # AFW
        self.AFW = torch.matmul(self.AF, self.weight).detach().numpy()
        return Func.relu(torch.Tensor(normalize(self.AFW)))