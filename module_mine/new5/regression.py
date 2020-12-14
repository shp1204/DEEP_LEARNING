import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as Func
import scipy.sparse as sp
from torch.nn import Linear, ReLU
from torch import Tensor

# A F W
# A eF W

class regression(torch.nn.Module):
    def __init__(self, train_X, train_Y):

        self.A = Adj
        self.matrix = matrix

    def forward(self):
        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.matrix)))
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))

        # weight 초기화
        torch.nn.init.xavier_uniform_(self.weight)
        self.AFW = torch.matmul(self.AF, self.weight).detach().numpy()

        return self.AFW