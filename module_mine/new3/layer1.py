import numpy as np
import torch
from torch.nn.parameter import Parameter
from torch.nn import  ReLU
import torch.nn.functional as Func

class layer_1(torch.nn.Module):
    def __init__(self, Adj, matrix):
        super(layer_1, self).__init__()
        self.A = Adj
        self.matrix = matrix

    def forward(self) -> torch.Tensor:

        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.matrix)))
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))

        # weight 초기화
        torch.nn.init.xavier_uniform_(self.weight)

        self.AFW = torch.matmul(self.AF, self.weight).detach().numpy()

        return self.AFW

