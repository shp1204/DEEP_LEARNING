from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch.nn.parameter import Parameter


from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class mymodule():
    def __init__(self, A, features, labels, in_channel, out_channel): # bias
        super(mymodule, self).__init__()
        self.A = A
        self.features = features
        self.labels = labels
        self.in_channel = in_channel
        self.out_channel = out_channel

        self.weight = Parameter(torch.Tensor(out_channel, in_channel)) # random 하게 (out, in) 크기의 weight 생성
        self.cal1 = matmul(self.A, self.features, reduce=self.aggr)
        print(self.cal1)
        self.cal2 = matmul(self.cal1, self.weight, reduce=self.aggr)
        print('==========update result==========')
        print(self.cal2)

        self.aggr = 'sum'
        # assert self.aggr in ['add', 'mean', 'max', None]


        # self.reset_parameters()

    # def reset_parameters(self):

    # def forward(self, x):
    #     if isinstance(x, Tensor):
    #         x: OptPairTensor = (x, x)

    # adj * features * weight + bias
    def update_features(self, x: OptPairTensor) -> Tensor:
        print('feature update')
        return matmul(self.A, self.features, self.weight)

    def __repr__(self):
        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channel,
                                   self.out_channel)