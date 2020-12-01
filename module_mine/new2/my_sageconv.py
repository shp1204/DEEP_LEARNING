from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size
import torch
from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops

class SAGElayer(torch.nn.Module):
    def __init__(self, in_features, out_features, A):
        super(SAGElayer).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.A = A
        self.fc = torch.nn.Linear(in_features, out_features)

    def forward(self, X):
        print('=====feature 정보=====')
        print(X)
        print('=====인접행렬 정보=====')
        print(self.A)

        return self.fc(torch.spmm(self.A, X))

class my_Sageconv(MessagePassing):
    def __init__(self, in_channels, out_channels, A):
        super(my_Sageconv, self).__init__(aggr='max')



        # self.linear = torch.nn.Linear(in_channels, out_channels)
        # print('self.linear : {}'.format(self.linear))
        # self.activate = torch.nn.ReLU()
        # print('self.activate : {}'.format(self.activate))
        #
        # self.update_linear = torch.nn.Linear(in_channels + out_channels, in_channels, bias=False)
        # self.update_activate = torch.nn.ReLU()
        # print('self.update_linear : {}'.format(self.update_linear))

        self.feature_extractor = torch.nn.Sequential(SAGElayer(in_channels, 16, A),
                                                        torch.nn.ReLU(),
                                                        SAGElayer(16, out_channels, A))


    # edge_index가 있으면 전달
    def forward(self, x : Union[Tensor, OptPairTensor], edge_index) -> Tensor:
        print('start forward')
        return self.feature_extractor(x)



        # x.shape = [N, in_channels]
        # edge_index.shape = [2, E]
        # if isinstance(x, Tensor):
        #     x: OptPairTensor = (x, x)
        #
        # # propagate type : (x : OptPairtensor)
        # out = self.propagate(edge_index, x=x, size=size)
        # print('out propagate : {}'.format(out))
        # out = self.linear(out)
        # print('out linear : {}'.format(out))
        #
        # # output dimension
        # x_r = x[1]
        # if x_r is not None:
        #     out += self.linear(x_r)
        #
        # return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    # linear, activate
    # def message(self, x_j):
    #     # x_j.shape = [E, in_channels]
    #     x_j = self.linear(x_j)
    #     x_j = self.activate(x_j)
    #     return x_j
    #
    # # update
    # def update_info(self, aggr_out, x):
    #     # aggr_out.shape = [N, out_channels]
    #     new_embedding = torch.cat([aggr_out, x], dim=0)
    #     new_embedding = self.update_linear(new_embedding)
    #     new_embedding = self.update_activate(new_embedding)
    #     return new_embedding