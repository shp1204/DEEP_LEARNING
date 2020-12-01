from typing import Union, Tuple
from torch_geometric.typing import OptPairTensor, Adj, Size

from torch import Tensor
from torch.nn import Linear
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.nn.conv import MessagePassing


class SAGEConv(MessagePassing):

    # in_channels : Union[size of source, target dimensionalities]
    # out_channels : size of each output sample
    # normalize = True : l2정규화
    # bias = True : bias 더함
    def __init__(self, in_channels: Union[int, Tuple[int, int]],
                 out_channels: int,
                 normalize: bool = False,
                 bias: bool = True,
                 **kwargs):  # yapf: disable
        kwargs.setdefault('aggr', 'mean')
        super(SAGEConv, self).__init__(**kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize

        # int형인지 체크
        if isinstance(in_channels, int):
            in_channels = (in_channels, in_channels)
            print('in_channels : {}'.format(in_channels)) # [4, 4]

        self.lin_l = Linear(in_channels[0], out_channels, bias=bias)
        self.lin_r = Linear(in_channels[1], out_channels, bias=False)

        self.reset_parameters()

        print('lin_l : {}'.format(self.lin_l)) # 4, 7
        print('lin_r : {}'.format(self.lin_r)) # 4, 7

    def reset_parameters(self):

        self.lin_l.reset_parameters()
        self.lin_r.reset_parameters()

    def forward(self, x: Union[Tensor, OptPairTensor], edge_index: Adj, size: Size = None) -> Tensor:
        """"""
        if isinstance(x, Tensor):
            x: OptPairTensor = (x, x)  # 이거 뭐한느 역할, 문법 특이하게 생김

        # propagate_type: (x: OptPairTensor)
        print(edge_index) # edge 연결정보
        print(x) # 노드 features
        out = self.propagate(edge_index, x=x, size=size)  # edge_index는 propagate 오류남
        out = self.lin_l(out)  # lin_l에 변수 두개 들어가야하는데 하나만 들어감 어떻게 계산?

        x_r = x[1]  # output dimension
        if x_r is not None:  # output 차원이 1이상인 경우에 대해서만
            out += self.lin_r(x_r)  # 얘도 변수 두개 들어가얗나느데 하나만 들어감. ???

        if self.normalize:
            out = F.normalize(out, p=2., dim=-1)

        return out

    def message(self, x_j: Tensor) -> Tensor:  # x_j 어디서 온건가 ? tensor?
        return x_j  # message 보내지는건가 ?

    def message_and_aggregate(self, adj_t: SparseTensor, x: OptPairTensor) -> Tensor:
        adj_t = adj_t.set_value(None, layout=None)
        return matmul(adj_t, x[0], reduce=self.aggr)

    # sparsetensor vs optpairtensor ? 현재 데이터의 tensor 는 어떤 종류 ?

    def __repr__(self):

        return '{}({}, {})'.format(self.__class__.__name__,
                                   self.in_channels,
                                   self.out_channels)
