import torch
from torch.nn.parameter import Parameter
import torch_sparse
import math
import numpy as np
# from .. import init

# 노드 feature만 사용
class mymodule():
    # input 형태
    # A : csr_matrix, features, labels : Tensor, weight_channel : int
    def __init__(self, A, features, labels, weight_channel):
        super(mymodule, self).__init__()
        self.A = A
        self.features = features # tensor
        self.labels = labels

        # weight 크기 설정에 따라 node_feature 가 업데이트 될 수 있게
        self.weight_channel = weight_channel

        print('=============A=============')
        print(self.A)
        print('=============features=============')
        print(self.features)

        print('A.size : {}'.format(self.A))
        print('features.size : {}'.format(self.features.size()))
        print('weight_channel : {}'.format(self.weight_channel))

        # A(a x b) * features(b x c) * weight( __ x d ) 일 때 __는 a
        # bias(__, 1)
        self.weight = Parameter(torch.Tensor([self.A.size,
                                             self.weight_channel]))
        self.bias = Parameter(torch.Tensor([self.A.size, 1]))

        # A(a x b) * features(b x c)

        self.cal1 = torch.matmul(self.cal1, self.features)
        self.cal2 = torch.matmul(self.cal1, self.weight)
        self.cal3 = self.cal2 + self.bias




        # self.reset_parameters()


    # def reset_parameters(self) -> None:
    #     # https://pytorch.org/docs/stable/_modules/torch/nn/modules/linear.html#Linear
    #     # reset_parameters() 참고
    #
    #     init.kaiming_uniform_(self.weight, a=math.sqrt(5))
    #     if self.bias is not None:
    #         fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
    #         bound = 1 / math.sqrt(fan_in)
    #         init.uniform_(self.bias, -bound, bound)

    # 1 : A * feature * weight + bias
        ## 1.1 : A.shape, feature.shape, weight.shape, bias.shape


    # 2 : forward

    # 3 : initialize_weight




# 외부에서 해야하는 작업
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD() # Adam optimizer