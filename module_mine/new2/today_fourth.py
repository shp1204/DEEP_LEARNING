import torch
from torch.nn.parameter import Parameter
# import torch_sparse
# import math
import numpy as np


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

class mymodel(torch.nn.Module):
    torch.manual_seed(111)
    def __init__(self, AF, labels, weight_channel):
        super(mymodel, self).__init__()
        self.AF = AF
        self.labels = labels
        self.weight_channel = weight_channel

        self.temp1 = self.AF.tocoo()
        self.temp2 = torch.sparse.Tensor([self.temp1.row.tolist(),
                                          self.temp1.col.tolist()])

        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size,
                                                            self.weight_channel)))
        self.bias = Parameter(torch.Tensor(np.random.rand(self.weight_channel,
                                                          1)))

        # self.bias = self.bias.unsqueeze(1)
        self.module = torch.matmul(self.temp2, self.weight) + self.bias
        # initialize weight, bias
        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.bias)
        print('2end')

    print('어디로가나?')

    # 2 : forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        print('forward 시작')
        x = self.module(x)
        # 어떤 작업을 하는지
        # cnn : convolution, avgpool, FClayer
        # AF X Weight + bias
        # self.bias = self.bias.unsqueeze(1)
        # x = torch.matmul(x, self.weight) + self.bias
        print('3end')
        return x

    # 3 : backward


    print('end')






# 외부에서 해야하는 작업
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD() # Adam optimizer