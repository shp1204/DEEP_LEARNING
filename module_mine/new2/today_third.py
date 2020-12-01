import torch
from torch.nn.parameter import Parameter
# import torch_sparse
# import math
import numpy as np
# from .. import init

# A, features 미리 곱해서 학습을 시키는 경우
# A x features 행렬을 tensor에 올림
class mymodule():
    # input 형태
    # AF : csr_matrix, labels : Tensor, weight_channel : int
    torch.manual_seed(111)
    def __init__(self, AF, labels, weight_channel):
        super(mymodule, self).__init__()
        self.AF = AF
        self.labels = labels

        # weight 크기 설정에 따라 node_feature 가 업데이트 될 수 있게
        self.weight_channel = weight_channel

        print('============= Input 값 확인 =============')
        print('AF.size : {}'.format(self.AF.size))
        print('weight_channel : {}'.format(self.weight_channel))


        print('==================csr_matrix to tensor==============================')
        # csr_matrix to tensor
        self.temp1 = self.AF.tocoo()
        self.temp2 = torch.sparse.Tensor([self.temp1.row.tolist(),
                                        self.temp1.col.tolist()])
        print('temp1.row : {}'.format(len(self.temp1.row.tolist())))
        print('temp1.col : {}'.format(len(self.temp1.col.tolist())))
        print('temp2.size : {}'.format(self.temp2.shape))


        print('======================weight, bias 설정==============================')
        #print(self.temp1.row)
        #print(self.temp1.col)

        # Weight, bias 설정
        # A(a x b) * features(b x c) * weight( c x __ )
        # bias(__, 1)
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size,
                                                            self.weight_channel)))
        self.bias = Parameter(torch.Tensor(np.random.rand(self.weight_channel,
                                                          1)))

        #print('weight.size : {}'.format(self.weight.size[:2]))
        #print('bias.size : {}'.format(self.bias.size[:2]))
        #print('weight')
        #print(self.weight)
        print(self.bias)
        print(self.bias.unsqueeze(1))
        self.bias = self.bias.unsqueeze(1)

        print('========================= calculate ======================')
        # AF X Weight + bias
        self.cal = torch.matmul(self.temp2, self.weight)
        print(self.cal)
        self.cal2 = torch.add(self.cal, self.bias)

        print(self.cal2)

        print('========================= process =========================')
        print('{} x {} = {}'.format(self.AF, self.weight, self.cal))



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

    # 2 : forward




    # 3 : initialize_weight




# 외부에서 해야하는 작업
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD() # Adam optimizer