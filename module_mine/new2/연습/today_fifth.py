
import torch
from torch.nn.parameter import Parameter
# import torch_sparse
# import math
import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F

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

def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드 정보 개수
    #print('=====row별 feature 특성 합=====')
    #print(rowsum)

    # r_inv
    # 역행렬로 np.power 수행
    # 0, 1, # power : 0, 1, 8, 27, ,,, / 0, 1, 4, 9, ,,, / 0, 1, 0.5, 0.333, 0.25
    r_inv = np.power(rowsum, 0.5).flatten()
    #print(r_inv)

    #print('===== 역행렬로 np.power 수행 =====')
    r_inv[np.isinf(r_inv)] = 0
    #print(r_inv)

    # r_mat_inv
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어줌
   # print(r_mat_inv.toarray())

    # 노드 adj 와 노드 feature 정보 행렬연산
    mx_update = r_mat_inv.dot(mx)

    # print(mx_update.toarray())
    return mx_update


class mymodel5(torch.nn.Module):

    torch.manual_seed(111)
    # weight_channel은 tuple로 설정
    # convolution이 2hop에 걸쳐 일어나기 때문에 2개 받을예정
    def __init__(self, A, F, labels, weight_channel:tuple):
        super(mymodel5, self).__init__()

        print('=================init 시작=====================')
        #self.A = torch.FloatTensor(normalize(A).toarray())
        #self.F = torch.FloatTensor(normalize(F).toarray())
        self.A = normalize(A).toarray()
        self.F = normalize(F).toarray()

        self.rawA = A

        print('self.A')
        print(self.A)

        self.labels = torch.LongTensor(labels)
        self.weight_channel = weight_channel

        print('A size : {}, F size : {}'.format(A.shape , F.shape))

        self.temp1 = np.matmul(self.A, self.F)
        self.AF = torch.FloatTensor(np.array(self.temp1)) # 1회
        print('==============================================')
        print(self.AF)
        print(torch.unsqueeze(self.AF, dim=1))


    def AFWlayer(self, weight_chan):
        print('start AFWlayer')
        self.weight_chan = weight_chan

        # row, col을 넣어줘야함 ( AF의 values )

        self.temp1 = np.array(self.temp1)
        self.temp2 = torch.sparse.Tensor([self.temp1.row.tolist(),
                                          self.temp1.col.tolist()])
        print(self.temp2)

        print(self.AF.size)
        print(self.weight_chan)

        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size,
                                                            self.weight_chan)))
        self.bias = Parameter(torch.Tensor(np.random.rand(self.weight_chan,
                                                          1)))

        torch.nn.init.xavier_uniform_(self.weight)
        torch.nn.init.xavier_uniform_(self.bias)
        print('end AFWlayer')
        print(self.temp2.size(), self.weight.size(), self.bias.size())
        return torch.matmul(self.temp2, self.weight) + self.bias


    def forward(self) -> torch.Tensor:
        # 1번
        print(self.weight_channel)
        for idx, hop in enumerate(self.weight_channel):
            print(hop)
            print('hop{} forward 시작'.format(idx+1))
            # self.F = F.relu(torch.matmul(self.temp2, self.weight) + self.bias)
            self.F = F.relu(torch.Tensor(self.AFWlayer(hop)))
            print('hop{} forward 완료'.format(idx+1))

        return self.F






# 외부에서 해야하는 작업
# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.SGD() # Adam optimizer