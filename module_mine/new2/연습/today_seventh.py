import numpy as np
import torch
from torch.nn.parameter import Parameter
import torch.nn.functional as Func
import scipy.sparse as sp
import torch.optim as optim


def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드 정보 개수
    r_inv = np.power(rowsum, 0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어줌
    mx_update = r_mat_inv.dot(mx)
    return mx_update


class mymodel7(torch.nn.Module):
    torch.manual_seed(111)

    def __init__(self, A, F, labels, epoch):
        super(mymodel7, self).__init__()
        self.A = normalize(A).toarray()  # 인접행렬
        self.F = normalize(F).toarray()  # feature정보
        self.labels = torch.LongTensor(labels)  # 학습을 위한 label
        self.epoch = epoch

        print('A size : {}, F size : {}'.format(A.shape, F.shape))  # A, F 사이즈 확인
        print('=' * 50)
        print(self.A)
        print(self.F)

    def AFWlayer(self):
        print('start AFWlayer')
        self.AF = torch.FloatTensor(np.array(np.matmul(self.A, self.F)))  # AF
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            self.AF.size()[1])))
        torch.nn.init.xavier_uniform_(self.weight)
        print('=' * 50)
        print(self.AF)
        print('=' * 50)

        return torch.matmul(self.AF, self.weight).detach().numpy()  # AF X Weight

    def forward(self) -> torch.Tensor:
        # 2번 반복 AFW
        for hop in range(2):
            print('hop{} forward 시작 전 shape : {} '.format(hop + 1, self.F.shape))
            self.F = Func.relu(torch.Tensor(self.AFWlayer()))
            print(self.F)
            print('hop{} forward 완료 후 shape : {}'.format(hop + 1, self.F.shape))
        return self.F

    def regression(self, lr) -> object:
        print('labels.size : {}'.format(self.labels.shape))
        self.weight = Parameter(torch.Tensor(np.random.rand(self.AF.size()[1],
                                                            1)))
        self.bias = Parameter(torch.Tensor(np.random.rand(self.F.size()[0],
                                                          1)))
        print(self.weight.shape, self.bias.shape)

        self.learning_rate = lr
        self.optim = optim.SGD([self.weight, self.bias], lr=1e-5)

        for i in range(self.epoch + 1):
            self.hypothesis = torch.matmul(self.F, self.weight) + self.bias
            ###print('hypothesis.size : {}'.format(self.hypothesis.shape))
            self.cost = torch.mean((self.hypothesis - self.labels)**2)

            self.optim.zero_grad()
            self.cost.backward()
            self.optim.step()

            if i % 1000 == 0 :
                print('Epoch {}/{} Cost: {:.6f}'.format(i, self.epoch, self.cost))

        return self.hypothesis, self.cost