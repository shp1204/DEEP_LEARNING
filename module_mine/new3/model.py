import torch.nn as nn
from layer1 import *
from calculate import *
import scipy.sparse as sp

def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드가 갖고 있는 정보 개수
    r_inv = np.power(rowsum, 0.5).flatten()
    r_inv[np.isinf(r_inv)] = 0
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어 준다
    mx_update = r_mat_inv.dot(mx)
    return mx_update

class mymodel(nn.Module):
    def __init__(self, A, F, edge_F, rawA, method):
        super(mymodel, self).__init__()

        self.A = normalize(A).toarray()  # 인접행렬
        self.F = normalize(F).toarray()  # feature정보
        self.edge_F = torch.Tensor(edge_F)  # edge_feature 정보
        self.new_features = [[0]] * F.shape[0]

        # 각 idx에 해당하는 edge_feature 값을 넣어줌
        for idx, A_idx in enumerate(rawA):
            edge_info = A_idx[0]
            if self.new_features[edge_info] != [0]:
                self.new_features[edge_info].append(edge_F[idx][0])
            else:
                self.new_features[rawA[idx][0]] = edge_F[idx]

        self.new_features = calculate(method, self.new_features).cal()
        print('각 노드에 해당하는 edge_features : {}'.format(self.new_features))

    def forward(self )-> torch.Tensor:

        # hop1
        self.F = layer_1(self.A, self.F).forward()
        self.F = Func.relu(torch.Tensor(self.F))
        self.new_features = layer_1(self.A, self.new_features).forward()

        print(self.F)
        print(self.new_features)

        # hop2
        self.F = layer_1(self.A, self.F).forward()
        self.F = Func.relu(torch.Tensor(self.F))
        self.new_features = layer_1(self.A, self.new_features).forward()

        print(self.F)
        print(self.new_features)

        # aggregate
        self.newF = torch.cat([torch.Tensor(self.F),
                               torch.Tensor(self.new_features)], dim=-1)

        return self.newF