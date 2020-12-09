import numpy as np
import scipy.sparse as sp
import torch.nn.functional as F


# data
#노드 연결 정보
edges = np.array([[0 ,1],[2 ,3],[1 ,4],[3 ,4],[4 ,5],[4 ,6]])
# 각 노드 특성 정보(H) = 7 X 4
features = sp.csr_matrix([[1, 0, 0, 0],[0, 1, 0, 0],[1, 0, 0, 0],[0, 1, 0, 0],[0, 0, 1, 0],[0, 0, 0, 1],[0, 0, 0, 1]])
# edge 특성 정보
edge_features = [[3],[5],[1],[10],[6],[8]]
# labels
labels = np.array([1,4,5,2,6,3,0])

# 단위 행렬 더해주기
# direction matrix 생성
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
             shape = (labels.shape[0], labels.shape[0]),
             dtype = np.float32)
adj = adj + sp.eye(adj.shape[0])
print(adj)
print(features)

# from today_seventh import *
#
# aa = mymodel7(adj, features, labels, 10000) #  input
# aa.forward() # hop 계산
# result = aa.regression(0.01) # regression
# print('=====================================================')
# print(result)

#####################################################################
import torch.optim as optimizer
from today_eight import *

# hop 2개 계산
model = mymodel8(adj, features, labels, 1000)
x = model.forward()


# 학습 parameter 설정
weight = Parameter(torch.Tensor(np.random.rand(x.size()[1],
                                               1)))
bias = Parameter(torch.Tensor(np.random.rand(x.size()[0],
                                             1)))
optims = optimizer.SGD([weight, bias], lr=0.01)

# regression
for i in range(1000 + 1):
    hypothesis = torch.matmul(x, weight) + bias
    cost = F.mse_loss(hypothesis.reshape(7,) , torch.FloatTensor(labels))

    optims.zero_grad()
    cost.backward()
    optims.step()

    if i % 100 == 0 :
        print('epoch : {} cost : {}'.format(i, cost))


print('='*50)
print('예측값 : {}'.format(hypothesis.flatten()))
print('실제값 : {}'.format(labels))
print('최종 cost값 : {}'.format(cost))














