import torch.nn.functional as F
import torch.optim as optimizer
from model import *

torch.manual_seed(111)

# data
# 노드 연결 정보 # 변하지 않음.
#edges = np.array([[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [4, 6]])
# # # 각 노드 특성 정보(H) = 7 X 4 # 시간 지나면서 계속 변함
#features = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
# # # edge 특성 정보 # 시간 지나면서 계속 변함
#edge_features = [[3], [5], [1], [10], [6], [8]]
# # # labels # train용은 안 변함.
#labels = np.array([1, 4, 5, 2, 6, 3, 0])

edges = np.array([[0,1], [1,2], [2,4], [3,4], [4,5], [5,6], [5,7]])
features = sp.csr_matrix([[1,2,3],[2,3,5],[1,3,7],[4,5,7],[0,2,3],[5,6,7],[8,3,1],[9,5,7]])
edge_features = [[0],[0],[1],[2],[6],[4],[10]]
labels = np.array([2, 3, 5, 0, 7, 4, 2, 10])

# edges = np.array([[0,1],[1,2],[2,3],[3,4],[3,5],[3,6],[4,7],[5,8],[6,9],[9,10]])
# features = sp.csr_matrix([[5,3,2],[1,5,7],[6,5,8],[9,8,2],[5,6,2],[8,6,5],[3,4,7],[9,1,4],[7,7,3],[1,6,5],[9,10,1]])
# edge_features = [[2],[1],[3],[15],[7],[9],[5],[8],[3],[6]]
# labels = np.array([1, 1, 3, 1, 4, 2, 9, 12, 5, 6, 13])



# 전처리
model = mymodel(edges, features, edge_features, 'mean') # sum, mean, min, max, linear
# hop 2회 진행
x = model.forward()

# train
# 학습 parameter 설정
weight = Parameter(torch.Tensor(np.random.rand(x.size()[1], 1)))
bias = Parameter(torch.Tensor(np.random.rand(x.size()[0], 1)))
optims = optimizer.SGD([weight, bias], lr=0.01)

torch.nn.init.xavier_uniform_(weight)

print(x)
## train
for i in range(10000 + 1):
    hypothesis = torch.matmul(x, weight) + bias
    cost = F.mse_loss(hypothesis.reshape(len(labels), ),
                      torch.FloatTensor(labels))

    optims.zero_grad()
    cost.backward()
    optims.step()

    if i % 10 == 0:
        print('epoch : {} cost : {}'.format(i, cost))

print('=' * 50)
print('예측값 : {}'.format(hypothesis.flatten()))
print('실제값 : {}'.format(labels))
print('최종 cost값 : {}'.format(cost))