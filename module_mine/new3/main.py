import torch.nn.functional as F
import torch.optim as optimizer
from model import *

torch.manual_seed(111)

# data
# 노드 연결 정보 # 변하지 않음.
edges = np.array([[0, 1], [2, 3], [1, 4],
                  [3, 4], [4, 5], [4, 6]])

# 각 노드 특성 정보(H) = 7 X 4 # 시간 지나면서 계속 변함
features = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0],
                          [1, 0, 0, 0], [0, 1, 0, 0],
                          [0, 0, 1, 0], [0, 0, 0, 1],
                          [0, 0, 0, 1]])

# edge 특성 정보 # 시간 지나면서 계속 변함
edge_features = [[3], [5], [1], [10], [6], [8]]

# labels # train용은 안 변함.
labels = np.array([1, 4, 5, 2, 6, 3, 0])

# 단위 행렬 더해주기
# direction matrix 생성
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                    shape=(labels.shape[0], labels.shape[0]),
                    dtype=np.float32)
adj = adj + sp.eye(adj.shape[0])

# model
# adj, features
# edge_features 와 adj matching에 edges 활용
# sum으로 하면 nan값 발생
model = mymodel(adj, features, edge_features, edges, 'mean')
x = model.forward()

print(x)

# 학습 parameter 설정
weight = Parameter(torch.Tensor(np.random.rand(x.size()[1], 1)))
bias = Parameter(torch.Tensor(np.random.rand(x.size()[0], 1)))
optims = optimizer.SGD([weight, bias], lr=0.01)

torch.nn.init.xavier_uniform_(weight)

## train
for i in range(1000 + 1):
    hypothesis = torch.matmul(x, weight) + bias
    cost = F.mse_loss(hypothesis.reshape(7, ),
                      torch.FloatTensor(labels))

    optims.zero_grad()
    cost.backward()
    optims.step()

    if i % 100 == 0:
        print('epoch : {} cost : {}'.format(i, cost))

print('=' * 50)
print('예측값 : {}'.format(hypothesis.flatten()))
print('실제값 : {}'.format(labels))
print('최종 cost값 : {}'.format(cost))
