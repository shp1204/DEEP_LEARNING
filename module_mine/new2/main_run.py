
import torch.nn.functional as F
import torch.optim as optimizer
# from today_eight import *
from version2_1 import *
from version2_2 import *
import torch


print('=====================================')

torch.manual_seed(111)

# data
#노드 연결 정보 # 변하지 않음.
edges = np.array([[0 ,1],[2 ,3],[1 ,4],[3 ,4],[4 ,5],[4 ,6]])
# 각 노드 특성 정보(H) = 7 X 4 # 시간 지나면서 계속 변함
features = sp.csr_matrix([[1, 0, 0, 0],[0, 1, 0, 0],
                          [1, 0, 0, 0],[0, 1, 0, 0],
                          [0, 0, 1, 0],[0, 0, 0, 1],
                          [0, 0, 0, 1]])
# edge 특성 정보 # 시간 지나면서 계속 변함
edge_features = [[3],[5],[1],[10],[6],[8]]
# labels # train용은 안 변함.
labels = np.array([1,4,5,2,6,3,0])


# 단위 행렬 더해주기
# direction matrix 생성
adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:,0], edges[:,1])),
             shape = (labels.shape[0], labels.shape[0]),
             dtype = np.float32)
adj = adj + sp.eye(adj.shape[0])
print('=============== adj =================')
print(adj)
print('=============== features =================')
print(features)


# mymodel8 : version2_0
# hop 2개의 정보를 계산
#model = mymodel8(adj, features)
#x = model.forward()

# mymodel9 : version2_1
# edge_feature 정보도 포함
#model = mymodel9(adj, features, edge_features)
#model.concat()
#x = model.forward()

# mymodel10 : version2_2
# edge_feature를 각 노드 feature에 할당한다. [0, 1] 사이의 edge 정보라면 0에 할당
# 그룹화 어떻게 해줄지?
# 분기점 [4, 5], [4, 6]의 경우 4에 해당하는 edge가 두 개이므로 sum, mean, min, max 등으로 설정할 수 있음
# node_feature에 이 부분을 반영한 뒤 학습 진행
model = mymodel10(adj, features, edge_features, edges)
model.aggregate('max')
x = model.forward()

# Regression
## 학습 parameter 설정
weight = Parameter(torch.Tensor(np.random.rand(x.size()[1],1)))
bias = Parameter(torch.Tensor(np.random.rand(x.size()[0],1)))
optims = optimizer.SGD([weight, bias], lr=0.01)

## train
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