import torch.nn.functional as Func
import torch.optim as optimizer
from torch.nn import Linear
from preprocessing import *
from torch.nn.parameter import Parameter
# data case1
# 노드 연결 정보 # 변하지 않음.
#edges = np.array([[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [4, 6]])
# # # 각 노드 특성 정보(H) = 7 X 4 # 시간 지나면서 계속 변함
#features = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
# # # edge 특성 정보 # 시간 지나면서 계속 변함
#edge_features = [[3], [5], [1], [10], [6], [8]]
# # # labels # train용은 안 변함.
#labels = np.array([1, 4, 5, 2, 6, 3, 0])

# data case2
edges = np.array([[0,1], [1,2], [2,4], [3,4], [4,5], [5,6], [5,7]])
features = sp.csr_matrix([[1,2,3],[2,3,5],[1,3,7],[4,5,7],[0,2,3],[5,6,7],[8,3,1],[9,5,7]])
edge_features = [[0],[0],[1],[2],[6],[4],[10]]
labels = np.array([2, 3, 5, 0, 7, 4, 2, 10])
features = features.tocoo()
features = torch.sparse.FloatTensor(torch.LongTensor([features.row.tolist(), features.col.tolist()]),torch.LongTensor(features.data.astype(np.int32))).to_dense().float()

# data case3
# edges = np.array([[0,1],[1,2],[2,3],[3,4],[3,5],[3,6],[4,7],[5,8],[6,9],[9,10]])
# features = sp.csr_matrix([[5,3,2],[1,5,7],[6,5,8],[9,8,2],[5,6,2],[8,6,5],[3,4,7],[9,1,4],[7,7,3],[1,6,5],[9,10,1]])
# edge_features = [[2],[1],[3],[15],[7],[9],[5],[8],[3],[6]]
# labels = np.array([1, 1, 3, 1, 4, 2, 9, 12, 5, 6, 13])
#
# def normalize(X):
#     nom = X - X.min(axis=0)
#     denom = X.max(axis=0) - X.min(axis=0)
#     denom[denom == 0] = 1
#     return nom / denom

def normalize(X):
    nom = X - torch.min(X, dim=0, keepdim=True).values
    denom = torch.max(X, dim=0, keepdim=True).values - torch.min(X, dim=0, keepdim=True).values
    denom[denom == 0] = 1
    return nom/denom

# 전처리
method = 'mean'
features = normalize(features)
pp = preprocessing(edges, features, edge_features, method) # method ? sum, mean, min, max, linear

# 단위행렬 더하기
A = pp.coo().toarray()

# edge_feature의 크기를 feature와 맞춰주기
if method == 'linear':
    edge_encoder = Linear(len(edge_features), features.shape[0])
    edge_encoder.reset_parameters()
    edge_features = torch.Tensor(np.array(edge_features))
    edge_features = Func.relu(edge_encoder(edge_features.T).T).detach().numpy()
else:
    edge_features = pp.convert()
edge_features = torch.tensor(edge_features)

# 행
row = len(labels)
# 열
col = features.shape[1] + 1

# train
# 학습 parameter 설정
# weight 5개로 설정 -> 추후에 hop개수에 따라 설정 가능하도록 변경해야함

# hop1
AFW = torch.FloatTensor(np.array(np.matmul(A, features)))
weight1 = Parameter(torch.Tensor(np.random.rand(AFW.size()[1],
                                                AFW.size()[1])))  # weight for hop1 _ F
weight2 = Parameter(torch.Tensor(np.random.rand(AFW.size()[1],
                                                AFW.size()[1])))  # weight for hop2 _ F
# hop2
AeFW = torch.FloatTensor(np.array(np.matmul(A, edge_features)))
weight3 = Parameter(torch.Tensor(np.random.rand(AeFW.size()[1],
                                                AeFW.size()[1])))  # weight for hop2 _ edge_F
weight4 = Parameter(torch.Tensor(np.random.rand(AeFW.size()[1],
                                                AeFW.size()[1])))  # weight for hop2 _ edge_F
# regression
weight5 = Parameter(torch.Tensor(np.random.rand(col, 1)))  # weight for regression
bias = Parameter(torch.Tensor(np.random.rand(row, 1)))

optims = optimizer.Adam([weight1, weight2, weight3, weight4, weight5, bias], lr=0.01)

torch.nn.init.xavier_uniform_(weight1)
torch.nn.init.xavier_uniform_(weight2)
torch.nn.init.xavier_uniform_(weight3)
torch.nn.init.xavier_uniform_(weight4)
torch.nn.init.xavier_uniform_(weight5)

## train
for i in range(1000+1):

    AFW = Func.relu((normalize(torch.matmul(AFW, weight1))))
    AFW = Func.relu((normalize(torch.matmul(AFW, weight2))))
    AeFW = Func.relu((normalize(torch.matmul(AeFW, weight3))))
    AeFW = Func.relu((normalize(torch.matmul(AeFW, weight4))))

    # regression
    hypothesis = torch.matmul(torch.cat((AFW, AeFW), dim=1), weight5) + bias
    # cost 는 hypothesis 에 사용된 weight, bias로 결정되기 때문에 얘네 두개만 학습됨
    # 이전 layer에 사용한 weight의 학습이 불가능
    cost = Func.mse_loss(hypothesis.reshape(len(labels), ),
                        torch.FloatTensor(labels))

    optims.zero_grad()
    cost.backward(retain_graph=True)
    optims.step()

    if i % 100 == 0:
        print('============epoch : {} cost : {}=========='.format(i, cost))
        print(weight1, weight2, weight3, weight4, weight5)

print('=' * 50)
print('예측값 : {}'.format(hypothesis.flatten()))
print('실제값 : {}'.format(labels))
print('최종 cost값 : {}'.format(cost))