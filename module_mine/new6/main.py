import torch.nn.functional as Func
import torch.optim as optimizer
from torch.nn import Linear
from preprocessing import *
from torch.nn.parameter import Parameter

# data
# 노드 연결 정보 # 변하지 않음.
# edges = np.array([[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [4, 6]])
# # # 각 노드 특성 정보(H) = 7 X 4 # 시간 지나면서 계속 변함
# features = sp.csr_matrix([[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
# # # edge 특성 정보 # 시간 지나면서 계속 변함
# edge_features = [[3], [5], [1], [10], [6], [8]]
# # # labels # train용은 안 변함.
# labels = np.array([1, 4, 5, 2, 6, 3, 0])

edges = np.array([[0, 1], [1, 2], [2, 4], [3, 4], [4, 5], [5, 6], [5, 7]])
features = sp.csr_matrix([[1, 2, 3], [2, 3, 5], [1, 3, 7], [4, 5, 7], [0, 2, 3], [5, 6, 7], [8, 3, 1], [9, 5, 7]])
edge_features = [[0], [0], [1], [2], [6], [4], [10]]
labels = np.array([2, 3, 5, 0, 7, 4, 2, 10])


# edges = np.array([[0,1],[1,2],[2,3],[3,4],[3,5],[3,6],[4,7],[5,8],[6,9],[9,10]])
# features = sp.csr_matrix([[5,3,2],[1,5,7],[6,5,8],[9,8,2],[5,6,2],[8,6,5],[3,4,7],[9,1,4],[7,7,3],[1,6,5],[9,10,1]])
# edge_features = [[2],[1],[3],[15],[7],[9],[5],[8],[3],[6]]
# labels = np.array([1, 1, 3, 1, 4, 2, 9, 12, 5, 6, 13])

def normalize(X):
    nom = X - X.min(axis=0)
    denom = X.max(axis=0) - X.min(axis=0)
    denom[denom == 0] = 1
    return nom / denom


# 전처리
method = 'mean'

features = normalize(features.toarray())
pp = preprocessing(edges, features, edge_features, method)  # method ? sum, mean, min, max, linear

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

# 행
row = len(labels)
# 열
col = features.shape[1] + 1


# train
# 학습 parameter 설정
# weight 5개로 설정 -> 추후에 hop개수에 따라 설정 가능하도록 변경


class model(nn.Module):
    def __init__(self, A, features, edge_features):
        super(model, self).__init__()
        self.A = A
        self.features = features
        self.edge_features = edge_features

        self.AF = np.matmul(A, features)
        self.AeF = np.matmul(A, edge_features)

        self.layer1 = nn.Sequential(nn.Linear(self.AF.shape[1], self.AF.shape[1], bias=False),
                                    nn.ReLU(),
                                    nn.Linear(self.AF.shape[1], self.AF.shape[1], bias=False),
                                    nn.ReLU())

        self.layer2 = nn.Sequential(nn.Linear(self.AeF.shape[1], self.AeF.shape[1], bias=False),
                                    nn.ReLU(),
                                    nn.Linear(self.AeF.shape[1], self.AeF.shape[1], bias=False),
                                    nn.ReLU())


        self.weight = Parameter(torch.Tensor(np.random.rand(col, 1)))
        self.bias = Parameter(torch.Tensor(np.random.rand(row, 1)))

        self.regression = torch.matmul(torch.Tensor(np.concatenate((self.AF, self.AeF), axis=1)), self.weight) + self.bias

    def forward(self):
        seq = self.regression()
        return seq

mymodel = model(A, features, edge_features)
print(mymodel)
optim = optimizer.Adam(mymodel.parameters(), lr=0.01)

for i in range(10 + 1):

    optim.zero_grad()

    outputs = mymodel()
    loss = Func.mse_loss(outputs, labels)
    loss.backward()
    optim.step()

    # if i % 10 == 0:
    print('============epoch : {} cost : {}=========='.format(i, loss))

print('=' * 50)
print('예측값 : {}'.format(outputs.flatten()))
print('실제값 : {}'.format(labels))
print('최종 cost값 : {}'.format(loss))
