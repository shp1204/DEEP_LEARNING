from today_fourth import mymodel
import numpy as np
import scipy.sparse as sp
import torch
# normalize
def normalize(mx):
    rowsum = np.array(mx.sum(1))  # 각 노드 정보 개수
    print('=====row별 feature 특성 합=====')
    # print(rowsum)

    # r_inv
    # 역행렬로 np.power 수행
    r_inv = np.power(rowsum,
                     0).flatten()  # 0, 1, # power : 0, 1, 8, 27, ,,, / 0, 1, 4, 9, ,,, / 0, 1, 0.5, 0.333, 0.25
    print('===== 역행렬로 np.power 수행 =====')
    r_inv[np.isinf(r_inv)] = 0
    # print(r_inv)

    # r_mat_inv
    r_mat_inv = sp.diags(r_inv)  # 행렬로 만들어줌
    # print(r_mat_inv.toarray())

    # 노드 adj 와 노드 feature 정보 행렬연산
    print('=====adj, feature 행렬곱=====')
    mx = r_mat_inv.dot(mx)

    print(mx.toarray())
    return mx

# def sparse_mx_to_torch_sparse_tensor(sparse_mx):
#     """Convert a scipy sparse matrix to a torch sparse tensor."""
#     sparse_mx = sparse_mx.tocoo().astype(np.float32)
#
#     # 노드
#     indices = torch.from_numpy(
#         np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))  # vstack : 행 추가
#     # 노드 간 edge의 정보
#     values = torch.from_numpy(sparse_mx.data)  # numpy.ndarray를 tensor로 올려줌
#     # 노드 개수, 특성 개수
#     shape = torch.Size(sparse_mx.shape)
#
#     return torch.sparse.Tensor(indices, values, shape)  # sparse : 크기에 맞게 값을 뿌려주는 것 같은데 규칙 잘 모르겠다

class preprocess():
    def __init__(self, edges, features, labels):
        super(preprocess, self).__init__()

        # 단위 행렬 더해주기
        # direction matrix 생성
        self.adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
        self.adj = self.adj + sp.eye(self.adj.shape[0])

        self.features = normalize(features)

        # 행렬곱을 위한 array
        self.adj_arr = normalize(self.adj).toarray()
        self.features_arr = self.features.toarray()

        self.idx_train = range(5)
        self.features = torch.FloatTensor(np.array(self.features.todense()))
        self.labels = torch.LongTensor(self.labels)  # 원핫인코딩 된 label 중 해당하는 label이 몇 번 째인지
        print(self.labels)

    def forward(self):
        # hop 에 따라 정보 전달
        hop = 2
        for idx in range(hop):
            features_arr = np.matmul(self.adj_arr, self.features_arr)  # 여기에 weight 곱하기, bias 더하기
            print('======{0}번째======'.format(idx + 1))
            print('{}'.format(self.features_arr))



    # t = sparse_mx_to_torch_sparse_tensor(adj)

    ##############################################################################
    ##############################################################################

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-5
    # optimizer = torch.optim.SGD(model.paraeters(), lr=learning_rate)
    num_epochs = 2
    # num_batches = len(train_loader)

    for epoch in range(num_epochs):
        print(epoch)
        x, x_labels = adj, labels
        pred = mymodel(x, x_labels, 2)  # 활성화함수 # predict 까지 있어야함
        loss = criterion(pred, x_labels)
        loss.backward()
        # optimizer.step()
        print(loss)