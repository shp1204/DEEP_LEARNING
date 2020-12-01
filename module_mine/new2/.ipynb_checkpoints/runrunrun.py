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


# run the program
if __name__ == "__main__":

    # data
    # 노드 연결 정보
    edges = np.array([[0, 1], [2, 3], [1, 4], [3, 4], [4, 5], [4, 6]])
    # 각 노드 특성 정보(H) = 7 X 4
    features = sp.csr_matrix(
        [[1, 0, 0, 0], [0, 1, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [0, 0, 0, 1]])
    # edge 특성 정보
    edge_features = [[3], [5], [1], [10], [6], [8]]
    # labels
    labels = np.array([1, 4, 5, 2, 6, 3, 0])

    # 단위 행렬 더해주기
    # direction matrix 생성
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
                        shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)
    adj = adj + sp.eye(adj.shape[0])


    features = normalize(features)
    adj = normalize(adj)  # 대각행렬

    # 행렬곱을 위한 array
    adj_arr = adj.toarray()
    features_arr = features.toarray()

    # hop 에 따라 정보 전달
    hop = 2
    for idx in range(hop):
        features_arr = np.matmul(adj_arr, features_arr)  # 여기에 weight 곱하기, bias 더하기
        print('======{0}번째======'.format(idx + 1))
        print('{}'.format(features_arr))

    idx_train = range(5)
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels)  # 원핫인코딩 된 label 중 해당하는 label이 몇 번 째인지
    print(labels)

    #t = sparse_mx_to_torch_sparse_tensor(adj)

    ##############################################################################
    ##############################################################################

    criterion = torch.nn.CrossEntropyLoss()
    learning_rate = 1e-5
    #optimizer = torch.optim.SGD(model.paraeters(), lr=learning_rate)
    num_epochs = 2
    #num_batches = len(train_loader)

    for epoch in range(num_epochs):
        print(epoch)
        x, x_labels = adj, labels
        pred = mymodel(x, x_labels, 2) # 활성화함수 # predict 까지 있어야함
        loss = criterion(pred, x_labels)
        loss.backward()
        # optimizer.step()
        print(loss)
