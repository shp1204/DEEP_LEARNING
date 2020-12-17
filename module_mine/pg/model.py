import torch.nn as nn
import torch
from concatenating import *
from make_layer import *


class model(nn.Module):
    def __init__(self, hop_num, edges, features, edge_features,
                 F_aggr_method, EF_aggr_method):
        super(model, self).__init__()

        self.hop = hop_num
        self.A = edges  # np.array
        self.F = features  # sp.csr_matrix
        self.EF = edge_features  # list
        self.Fmethod = F_aggr_method
        self.EFmethod = EF_aggr_method


        # hop 개수만큼 layer 생성
        self.cfg = []
        for turn in range(1, self.hop+1):
            # convolution 할 숫자를 append
            self.cfg.append('layer')

        self.cfg.append('final_layer')

        print(self.cfg) # [ layer, layer, final_layer ]


        for turn in self.cfg:
            self.F, self.EF = concatenating(self.A, self.F, self.EF,'mean', 'mean').result() # concat F, EF 가 나오면

            print(self.F.shape) # 13, 4
            print(self.EF.shape) # 13, 2

            if turn != 'final_layer':
                pass
                # CNN
                # layer
                # features convolution
                # edge_features convolution -> (shape 이 F, EF와 같도록)
            else :
                pass
                # final_layer
                # convolution -> (1, 1)

    def forward(self,x):
        x = self.features(x)
        x = self.avgpool(x) # 제외하고 진행
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
