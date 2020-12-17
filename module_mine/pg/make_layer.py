import torch.nn as nn

class make_layer(nn.Module):
    def __init__(self, batch_size, channel, height, weight):
        super(make_layer, self).__init__()
        self.batch_size = batch_size
        self.channel = channel
        self.height = height
        self.weight = weight

    def conv(self):
        return nn.Conv2d(1, 1, stride=1, padding=1)

    def pool(self): # pooling하면 행과 열 모두 다 작아짐 # 열만 작아져야하는데
        return nn.MaxPool2d(kernel_size=2, stride=2)

    def fc(self):
        return