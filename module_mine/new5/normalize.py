import torch

class normalize(torch.nn.Module):
    def __init__(self, X):
        super(normalize, self).__init__()
        self.X = X

    def norm(self):

        nom = self.X - self.X.min(axis=0)
        denom = self.X.max(axis=0) - self.X.min(axis=0)
        denom[denom==0] = 1
        return nom/denom