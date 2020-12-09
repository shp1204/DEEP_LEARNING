import numpy as np
import torch

class calculate(torch.nn.Module):
    def __init__(self, method, new_features):

        self.method = method
        self.new_features = new_features

    def cal(self):

        cal_result = []
        if self.method == 'sum':
            for idx in self.new_features:
                if len(idx) >= 2:
                    cal_result.append([sum(idx)])
                else:
                    cal_result.append(idx)
        elif self.method == 'mean':
            for idx in self.new_features:
                if len(idx) >= 2:
                    cal_result.append([np.mean(idx)])
                else:
                    cal_result.append(idx)
        elif self.method == 'min':
            for idx in self.new_features:
                if len(idx) >= 2:
                    cal_result.append([min(idx)])
                else:
                    cal_result.append(idx)
        elif self.method == 'max':
            for idx in self.new_features:
                if len(idx) >= 2:
                    cal_result.append([max(idx)])
                else:
                    cal_result.append(idx)
        else:
            print('Not expected method. Expected [sum, mean, min, max].')

        return cal_result