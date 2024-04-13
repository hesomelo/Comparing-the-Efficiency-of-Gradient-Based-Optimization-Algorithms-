import torch as t
import torch.optim
import torch.nn as nn



class Sgd(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        defaults= dict(lr = lr)
        super(Sgd, self).__init__(params, defaults)

    def step():
        x = 1

class SgdMomentum(torch.optim.Optimizer):
    def __init__(self,params, lr):
        defaults = dict(lr = lr)
        super(SgdMomentum, self).__init__(params, defaults)
    def step():
        x = 1

class Adam(torch.optim.Optimizer):
    def __init__(self,params, lr):
        defaults = dict(lr = lr)
        super(Adam, self).__init__(params, defaults)
    def step():
        x = 1

class RmsProp(torch.optim.Optimizer):
    def __init__(self,params, lr):
        defaults = dict(lr = lr)
        super(RmsProp, self).__init__(params, defaults)
    def step():
        x = 1

class AdaGrad(torch.optim.Optimizer):
    def __init__(self,params, lr):
        defaults = dict(lr = lr)
        super(AdaGrad, self).__init__(params, defaults)
    def step():
        x = 1