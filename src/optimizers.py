import torch as t
import torch.optim
import torch.nn as nn



class Sgd(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(Sgd, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad


class SgdMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(SgdMomentum, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(Adam, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad

class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(RMSProp, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad

class AdaGrad(torch.optim.Optimizer):
    def __init__(self, params, lr):
        self.lr = lr
        params = list(params) #generator
        self.params = params
        defaults= dict(lr = lr, params = self.params)
        super(AdaGrad, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.params[i] -= self.lr*grad