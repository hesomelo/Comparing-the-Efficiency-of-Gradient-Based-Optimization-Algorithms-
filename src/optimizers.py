import torch as t
import torch.optim
import torch.nn as nn



# class Sgd(torch.optim.Optimizer):
    # def __init__(self, params, lr):
    #     self.lr = lr
    #     defaults= dict(lr = lr, params = params)
    #     super(Sgd, self).__init__(params, defaults)

# this includes not only momentum but also other features such as dampening, 
#weight decay, Nesterov's momentum, and support for maximization and these are the parameters we can set as default
class Sgd(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0, dampening=0, weight_decay=0, nesterov=False, maximize=False):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0 <= dampening <= 1:
            raise ValueError("Invalid dampening value: {}".format(dampening))
        
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov, maximize=maximize)
        super(Sgd, self).__init__(params, defaults)


        def step(self, closure=None):
            loss = None
            if closure is not None:
                with torch.enable_grad():
                    loss = closure()
        
            for group in self.param_groups:
                for p in group['params']:
                    if p.grad is None:
                        continue
                    d_p = p.grad.data
                    if group['weight_decay'] != 0:
                        d_p.add_(p.data, alpha=group['weight_decay'])
                    if group['momentum'] != 0:
                        param_state = self.state[p]
                        if 'momentum_buffer' not in param_state:
                            buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                        else:
                            buf = param_state['momentum_buffer']
                            buf.mul_(group['momentum']).add_(d_p, alpha=1 - group['dampening'])
                        if group['nesterov']:
                            d_p = d_p.add(buf, alpha=group['momentum'])
                        else:
                            d_p = buf

                    if group['maximize']:
                        p.data.add_(d_p, alpha=group['lr'])
                    else:
                        p.data.add_(d_p, alpha=-group['lr'])
            
            return loss     

import torch

class SgdMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum >= 1:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(SgdMomentum, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            momentum = group['momentum']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                # Retrieve the momentum buffer
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(d_p, alpha=1 - momentum)

                # Update the parameters
                p.data.add_(buf, alpha=-group['lr'])

        return loss






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