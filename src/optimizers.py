import torch as t
import torch.optim
import torch.nn as nn


<<<<<<< HEAD


=======
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

        @t.inference_mode()
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




class SgdMomentum(torch.optim.Optimizer):
    def __init__(self, params, lr, momentum=0.9):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0 or momentum >= 1:
            raise ValueError("Invalid momentum value: {}".format(momentum))

        defaults = dict(lr=lr, momentum=momentum)
        super(SgdMomentum, self).__init__(params, defaults)
    
    @t.inference_mode()
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

>>>>>>> 16889452ed75fb765101dcaceaf68f5b54ab24bd

class Adam(torch.optim.Optimizer):
    def __init__(self, params, lr, b1=0.9, b2=0.999, ep=10e-8):
        params = list(params) #generator
        self.params = params
        self.t = 0
        self.m = [t.zeros_like(p) for p in self.params] #first moment vector
        self.v = [t.zeros_like(p) for p in self.params] #second moment vector
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.ep = ep
        defaults = dict(params = self.params, lr = lr, b1 = b1, b2 = b2, ep = ep)
        super(Adam, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        self.t += 1
        for i,p in enumerate(self.params):
            grad = p.grad
            self.m[i] = self.b1*self.m[i] + (1-self.b1)*grad
            self.v[i] = self.b2*self.v[i] + (1-self.b2)*(grad**2)
            m_hat = self.m[i]/(1-self.b1**self.t)
            v_hat = self.v[i]/(1-self.b2**self.t)
            self.params[i] -= self.lr*grad*m_hat/(t.sqrt(v_hat)+self.ep)

class RMSProp(torch.optim.Optimizer):
    def __init__(self, params, lr=0.01, alpha=0.99, ep=1e-8):
        params = list(params) # generator
        self.params = params
        self.sqa = [t.zeros_like(p) for p in self.params] # square average
        self.lr = lr
        self.alpha = alpha
        self.ep = ep
        defaults= dict(params = self.params, lr = lr, alpha = alpha, ep = ep)
        super(RMSProp, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.sqa[i] = self.alpha*self.sqa[i] + (1 - self.alpha)*(grad**2)
            self.params[i] -= self.lr*grad/(t.sqrt(self.sqa[i]) + self.ep)


class AdaGrad(torch.optim.Optimizer):
<<<<<<< HEAD
    def __init__(self,params, lr): 
        defaults = dict(lr = lr)
=======
    def __init__(self, params, lr=0.01, ep=1e-8):
        params = list(params) #generator
        self.params = params
        self.lr = lr
        self.ep = ep
        self.st = [t.zeros_like(p) for p in self.params] #state sum
        defaults= dict(params = self.params, lr = lr, ep = ep)
>>>>>>> 16889452ed75fb765101dcaceaf68f5b54ab24bd
        super(AdaGrad, self).__init__(params, defaults)

    @t.inference_mode()
    def step(self):
        for i,p in enumerate(self.params):
            grad = p.grad
            self.st[i] += grad**2
            self.params[i] -= self.lr*grad/(t.sqrt(self.st[i]) + self.ep)