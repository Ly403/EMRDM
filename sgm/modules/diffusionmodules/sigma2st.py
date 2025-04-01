
import torch

class Sigma2St:
    def __call__(self, sigma: torch.Tensor):
        raise NotImplementedError("Base Class Does Not Have `call` Function!")

    def get_derivative_st(self):
        raise NotImplementedError("Base Class Does Not Have `get_derivative_st` Function!")

class EDMSigma2St(Sigma2St):
    def __init__(self, alpha: float=1.0):
        self.alpha = alpha
        
    def __call__(self, sigma: torch.Tensor):
        # sigma = t
        # st = exp(-t)
        return 1. / (1. + self.alpha * sigma)

    def get_derivative_st(self):
        return lambda t: - self.alpha / ((1. + self.alpha * t) ** 2)
    
class PSigma2St(Sigma2St):
    def __init__(self, p, q=1):
        self.p = p
        self.q = q
    
    def __call__(self, sigma: torch.Tensor):
        q = self.q
        p = self.p
        temp =  (q ** (1 - p)) * (sigma ** p)
        return (1. / 1. + temp)
    
    def get_derivative_st(self):
        q = self.q
        p = self.p
        def func(t):
            temp =  (q ** (1 - p)) * (t ** p)
            coeff = - 1. / ((1. + temp) ** 2)
            return coeff * (p * (q ** (1 - p)) * (t ** (p - 1)))
        return func

class NaiveSigma2St(Sigma2St):
    """
        这是用于消融的类，效果是设置st = 1
    """
    def __call__(self, sigma: torch.Tensor):
        return torch.ones_like(sigma)
    
    def get_derivative_st(self):
        return lambda t: torch.zeros_like(t)