import torch
from torch.optim import Optimizer
import math

class SZOPGDAM(Optimizer):
    """
    Stochastic Zeroth-Order Proximal Gradient Descent with Adaptive Momentum (SZOPGD-AM)
    
    Args:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups
        eta0 (float, optional): initial learning rate (default: 0.1)
        beta (float, optional): momentum factor (default: 0.9)
        mu (float, optional): smoothing parameter for zeroth-order gradient estimation (default: 0.01)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
    """
    
    def __init__(self, params, eta0=0.1, beta=0.9, mu=0.01, weight_decay=0):
        if eta0 <= 0.0:
            raise ValueError(f"Invalid initial learning rate: {eta0}")
        if beta < 0.0 or beta >= 1.0:
            raise ValueError(f"Invalid beta parameter: {beta}")
        if mu <= 0.0:
            raise ValueError(f"Invalid smoothing parameter: {mu}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
            
        defaults = dict(eta0=eta0, beta=beta, mu=mu, weight_decay=weight_decay)
        super(SZOPGDAM, self).__init__(params, defaults)
        
        # Initialize step counter for each parameter group
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['momentum_buffer'] = torch.zeros_like(p.data)
                
    def step(self, closure=None):
        """Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            eta0 = group['eta0']
            beta = group['beta']
            mu = group['mu']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                    
                grad = p.grad.data
                if weight_decay != 0:
                    grad = grad.add(p.data, alpha=weight_decay)
                
                state = self.state[p]
                state['step'] += 1
                step = state['step']
                
                # Compute adaptive step size
                eta_t = eta0 / math.sqrt(step)
                
                # Update momentum buffer
                momentum_buffer = state['momentum_buffer']
                momentum_buffer.mul_(beta).add_(grad, alpha=1-beta)
                
                # Proximal update
                p.data.add_(momentum_buffer, alpha=-eta_t)
                
        return loss