import math
import torch


class RAdam(torch.optim.Optimizer):
    """
    Adapted from https://github.com/LiyuanLucasLiu/RAdam
    """

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)

        super().__init__(params, defaults)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue

                p.data.mul_(1 - group['lr'] * group['weight_decay'])

                grad = p.grad.data
                if grad.is_sparse:
                    raise RuntimeError('RAdam does not support sparse gradients')

                state = self.state[p]

                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p.data)
                    state['exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(1 - beta1, grad)

                state['step'] += 1
                beta2_t = beta2 ** state['step']
                N_sma_max = 2 / (1 - beta2) - 1
                N_sma = N_sma_max - 2 * state['step'] * beta2_t / (1 - beta2_t)

                # more conservative since it's an approximated value
                if N_sma >= 5:
                    step_size = group['lr'] * math.sqrt(
                        (1 - beta2_t) * (N_sma - 4) / (N_sma_max - 4) * (N_sma - 2)
                        / N_sma * N_sma_max / (N_sma_max - 2)) / (1 - beta1 ** state['step'])
                    denom = exp_avg_sq.sqrt().add_(group['eps'])
                    p.data.addcdiv_(-step_size, exp_avg, denom)
                else:
                    step_size = group['lr'] / (1 - beta1 ** state['step'])
                    p.data.add_(-step_size, exp_avg)

        return loss


class ActiveSGD(torch.optim.SGD):
    def __init__(self, params, lr,
                 momentum=0, dampening=0, weight_decay=0, nesterov=False):
        params = list(params)
        assert not isinstance(
            params[0], dict), 'Only a single param group is supported'
        super().__init__(params, lr, momentum, dampening, weight_decay, nesterov)

    def step(self, active_params=None, closure=None):
        """Performs a single optimization step.

        Arguments:
            active_params (iterable | None):
                An iterable containing parameters to be updated by 
                this optimization step
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        lr = self.param_groups[0]['lr']
        weight_decay = self.param_groups[0]['weight_decay']
        momentum = self.param_groups[0]['momentum']
        dampening = self.param_groups[0]['dampening']
        nesterov = self.param_groups[0]['nesterov']

        params = active_params if active_params is not None \
            else self.param_groups[0]['params']

        for p in params:
            if p.grad is None:
                continue
            d_p = p.grad.data
            if weight_decay != 0:
                d_p.add_(weight_decay, p.data)
            if momentum != 0:
                param_state = self.state[p]
                if 'momentum_buffer' not in param_state:
                    buf = param_state['momentum_buffer'] = torch.zeros_like(
                        p.data)
                    buf.mul_(momentum).add_(d_p)
                else:
                    buf = param_state['momentum_buffer']
                    buf.mul_(momentum).add_(1 - dampening, d_p)
                if nesterov:
                    d_p = d_p.add(momentum, buf)
                else:
                    d_p = buf

            p.data.add_(-lr, d_p)

        return loss


def cosine_schedule(max_value, min_value, period_init=10, period_mult=2, n=1000):
    """ Generator that produces cosine learning rate schedule,
        as defined in Loshchilov & Hutter, 2017, https://arxiv.org/abs/1608.03983
    Arguments:
        max_value (float): maximum value
        min_value (float): minimum value
        period_init (int): intial learning rate restart period
        period_mult (int): period multiplier that is applied at each restart
        n (int): number of iterations
    Yield:
        learning rate
    """
    i = 0
    epoch = 0
    period = period_init
    while i < n:
        lr = min_value + (max_value - min_value) * \
            (1 + math.cos(math.pi * epoch / period)) / 2
        yield lr
        i += 1
        epoch += 1
        if epoch % period == 0:
            period *= period_mult
            epoch = 0
