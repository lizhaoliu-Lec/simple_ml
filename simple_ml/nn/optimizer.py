import numpy as np

__all__ = [
    # original class
    'SGD', 'Momentum',

    # alias
    'sgd', 'momentum',

    # factory interface
    'get_optimizer'
]


class Optimizer(object):
    def __init__(self,
                 lr: float = 1e-3, decay: float = 0.,
                 grad_clip: float = 0,
                 lr_min: float = 0.,
                 lr_max: float = np.inf):
        self.lr = lr
        self.decay = decay
        self.clip = grad_clip
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.iterations = 0

    def update(self):
        self.iterations += 1
        self.lr *= 1. / (1 + self.decay * self.iterations)
        self.lr = np.clip(self.lr, self.lr_min, self.lr_max)

    def minimize(self, params, grads):
        raise NotImplementedError

    def maximize(self, params, grads):
        raise NotImplementedError


class SGD(Optimizer):
    def __init__(self, *args, **kwargs):
        super(SGD, self).__init__(*args, **kwargs)

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            p -= self.lr * _grad_clip(g, self.clip)
        self.update()

    def maximize(self, params, grads):
        for p, g in zip(params, grads):
            p += self.lr * _grad_clip(g, self.clip)
        self.update()


class Momentum(Optimizer):
    """
    Performs stochastic gradient descent with momentum.

    momentum: Scalar between 0 and 1 giving the momentum value.
              Setting momentum = 0 reduces to sgd.
    velocity: A numpy array of the same shape as w and dw used to store a moving
              average of the gradients.
    """

    def __init__(self, momentum=0.9, nesterov=False, *args, **kwargs):
        super(Momentum, self).__init__(*args, **kwargs)
        self.momentum = momentum
        self.nesterov = nesterov
        self.velocity = dict()

    def minimize(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            if self.nesterov:
                p = p + self.momentum * v - self.lr * g
            else:
                p = p + v
            self.velocity[id(p)] = v
        self.update()

    def maximize(self, params, grads):
        for p, g in zip(params, grads):
            v = self.velocity.get(id(p), np.zeros_like(p))
            v = self.momentum * v - self.lr * g
            p -= v
            self.velocity[id(p)] = p
        self.update()


sgd = SGD
momentum = Momentum


def _grad_clip(grad, clip):
    if clip > 0:
        return np.clip(grad, -clip, clip)
    else:
        return grad


_optimizer_map = {
    'sgd': SGD,
    'momentum': Momentum,
}


def get_optimizer(optimizer):
    if isinstance(optimizer, Optimizer):
        return optimizer
    elif isinstance(optimizer, str):
        original_optimizer = optimizer
        optimizer = optimizer.lower()
        if optimizer in _optimizer_map:
            return _optimizer_map[optimizer]()
        else:
            raise ValueError('Unknown optimizer name `{}`'.format(original_optimizer))
    else:
        raise ValueError('Unknown optimizer type `{}`'.format(optimizer.__class__.__name__))
