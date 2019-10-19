import numpy as np

from .ops import l2, delta_l2

__all__ = [
    # original class
    'L2', 'Zero',
    # alias
    'L2_Regularizer',

    # factory interface
    'get_regularizer'
]


class Regularizer(object):
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, x, *args, **kwargs):
        raise NotImplementedError


class L2(Regularizer):
    def __init__(self, weight_decay=1e-3):
        self.weight_decay = weight_decay

    def forward(self, x, *args, **kwargs):
        return l2(x, self.weight_decay)

    def backward(self, x, *args, **kwargs):
        return delta_l2(x, self.weight_decay)


L2_Regularizer = L2


class Zero(Regularizer):
    def forward(self, x, *args, **kwargs):
        return 0

    def backward(self, x, *args, **kwargs):
        return np.zeros_like(x)


_activation_map = {
    'l2': L2,

}


def get_regularizer(regularizer):
    if regularizer is None:
        return Zero()
    elif isinstance(regularizer, str):
        original_regularizer = regularizer
        regularizer = regularizer.lower()
        if regularizer in _activation_map:
            return _activation_map[regularizer]()
        else:
            raise ValueError('Unknown activation name `{}`'.format(original_regularizer))
    elif isinstance(regularizer, Regularizer):
        return regularizer
    else:
        raise ValueError('Unknown regularizer type `{}`'.format(regularizer.__class__.__name__))
