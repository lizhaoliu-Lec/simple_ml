import numpy as np

from .ops import l2, delta_l2
from .ops import l1, delta_l1

__all__ = [
    # original class
    'L2', 'Zero', 'L1', 'L1L2',
    # alias
    'L2_Regularizer', 'L1_Regularizer', 'L1L2_Regularizer',

    # factory interface
    'get_regularizer'
]


class Regularizer(object):
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, x, *args, **kwargs):
        raise NotImplementedError


class Zero(Regularizer):
    def forward(self, x, *args, **kwargs):
        return 0

    def backward(self, x, *args, **kwargs):
        return np.zeros_like(x)


class L2(Regularizer):
    def __init__(self, weight_decay=1e-3):
        self.weight_decay = weight_decay

    def forward(self, x, *args, **kwargs):
        return l2(x, self.weight_decay)

    def backward(self, x, *args, **kwargs):
        return delta_l2(x, self.weight_decay)


L2_Regularizer = L2


class L1(Regularizer):
    def __init__(self, weight_decay=1e-3):
        self.weight_decay = weight_decay

    def forward(self, x, *args, **kwargs):
        return l1(x, self.weight_decay)

    def backward(self, x, *args, **kwargs):
        return delta_l1(x, self.weight_decay)


L1_Regularizer = L1


class L1L2(L1, L2):
    def __init__(self, l1=1e-3, l2=1e-3):
        L1.__init__(self, weight_decay=l1)
        L2.__init__(self, weight_decay=l2)

    def forward(self, x, *args, **kwargs):
        l1 = L1.forward(self, x, *args, **kwargs)
        l2 = L2.forward(self, x, *args, **kwargs)
        return 0.5 * (l1 + l2)

    def backward(self, x, *args, **kwargs):
        l1 = L1.backward(self, x, *args, **kwargs)
        l2 = L2.backward(self, x, *args, **kwargs)
        return 0.5 * (l1 + l2)


L1L2_Regularizer = L1L2

_activation_map = {
    'l2': L2,
    'l2_regularizer': L2,
    'l1': L1,
    'l1_regularizer': L1,
    'l1l2': L1L2,
    'l1l2_regularizer': L1L2,
    'l1_l2': L1L2,
    'l1_l2_regularizer': L1L2,

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
