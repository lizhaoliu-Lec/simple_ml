from .ops import sigmoid, delta_sigmoid
from .ops import relu, delta_relu
from .ops import identity, delta_identity
from .ops import softmax, delta_softmax

__all__ = [
    # original class
    'Sigmoid', 'Relu', 'Identity',

    # alias
    'Linear',

    # factory interface
    'get_activation'
]


class Activation(object):
    def forward(self, x, *args, **kwargs):
        raise NotImplementedError

    def backward(self, x, *args, **kwargs):
        raise NotImplementedError


class Sigmoid(Activation):
    def forward(self, x, *args, **kwargs):
        return sigmoid(x)

    def backward(self, x, *args, **kwargs):
        return delta_sigmoid(x)


class Relu(Activation):
    def __init__(self, alpha=0., max_value=None):
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, x, *args, **kwargs):
        return relu(x, self.alpha, self.max_value)

    def backward(self, x, *args, **kwargs):
        return delta_relu(x, self.alpha, self.max_value)


class Identity(Activation):
    def forward(self, x, *args, **kwargs):
        return identity(x)

    def backward(self, x, *args, **kwargs):
        return delta_identity(x)


# alias
Linear = Identity


class Softmax(Activation):
    def forward(self, z, *args, **kwargs):
        return softmax(z)

    def backward(self, z, *args, **kwargs):
        return delta_softmax(z)


_activation_map = {
    'sigmoid': Sigmoid,
    'relu': Relu,
    'linear': Linear,
    'identity': Identity,
    'softmax': Softmax
}


def get_activation(activation):
    if activation is None:
        return Identity()
    elif isinstance(activation, str):
        original_activation = activation
        activation = activation.lower()
        if activation in _activation_map:
            return _activation_map[activation]()
        else:
            raise ValueError('Unknown activation name `{}`'.format(original_activation))
    elif isinstance(activation, Activation):
        return activation
    else:
        raise ValueError('Unknown activation type `{}`'.format(activation.__class__.__name__))
