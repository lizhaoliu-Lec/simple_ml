from .ops import sigmoid, delta_sigmoid
from .ops import relu, delta_relu
from .ops import swish, delta_swish
from .ops import identity, delta_identity
from .ops import softmax, delta_softmax
from .ops import softplus, delta_softplus
from .ops import tanh, delta_tanh
from .ops import mish, delta_mish

__all__ = [
    # original class
    'Sigmoid', 'Relu', 'Identity',
    'Swish', 'Softplus', 'Tanh',
    'Mish', 'Relu6', 'LeakyRelu',

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
    def __init__(self, alpha=None, max_value=None):
        self.alpha = alpha
        self.max_value = max_value

    def forward(self, x, *args, **kwargs):
        return relu(x, self.alpha, self.max_value)

    def backward(self, x, *args, **kwargs):
        return delta_relu(x, self.alpha, self.max_value)


class Relu6(Relu):
    def __init__(self, alpha=None):
        super(Relu6, self).__init__(alpha=alpha, max_value=6)


class LeakyRelu(Relu):
    def __init__(self, alpha=0.1, max_value=None):
        super(LeakyRelu, self).__init__(alpha=alpha, max_value=max_value)


class Swish(Activation):
    def __init__(self, beta=1.):
        self.beta = beta

    def forward(self, x, *args, **kwargs):
        return swish(x, self.beta)

    def backward(self, x, *args, **kwargs):
        return delta_swish(x, self.beta)


class Identity(Activation):
    def forward(self, x, *args, **kwargs):
        return identity(x)

    def backward(self, x, *args, **kwargs):
        return delta_identity(x)


# alias
Linear = Identity


class Softplus(Activation):

    def forward(self, x, *args, **kwargs):
        return softplus(x)

    def backward(self, x, *args, **kwargs):
        return delta_softplus(x)


class Tanh(Activation):

    def forward(self, x, *args, **kwargs):
        return tanh(x)

    def backward(self, x, *args, **kwargs):
        return delta_tanh(x)


class Mish(Activation):

    def forward(self, x, *args, **kwargs):
        return mish(x)

    def backward(self, x, *args, **kwargs):
        return delta_mish(x)


class Softmax(Activation):
    def forward(self, z, *args, **kwargs):
        return softmax(z)

    def backward(self, z, *args, **kwargs):
        return delta_softmax(z)


_activation_map = {
    'sigmoid': Sigmoid,
    'relu': Relu,
    'swish': Swish,
    'linear': Linear,
    'identity': Identity,
    'softmax': Softmax,
    'softplus': Softplus,
    'tanh': Tanh,
    'mish': Mish,
    'relu6': Relu6,
    'leaky_relu': LeakyRelu,
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
