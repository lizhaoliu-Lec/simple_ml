import numpy as np

__all__ = [
    'sigmoid', 'delta_sigmoid',
    'relu', 'delta_relu',
    'swish', 'delta_swish',
    'identity', 'delta_identity',
    'softmax', 'delta_softmax',
    'softplus', 'delta_softplus',
    'tanh', 'delta_tanh',
    'mish', 'delta_mish',
]


###########
# sigmoid #
###########
def sigmoid(x):
    x = np.asarray(x)
    return 1. / (1 + np.exp(-x))


def delta_sigmoid(x):
    x = np.asarray(x)
    cache = sigmoid(x)
    return cache * (1. - cache)


########
# relu #
########
def relu(z, alpha=0., max_value=None):
    z = np.asarray(z)
    x = np.maximum(z, 0)
    if max_value is not None:
        x = np.clip(x, 0, max_value)
    if alpha != 0.:
        negative_part = np.maximum(-z, 0)
        x -= alpha * negative_part
    return x


def delta_relu(z, alpha=0., max_value=None):
    z = np.asarray(z)
    if max_value is not None:
        return (np.where(z <= max_value, z, -1e-6 * z) >= 0).astype(int) \
               + alpha * (z < 0).astype(int)
    else:
        return (z >= 0).astype(int) + alpha * (z < 0).astype(int)


#########
# swish #
#########
def swish(z, beta=1.):
    return z * sigmoid(beta * z)


def delta_swish(z, beta=1.):
    cache = sigmoid(beta * z)
    return cache + z * beta * cache * (1 - cache)


def identity(z):
    z = np.asarray(z)
    return z


def delta_identity(z):
    z = np.asarray(z)
    return np.ones(z.shape)


###########
# softmax #
###########
def softmax(z):
    z = np.asarray(z)
    if len(z.shape) > 1:
        z -= np.sum(z, axis=1).reshape([z.shape[0], 1])
        z = np.exp(z)
        z /= np.sum(z, axis=1).reshape([z.shape[0], 1])
        return z
    else:
        z -= np.max(z)
        z = np.exp(z)
        z /= np.sum(z)
        return z


def delta_softmax(z):
    z = np.asarray(z)
    return np.ones(z.shape, dtype=z.dtype)


############
# softplus #
############
def softplus(z):
    return np.log(1 + np.exp(z))


def delta_softplus(z):
    return sigmoid(z)


########
# tanh #
########
def tanh(z):
    z = np.asarray(z)
    return np.tanh(z)


def delta_tanh(z):
    z = np.asarray(z)
    return 1 - np.power(np.tanh(z), 2)


########
# mish #
########
def mish(z):
    return tanh(softplus(z))


def delta_mish(z):
    return delta_tanh(softplus(z)) * delta_softplus(z)
