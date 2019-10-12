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
def sigmoid(z):
    z = np.asarray(z)
    return 1. / (1 + np.exp(-z))


def delta_sigmoid(z):
    z = np.asarray(z)
    cache = sigmoid(z)
    return cache * (1. - cache)


########
# relu #
########
def relu(z, alpha=None, max_value=None):
    z = np.asarray(z)
    x = np.maximum(z, 0)
    if max_value is not None:
        x = np.clip(x, 0, max_value)
    if alpha is not None:
        neg_mask = (z < 0).astype(int)
        x += alpha * z * neg_mask
    return x


def delta_relu(z, alpha=None, max_value=None):
    z = np.asarray(z)
    x = (z >= 0).astype(int)
    if alpha is not None:
        neg = alpha * (z < 0).astype(float)
        x = x + neg
    if max_value is not None:
        pos_big = (z > max_value).astype(int)
        x = x - pos_big
    return x


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
    # z = np.asarray(z)
    # if len(z.shape) > 1:
    #     z -= np.sum(z, axis=1).reshape([z.shape[0], 1])
    #     z = np.exp(z)
    #     z /= np.sum(z, axis=1).reshape([z.shape[0], 1])
    #     return z
    # else:
    #     z -= np.max(z)
    #     z = np.exp(z)
    #     z /= np.sum(z)
    #     return z
    z = np.asanyarray(z)
    # shift
    z -= np.max(z, axis=-1, keepdims=True)
    z = np.exp(z)
    z /= np.sum(z, axis=-1, keepdims=True)
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


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    x = np.linspace(start=-10, stop=10, num=500)
    y = relu(x, alpha=0.1, max_value=6)
    dy = delta_relu(x, alpha=0.1, max_value=6)
    plt.plot(x, y)
    plt.plot(x, dy)
    plt.show()
