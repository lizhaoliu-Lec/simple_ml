import numpy as np

__all__ = [
    # original name
    'xavier_uniform_initializer', 'default_weight_initializer', 'ones', 'zeros',

    # alias
    'glorot_uniform_initializer',

    # factory interface
    'get_initializer'
]


def xavier_uniform_initializer(shape):
    m = shape[0]
    n = shape[1] if len(shape) > 1 else shape[0]
    bound = np.sqrt(6. / (m + n))
    out = np.random.uniform(-bound, bound, shape)
    return out


# alias
glorot_uniform_initializer = xavier_uniform_initializer


def default_weight_initializer(shape):
    m = shape[0]
    n = shape[1] if len(shape) > 1 else shape[0]
    return np.random.randn(m, n) / np.sqrt(n)


def zeros(shape):
    return np.zeros(shape)


def ones(shape):
    return np.ones(shape)


_initializer_map = {
    'xavier_uniform_initializer': xavier_uniform_initializer,
    'glorot_uniform_initializer': glorot_uniform_initializer,
    'default_weight_initializer': default_weight_initializer,
    'zeros': zeros,
    'ones': ones,
    'random': default_weight_initializer,
}


def get_initializer(initializer):
    if isinstance(initializer, str):
        original_initializer = initializer
        initializer = initializer.lower()

        if initializer in _initializer_map:
            return _initializer_map[initializer]
        else:
            raise ValueError('Unknown initializer name `{}`'.format(original_initializer))
    elif isinstance(initializer, type(lambda k: k)):
        return initializer
    else:
        raise ValueError('Unknown initializer type `{}`'.format(initializer.__name__))
