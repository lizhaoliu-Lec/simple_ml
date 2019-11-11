import numpy as np

__all__ = [
    # original class
    'MeanSquareLoss', 'CrossEntropy', 'LogLikelihoodLoss',
    'MeanAbsoluteLoss', 'HuberLoss', 'BinaryCrossEntropy',
    'HingeLoss', 'ExponentialLoss',

    # alias
    'mse', 'ce', 'MSE', 'CE', 'MAE', 'mae',
    'hb', 'HB', 'bce', 'BCE', 'hl', 'HL', 'el', 'EL',

    # factory interface
    'get_loss'
]


class Loss(object):
    """
    Base Loss.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        raise NotImplementedError

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        raise NotImplementedError


class MeanSquareLoss(Loss):
    """
    calculate the MSE
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        batch_size = y.shape[0]
        y_hat = np.reshape(y_hat, (batch_size, -1))
        y = np.reshape(y, (batch_size, -1))
        return 0.5 * np.sum(np.mean((y_hat - y) ** 2, axis=-1), axis=0)

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        return y_hat - y


class MeanAbsoluteLoss(Loss):
    """
    calculate the MAE
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        batch_size = y.shape[0]
        y_hat = np.reshape(y_hat, (batch_size, -1))
        y = np.reshape(y, (batch_size, -1))
        return np.sum(np.mean(np.abs(y_hat - y), axis=-1), axis=0)

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        gradient = (y_hat > y).astype(int) - (y_hat < y).astype(int)
        return np.array(gradient, dtype=float)


class HuberLoss(Loss):
    """
    calculate the huber loss. A combination of MAE loss and MSE loss.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        batch_size = y.shape[0]
        y_hat = np.reshape(y_hat, (batch_size, -1))
        y = np.reshape(y, (batch_size, -1))
        abs_diff = np.abs(y_hat - y)
        inner_part = (abs_diff <= 1).astype(int)
        outer_part = (abs_diff > 1).astype(int)
        diff = inner_part * 0.5 * abs_diff ** 2 + outer_part * (abs_diff - 0.5)

        return np.sum(np.mean(diff, axis=-1), axis=0)

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        abs_diff = np.abs(y_hat - y)
        inner_part = (abs_diff <= 1).astype(int)
        outer_part = (abs_diff > 1).astype(int)
        gradient = inner_part * (y_hat - y) + outer_part * ((y_hat > y).astype(int) - (y_hat < y).astype(int))
        return np.array(gradient, dtype=float)


class LogLikelihoodLoss(Loss):
    """
        Multi class CrossEntropy.
        # TODO fix the connection without softmax.
        # for now, softmax layer's back prop is identical value
        # because this loss function do the job for him.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        y = y.astype(int).squeeze()
        y_hat = _cut_off(y_hat)
        log_probs = np.log(y_hat)
        batch_size = y_hat.shape[0]

        return -np.sum(log_probs[np.arange(batch_size), y], axis=0)

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        y = y.astype(int)
        y_hat = _cut_off(y_hat)
        batch_size = y_hat.shape[0]
        y_hat[np.arange(batch_size), y] -= 1
        return y_hat


class BinaryCrossEntropy(Loss):
    """
    Binary CrossEntropy.
    for binary classification.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        y = y.astype(int).squeeze()
        y_hat = _cut_off(y_hat)
        y = np.reshape(y, (-1, 1))
        return -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat))

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        y_hat = _cut_off(y_hat)
        return (y_hat - y) / ((1 - y_hat) * y_hat)


class HingeLoss(Loss):
    """
    Hinge Loss, for binary svm classification.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        y = np.reshape(y, (-1,)).astype(int).squeeze()
        y_hat = np.reshape(y_hat, (-1,))
        z = y_hat * y
        return np.sum(np.maximum(0, 1 - z))

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        z = y_hat * y
        return np.array(-1 * (z <= 1).astype(int) * y, dtype=np.float)


class ExponentialLoss(Loss):
    """
    A approximation to hinge loss.
    # for now, it is not stable.
    """

    @staticmethod
    def forward(y_hat: np.array, y: np.array):
        y_hat = _chop_off(y_hat)
        z = y_hat * y
        return np.sum(np.exp(- z))

    @staticmethod
    def backward(y_hat: np.array, y: np.array):
        y_hat = _chop_off(y_hat)
        z = y_hat * y
        return np.array(-1 * np.exp(-z) * y, dtype=np.float)


# alias
mse = MSE = MeanSquareLoss
mae = MAE = MeanAbsoluteLoss
ce = CE = CrossEntropy = LogLikelihoodLoss
bce = BCE = BinaryCrossEntropy
hb = HB = HuberLoss
hl = HL = HingeLoss
el = EL = ExponentialLoss

cut_off = 1e-12
chop_off = 3


def _cut_off(z):
    return np.clip(z, cut_off, 1 - cut_off)


def _chop_off(z):
    return np.clip(z, -chop_off, chop_off)


_loss_map = {
    'mse': MeanSquareLoss,
    'mae': MeanAbsoluteLoss,
    'ce': CrossEntropy,
    'hb': HuberLoss,
    'bce': BinaryCrossEntropy,
    'hl': HingeLoss,
    'el': ExponentialLoss,
}


def get_loss(loss):
    if isinstance(loss, Loss):
        return loss
    elif isinstance(loss, str):
        original_loss = loss
        loss = loss.lower()
        if loss in _loss_map:
            return _loss_map[loss]()
        else:
            raise ValueError('Unknown loss name `{}`'.format(original_loss))
    else:
        raise ValueError('Unknown loss type `{}`'.format(loss.__class__.__name__))
